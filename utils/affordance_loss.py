"""
§9 Lightweight Pipeline: Invariant Affordance Knowledge losses.

MIFAG-inspired losses for learning 2D invariant affordance embeddings
from interaction images and aligning them with 3D point features.

All losses are gated by config flags in the training script;
none of these functions is invoked by default.

Reference: https://arxiv.org/abs/2408.13024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------

def loss_invariant(z_list, anchor=None, anchor_proj=None, eps=1e-6):
    """
    Invariance constraint: k interaction images of the same affordance
    should map to similar embeddings via the trainable AffordanceProj.

    Includes an anchor term that prevents AffordanceProj from collapsing
    to a constant vector.  ``anchor`` is a list of raw DINO global
    features (same order as z_list); each z_j is pulled toward its own
    anchor.  Since anchor dim (dino_dim) != z dim (affordance_dim),
    ``anchor_proj`` projects the anchor into the same space before
    cosine comparison.

    Args:
        z_list: list of [B, C] tensors, one per image (k items total).
        anchor: optional list of [B, C_dino] raw DINO features.
        anchor_proj: optional nn.Linear(C_dino, C) to project anchor
                     into z space.  If anchor is provided but
                     anchor_proj is None, anchor is ignored.

    Returns:
        scalar loss = mean pairwise cosine distance + anchor loss.
    """
    if len(z_list) < 2:
        return torch.tensor(0.0, device=z_list[0].device)

    # Pairwise cosine distance among k image embeddings
    loss = 0.0
    n_pairs = 0
    for a in range(len(z_list)):
        for b in range(a + 1, len(z_list)):
            loss += (1 - F.cosine_similarity(z_list[a], z_list[b], dim=-1)).mean()
            n_pairs += 1
    loss = loss / n_pairs if n_pairs > 0 else torch.tensor(0.0, device=z_list[0].device)

    # Anchor: each z_j should be close to its own raw DINO anchor.
    if anchor is not None and len(anchor) == len(z_list) and anchor_proj is not None:
        anchor_loss = 0.0
        for z, a in zip(z_list, anchor):
            a_proj = F.normalize(anchor_proj(a.detach()), dim=-1)
            anchor_loss += (1 - F.cosine_similarity(z, a_proj, dim=-1)).mean()
        anchor_loss = anchor_loss / len(z_list)
        loss = loss + anchor_loss

    return loss


def loss_3d2img(z_img, pred_3d, z_3d, label, img_3d_proj=None, eps=1e-6):
    """
    Core loss: aggregate 3D foreground feature and align it with the
    invariant affordance embedding from interaction images.

    This is the ONLY loss in the pipeline whose gradient reaches
    ``pred_3d`` (and thus the 3D branch).  L_invariant shapes
    z_img quality; L_3d2img propagates that signal to 3D.

    Args:
        z_img: [B, C_img] — invariant affordance embedding (aggregated
               from k images).  Detached to avoid conflicting gradients
               with L_invariant.
        pred_3d: [B, N] — 3D affordance logits (used to build a soft
                 foreground mask; detached).
        z_3d: [B, N, C_3d] — 3D per-point affordance features
              (downsampled_feat from Branch3D, where N matches label).
        label: [B, N] — point-level GT (0/1).
        img_3d_proj: optional nn.Linear(C_img, C_3d).  When None,
                     C_img must == C_3d.

    Returns:
        scalar mean cosine distance between foreground 3D feature
        and z_img.
    """
    # Soft foreground mask from current model prediction
    with torch.no_grad():
        soft_mask = torch.sigmoid(pred_3d.detach()).unsqueeze(-1)

    z_3d_fg = (z_3d * soft_mask).sum(1) / soft_mask.sum(1).clamp_min(eps)

    z_img_proj = F.normalize(z_img.detach(), dim=-1)
    z_3d_fg_norm = F.normalize(z_3d_fg, dim=-1)

    C_img, C_3d = z_img_proj.shape[-1], z_3d_fg_norm.shape[-1]

    if C_img != C_3d:
        if img_3d_proj is None:
            raise ValueError(
                f"z_img dim ({C_img}) != z_3d dim ({C_3d}). "
                "Pass a Linear(C_img, C_3d) as img_3d_proj."
            )
        z_img_proj = F.normalize(img_3d_proj(z_img_proj), dim=-1)

    loss = (1 - F.cosine_similarity(z_3d_fg_norm, z_img_proj, dim=-1)).mean()
    return loss


def loss_contrastive(z_img, z_3d, label, img_3d_proj=None, temp=0.1, eps=1e-6):
    """
    Contrastive loss using point-level labels to construct
    positive/negative pairs.

    For each image embedding z_img_j:
      - positive: 3D foreground points (label=1) should be close
      - negative: 3D background points (label=0) should be far

    Uses a per-point cosine similarity against z_img_j with BCE loss.

    Args:
        z_img: [B, k, C_img] — per-image embeddings.
        z_3d: [B, N, C_3d] — 3D per-point features.
        label: [B, N] — point-level GT (0/1).
        img_3d_proj: optional nn.Linear(C_img, C_3d).
        temp: temperature for cosine similarity.

    Returns:
        scalar contrastive loss.
    """
    B, k, C_img = z_img.shape
    N = z_3d.shape[1]

    z_3d_norm = F.normalize(z_3d, dim=-1)
    z_img_proj = F.normalize(z_img, dim=-1)

    C_3d = z_3d_norm.shape[-1]
    if C_img != C_3d:
        if img_3d_proj is None:
            raise ValueError(
                f"z_img dim ({C_img}) != z_3d dim ({C_3d}). "
                "Pass a Linear(C_img, C_3d) as img_3d_proj."
            )
        z_img_proj = F.normalize(img_3d_proj(z_img_proj), dim=-1)

    sim = torch.bmm(
        z_3d_norm.transpose(1, 2),
        z_img_proj.transpose(1, 2)
    ).transpose(1, 2)  # [B, k, N]
    sim_scaled = sim / temp

    fg_mask = (label > 0.5).unsqueeze(1).expand(-1, k, -1)
    loss = F.binary_cross_entropy_with_logits(sim_scaled, fg_mask.float())

    return loss


# ---------------------------------------------------------------------
# Projector factory (lifecycle management, kept out of train scripts)
# ---------------------------------------------------------------------

def build_projectors(model_2d, model_3d_cfg, device, optimizer):
    """
    Create and register the img_3d_proj and anchor_proj linear layers
    required by the §9 lightweight pipeline.

    Both projectors are initialized with Kaiming normal and registered
    as separate parameter groups on the existing optimizer so they are
    trained alongside the 3D branch without affecting other models.

    Args:
        model_2d: Branch2D instance.  Must have ``affordance_proj``
                  when use_affordance_proj is enabled.
        model_3d_cfg: dict with keys ``project_dim`` and ``dino_dim``.
        device: torch device.
        optimizer: existing optimizer (parameter groups are added).

    Returns:
        tuple: (img_3d_proj, anchor_proj) — either nn.Linear or None.
    """
    img_3d_proj = None
    anchor_proj = None

    if not hasattr(model_2d, "affordance_proj"):
        return img_3d_proj, anchor_proj

    affordance_dim = model_2d.affordance_dim
    project_dim = model_3d_cfg.get("project_dim", 64)
    dino_dim = model_3d_cfg.get("dino_dim", 768)
    lr = optimizer.defaults.get("lr", optimizer.param_groups[0]["lr"])

    # img_3d_proj: bridge image embedding space → 3D feature space
    img_3d_proj = nn.Linear(affordance_dim, project_dim).to(device)
    nn.init.kaiming_normal_(img_3d_proj.weight, nonlinearity="relu")
    optimizer.add_param_group({"params": img_3d_proj.parameters(), "lr": lr})

    # anchor_proj: bridge DINO anchor space → affordance embedding space
    anchor_proj = nn.Linear(dino_dim, affordance_dim).to(device)
    nn.init.kaiming_normal_(anchor_proj.weight, nonlinearity="relu")
    optimizer.add_param_group({"params": anchor_proj.parameters(), "lr": lr})

    return img_3d_proj, anchor_proj
