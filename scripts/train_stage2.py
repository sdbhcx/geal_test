"""
Stage 2 Training Script (3D Affordance Alignment)

This stage distills knowledge from the pretrained 2D Branch
into the 3D branch, aligning multi-view 2D representations with 3D features.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from utils.utils import seed_torch, read_yaml
from utils.logger import setup_logger
from utils.metrics import evaluating, cal_SIM_3d

from dataset.laso import LasoDataset
from dataset.piad import PiadDataset
from model.branch_2d import Branch2D
from model.branch_3d import Branch3D
from model.soft_prior_backprojector import SoftPriorBackprojector
from utils.loss import HM_Loss

# §9 Lightweight Pipeline imports (isolated from baseline path)
try:
    from utils.affordance_loss import (
        loss_invariant,
        loss_3d2img,
        loss_contrastive,
        build_projectors,
    )
    _HAS_LIGHTWEIGHT = True
except ImportError:
    _HAS_LIGHTWEIGHT = False

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def count_trainable_params(model):
    """Print number of trainable parameters per submodule."""
    for name, module in model.named_children():
        if any(p.requires_grad for p in module.parameters()):
            num = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"Module: {name:20s} | Trainable params: {num / 1e6:.2f}M")


def build_dataloader(cfg):
    """Initialize train/test dataloaders."""
    ds_cfg = {
        "use_image": cfg.get("use_image", False),
        "img_size": cfg.get("img_size", 224),
        "k_images": cfg.get("k_images", 1),
    }
    if cfg["category"] == "piad":
        train_dataset = PiadDataset(
            cfg["train_split"], cfg["setting"], data_root=cfg["data_root"], **ds_cfg
        )
        test_dataset = PiadDataset(
            cfg["test_split"], data_root=cfg["data_root"], **ds_cfg
        )
    elif cfg["category"] == "laso":
        train_dataset = LasoDataset(cfg["train_split"], cfg["setting"], data_root=cfg["data_root"])
        test_dataset = LasoDataset(cfg["test_split"], data_root=cfg["data_root"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
    )
    return train_loader, test_loader


def build_optimizer(model_3d, opt_cfg, model_2d=None):
    """Set up optimizer and scheduler for the 3D branch and AffordanceProj."""
    param_dicts = [
        {"params": [p for n, p in model_3d.named_parameters()
                    if "text_encoder" not in n and p.requires_grad]},
        {"params": [p for n, p in model_3d.named_parameters()
                    if "text_encoder" in n and p.requires_grad],
         "lr": opt_cfg["tlr"]},
    ]
    if model_2d is not None and hasattr(model_2d, "affordance_proj"):
        param_dicts.append({
            "params": model_2d.affordance_proj.parameters(),
            "lr": opt_cfg["lr"],
        })

    optimizer = torch.optim.Adam(
        params=param_dicts,
        lr=opt_cfg["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=opt_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=opt_cfg["step_size"], gamma=opt_cfg["gamma"]
    )
    return optimizer, scheduler


# ---------------------------------------------------------------------
# Training and Evaluation Loops
# ---------------------------------------------------------------------

def train_one_epoch(model_3d, model_2d, loader, optimizer, device, criterion_hm,
                    logger, epoch, train_cfg, img_3d_proj=None, anchor_proj=None,
                    soft_prior_bp=None):
    """
    One training epoch.

    Baseline path (no lightweight pipeline):
      - Freeze 2D branch (teacher)
      - Compute 3D affordance heatmaps and alignment loss

    §9 Lightweight pipeline (when train_cfg["use_new_losses"] is True):
      - L_invariant: cross-image consistency via AffordanceProj
      - L_3d2img: 3D→2D alignment (the only gradient source for the 3D branch)
      - L_contrastive: optional per-point contrastive loss

    Args:
        img_3d_proj: nn.Linear from AffordanceProj output dim to 3D project_dim.
        anchor_proj: nn.Linear from DINO dim to AffordanceProj output dim.
    """
    model_3d.train()
    model_2d.eval()
    loss_sum = 0

    for _, p in model_2d.named_parameters():
        p.requires_grad = False

    # ── §9 Lightweight Pipeline ──────────────────────────────────────────
    use_lw = train_cfg.get("use_new_losses", False) and _HAS_LIGHTWEIGHT
    if use_lw and hasattr(model_2d, "affordance_proj"):
        for p in model_2d.affordance_proj.parameters():
            p.requires_grad = True

    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        point, label = batch[0].to(device), batch[5].to(device)

        if use_lw:
            # Lightweight batch: (point, class_id, binary_mask, questions,
            #                      affordance_id, label, imgs)
            images = batch[6].to(device)          # [B, k, 3, H, W] or [B, 3, H, W]
            # Normalise to [B, k, 3, H, W] for uniform handling
            if images.dim() == 4:
                images = images.unsqueeze(1)

            B, k = images.shape[0], images.shape[1]

            # 3D forward
            question = batch[3]
            pred_3d, feat_3d = model_3d(question, point)  # [B, N], [B, N, 64]
            # feat_3d is already [B, 2048, 64] from Branch3D feature_downsampler,
            # matching loss_3d2img / loss_contrastive expectations (N=2048)

            # 2D forward (baseline path) — always called to compute loss_kld
            feat_2d, render_feats = model_2d(question, point, feat_3d)

            # Per-image AffordanceProj forward
            z_imgs, dino_anchors = [], []
            for j in range(k):
                with torch.no_grad():
                    _, global_feat = model_2d.get_raw_dino_features(images[:, j])
                z = F.normalize(model_2d.affordance_proj(global_feat), dim=-1)
                z_imgs.append(z)
                dino_anchors.append(F.normalize(global_feat, dim=-1))

            z_img_mean = torch.stack(z_imgs).mean(0)  # [B, affordance_dim]

            # Base heatmap + KLD losses (always present; matches baseline)
            loss_hm = criterion_hm(pred_3d, label)
            loss_kld = nn.MSELoss()(render_feats, feat_2d)
            loss = loss_hm + train_cfg.get("kl_loss_weight", 0.0) * loss_kld

            # §9 losses stacked on top
            if train_cfg.get("use_invariant_loss", False):
                lw_inv = loss_invariant(z_imgs,
                                        anchor=dino_anchors,
                                        anchor_proj=anchor_proj)
                loss += train_cfg.get("inv_loss_weight", 0.1) * lw_inv
            if train_cfg.get("use_3d2img_loss", False):
                lw_3d2img = loss_3d2img(z_img_mean, pred_3d, feat_3d,
                                         label, img_3d_proj=img_3d_proj)
                loss += train_cfg.get("loss_3d2img_weight", 0.5) * lw_3d2img
            if train_cfg.get("use_contrastive_loss", False):
                z_img_stacked = torch.stack(z_imgs)  # [B, k, affordance_dim]
                lw_cont = loss_contrastive(z_img_stacked, feat_3d,
                                           label, img_3d_proj=img_3d_proj)
                loss += train_cfg.get("contrastive_loss_weight", 0.1) * lw_cont

        else:
            # ── Baseline path ───────────────────────────────────────────
            question = batch[3]

            if soft_prior_bp is not None:
                # --- V2: 2D→3D soft-prior injection ---
                # 1) Pure-3D forward to obtain the features the 2D renderer splats.
                pred_3d, feat_3d = model_3d(question, point)

                # 2) Frozen 2D teacher: render + decode affordance map, and pull
                #    back the splatting correspondences needed for backprojection.
                feat_2d, render_feats, attn_map, render_idx, rendered_contrib = \
                    model_2d(question, point, feat_3d, return_affordance_map=True)

                # 3) Backproject the 2D prior onto the point cloud.
                num_points = point.shape[-1]
                q, u = soft_prior_bp(attn_map, render_idx, rendered_contrib,
                                     num_points=num_points)
                soft_prior = torch.stack([q, u], dim=1)   # [B, 2, N]

                # 4) Re-run the 3D branch WITH the injected prior → final outputs.
                pred_3d, feat_3d = model_3d(question, point, soft_prior=soft_prior)
            else:
                pred_3d, feat_3d = model_3d(question, point)
                feat_2d, render_feats = model_2d(question, point, feat_3d)

            loss_kld = nn.MSELoss()(render_feats, feat_2d)
            loss_hm = criterion_hm(pred_3d, label)
            loss = loss_hm + train_cfg["kl_loss_weight"] * loss_kld

        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        if i % 10 == 0:
            if use_lw:
                logger.debug(
                    f"[Epoch {epoch}] Iter {i}/{len(loader)} | Loss: {loss.item():.4f} "
                    f"(hm: {loss_hm.item():.4f}, mse: {loss_kld.item():.4f}, "
                    f"inv: {lw_inv.item():.4f}, 3d2img: {lw_3d2img.item():.4f})"
                )
            else:
                logger.debug(
                    f"[Epoch {epoch}] Iter {i}/{len(loader)} | Loss: {loss.item():.4f} "
                    f"(hm: {loss_hm.item():.4f}, mse: {loss_kld.item():.4f})"
                )

    # Gate diagnostics (V1 tokenizer): log per-epoch gate mean / std / saturation
    if (
        hasattr(model_3d, "token_fusion")
        and model_3d.token_fusion is not None
        and model_3d.token_fusion.gate_value is not None
    ):
        gv = model_3d.token_fusion.gate_value.detach().cpu()
        numel = gv.numel()
        sat0 = (gv < 0.01).sum().item() / numel
        sat1 = (gv > 0.99).sum().item() / numel
        logger.debug(
            f"[Epoch {epoch}] Gate stats | "
            f"mean={gv.mean():.4f} std={gv.std():.4f} "
            f"min={gv.min():.4f} max={gv.max():.4f} "
            f"saturated_0={sat0:.2%} saturated_1={sat1:.2%} "
            f"residual_scale={model_3d.token_fusion.residual_scale.item():.4f}"
        )

    return loss_sum / len(loader)


def evaluate(model_3d, loader, device, criterion_hm, logger):
    """
    Validation loop:
      - Computes IOU, SIM, MAE, and AUC across all test samples.
    """
    model_3d.eval()
    results, targets = [], []
    total_mae, total_points = 0, 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            # index-based: works for both 6-tuple (baseline) and 7-tuple (use_image)
            point, label = batch[0].to(device), batch[5].to(device)
            question = batch[3]
            pred = model_3d(question, point)

            val_loss = criterion_hm(pred, label)
            mae, n_pts = evaluating(pred, label)
            total_mae += mae.item()
            total_points += n_pts

            # 按样本展开，避免不同 batch 形状不一致导致 np.array 失败
            results.extend(list(pred.cpu().numpy()))
            targets.extend(list(label.cpu().numpy()))

            logger.debug(f"[Val] Batch {i}/{len(loader)} | Loss: {val_loss.item():.4f}")

    mean_mae = total_mae / total_points

    # Compute similarity and AUC/IOU
    sim_scores = np.array([cal_SIM_3d(r, t) for r, t in zip(results, targets)])
    SIM = np.nanmean(sim_scores)

    IOUs, AUCs = [], []
    IOU_thres = np.linspace(0, 1, 20)

    for t_true, p_score in zip(targets, results):
        t_true = (t_true >= 0.5).astype(int)     # 逐样本二值化
        if np.sum(t_true) == 0:
            continue
        auc = roc_auc_score(t_true.flatten(), p_score.flatten())
        AUCs.append(auc)
        temp_iou = []
        for thr in IOU_thres:
            p_mask = (p_score >= thr).astype(int)
            intersect = np.sum(p_mask & t_true)
            union = np.sum(p_mask | t_true)
            temp_iou.append(intersect / (union + 1e-6))
        IOUs.append(np.mean(temp_iou))

    IOU = np.nanmean(IOUs)
    AUC = np.nanmean(AUCs)
    logger.debug(f"Validation → IOU: {IOU:.4f}, AUC: {AUC:.4f}, SIM: {SIM:.4f}, MAE: {mean_mae:.4f}")
    return IOU, mean_mae


# ---------------------------------------------------------------------
# Main Training Entry
# ---------------------------------------------------------------------

def main(cfg_path="config/train_stage2.yaml"):
    """
    Stage 2: 3D Affordance Alignment Training
    """
    cfg = read_yaml(cfg_path)
    train_cfg = cfg["train"]

    # Select device
    gpu_id = str(train_cfg.get("gpu", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"[INFO] Using GPU {gpu_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_torch(train_cfg["seed"])
    logger, sign = setup_logger(cfg)

    # Dataset config includes lightweight pipeline image params
    ds_cfg = {**cfg["dataset"], "batch_size": train_cfg["batch_size"]}
    ds_cfg["use_image"] = ds_cfg.get("use_image", False)
    ds_cfg["k_images"] = ds_cfg.get("k_images", 1)
    ds_cfg["img_size"] = ds_cfg.get("img_size", 224)
    train_loader, test_loader = build_dataloader(ds_cfg)

    # Build models
    model_2d = Branch2D(cfg["model_2d"], cfg["renderer"]).to(device)
    model_3d = Branch3D(cfg["model_3d"]).to(device)
    criterion_hm = HM_Loss().to(device)

    # ── V2 soft-prior backprojector (frozen 2D teacher → 3D points) ──
    sp_cfg = cfg["model_3d"].get("soft_prior", {})
    soft_prior_bp = None
    if sp_cfg.get("enabled", False):
        soft_prior_bp = SoftPriorBackprojector(
            mode=sp_cfg.get("mode", "vis_unc"),
            num_points=cfg["model_3d"].get("num_points",
                                           cfg["model_3d"].get("N_p", 2048)),
        ).to(device)
        logger.debug(
            f"[V2] soft-prior backprojector enabled | mode={soft_prior_bp.mode} | "
            f"num_points={soft_prior_bp.num_points}"
        )

    # Load pretrained 2D weights (frozen teacher)
    if train_cfg.get("pretrained_2d", None):
        ckpt_path = train_cfg["pretrained_2d"]
        logger.debug(f"Loading pretrained 2D model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model_2d.load_state_dict(ckpt["model"], strict=False)

    if train_cfg["resume"]:
        ckpt_path = train_cfg["checkpoint_path"]
        logger.debug(f"Resuming 3D model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model_3d.load_state_dict(ckpt["model"], strict=False)

    use_lw = train_cfg.get("use_new_losses", False) and _HAS_LIGHTWEIGHT
    model_2d_for_opt = model_2d if use_lw else None
    optimizer, scheduler = build_optimizer(
        model_3d, cfg["optimizer"], model_2d=model_2d_for_opt
    )

    # §9 Lightweight Pipeline: build projectors (img_3d_proj, anchor_proj)
    img_3d_proj, anchor_proj = None, None
    if train_cfg.get("use_new_losses", False) and _HAS_LIGHTWEIGHT:
        img_3d_proj, anchor_proj = build_projectors(
            model_2d, cfg["model_3d"], device, optimizer
        )

    # Count trainable params
    count_trainable_params(model_3d)

    # Training loop
    best_IOU = 0
    save_dir = os.path.join(train_cfg["save_dir"], train_cfg["name"])
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        logger.debug(f"Epoch {epoch} start → lr={optimizer.param_groups[0]['lr']:.6f}")

        train_loss = train_one_epoch(model_3d, model_2d, train_loader, optimizer, device,
                                     criterion_hm, logger, epoch, train_cfg,
                                     img_3d_proj=img_3d_proj, anchor_proj=anchor_proj,
                                     soft_prior_bp=soft_prior_bp)
        IOU, mae = evaluate(model_3d, test_loader, device, criterion_hm, logger)
        scheduler.step()

        if IOU > best_IOU:
            best_IOU = IOU
            model_path = os.path.join(save_dir, f"best_model_{sign}.pt")
            torch.save({
                "model": model_3d.state_dict(),
                "optimizer": optimizer.state_dict(),
                "Epoch": epoch
            }, model_path)
            logger.debug(f"New best model saved → IOU={best_IOU:.4f} | {model_path}")

    logger.debug(f"Training complete. Best IOU: {best_IOU:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_stage2.yaml")
    args = parser.parse_args()
    main(args.config)
