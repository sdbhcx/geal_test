import torch
import torch.nn as nn


class SoftPriorBackprojector(nn.Module):
    """
    V2 — 2D→3D soft-prior backprojection (GEAL-TASA 改造计划 §15).

    Maps the frozen 2D teacher's per-view affordance probability maps back onto
    the 3D point cloud, using the differentiable Gaussian-splatting
    correspondences that the renderer already produces:

      * ``render_idx``      [B, V, H, W]  (long)  — which 3D point wins each pixel
      * ``rendered_contrib``[B, V, H, W]           — per-pixel splatting weight,
                                                     used here as visibility ``a_jv``.

    For every 3D point ``p_j`` this yields a soft prior ``q_j`` (where the 2D
    teacher thinks affordance is) and an uncertainty ``u_j`` (how much the views
    disagree). Occluded / back-facing points get ``a_jv ≈ 0`` and are naturally
    down-weighted.

    Inputs (all view-major, [B, V, ...]):
        attn_map    [B, V, 1, H, W]  2D affordance probability (sigmoid, in [0, 1])
        render_idx  [B, V, H, W]      long, point index winning each pixel
        contrib     [B, V, H, W]      per-pixel contribution (visibility proxy)
        num_points  int | None        number of 3D points N (== xyz.shape[1])
        c_v         [V] | None        per-view weight (None → uniform 1/V)

    Outputs:
        q  [B, N]   visibility (or mean) weighted soft prior
        u  [B, N]   cross-view uncertainty (zeros when mode in {'mean', 'vis'})

    Implementation note (memory-safe):
        We scatter the flattened [B, V*H*W] pixel values into [B, N] buffers with
        ``scatter_add_``. We never materialize a dense [B, N, H, W] tensor, so the
        peak memory is O(V*H*W), not O(V*H*W*N) (see §15.5 risk 5).
    """

    def __init__(self, mode="vis_unc", eps=1e-6, num_points=2048):
        super().__init__()
        if mode not in ("mean", "vis", "vis_unc"):
            raise ValueError(f"unknown soft-prior mode: {mode!r}")
        self.mode = mode
        self.eps = eps
        self.num_points = int(num_points)

    def forward(self, attn_map, render_idx, contrib, num_points=None, c_v=None):
        if num_points is None:
            num_points = self.num_points
        N = int(num_points)

        B, V, _, H, W = attn_map.shape
        P = H * W

        attn = attn_map.reshape(B, V, P)                       # [B, V, P]
        idx = render_idx.reshape(B, V, P).long().clamp_(0, N - 1)
        a = contrib.reshape(B, V, P).clamp_min(0.0)            # visibility a_jv

        if c_v is None:
            c_v = attn.new_ones(V) / V
        c_v = c_v.view(1, V, 1)                                # [1, V, 1]

        # Flatten the view & pixel dims so a single scatter_add fills [B, N].
        attn_f = attn.reshape(B, V * P)                        # [B, V*P]
        w = (a * c_v).reshape(B, V * P)                        # visibility weight
        y_f = attn_f * w                                       # weighted attn
        y2_f = attn_f * attn_f * w                             # weighted attn^2
        idx_f = idx.reshape(B, V * P)                          # [B, V*P]

        Yu = torch.zeros(B, N, device=attn.device)
        Yy = torch.zeros(B, N, device=attn.device)
        Yu2 = torch.zeros(B, N, device=attn.device)
        Z = torch.zeros(B, N, device=attn.device)

        # self[b, idx_f[b,k]] += src[b,k]
        Yu.scatter_add_(1, idx_f, attn_f)
        Yy.scatter_add_(1, idx_f, y_f)
        Yu2.scatter_add_(1, idx_f, y2_f)
        Z.scatter_add_(1, idx_f, w)

        if self.mode == "mean":
            # P1: simple mean over views, visibility ignored (still 0 if not winner)
            q = Yu / V
            u = torch.zeros_like(q)
        else:
            # P2/P3: visibility-weighted mean
            q = Yy / (Z + self.eps)
            if self.mode == "vis_unc":
                # P3: contribution-weighted variance  E[a y^2]/E[a] - (E[a y]/E[a])^2
                u = (Yu2 / (Z + self.eps)) - q * q
                u = u.clamp_min_(0.0)
            else:
                u = torch.zeros_like(q)

        return q, u
