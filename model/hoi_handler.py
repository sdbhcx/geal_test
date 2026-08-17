#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HOIHandler: H_raw → H* 在线派生管线
===================================
阶段 A（numpy，不可微）: conf 过滤 → 降采样 → FPS 全局候选 → conf 加权重要性 → top-k 锚点 → 局部 patch
阶段 B（torch，可学习）: AnchorEncoder（PointNet 式 MLP + max-pool）→ H* ∈ R^{k×d_model}

与 compute_hstar.py 的关系：复用相同的 numpy 锚点选择逻辑，但 AnchorEncoder 是可学习的
（compute_hstar.py 中的 encoder 是冻结的），且本模块支持 batch 处理。
"""

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
@dataclass
class HStarCfg:
    M_ds: int = 40000
    N_fps: int = 512
    k: int = 128
    conf_floor: float = 1.0
    conf_hand_high: float = 1.5
    w_hand: float = 0.7
    w_struct: float = 0.3
    patch_r_ratio: float = 0.05
    m: int = 32
    d: int = 256               # AnchorEncoder 输出维度
    use_conf_channel: bool = True
    k_struct_nn: int = 16
    hstar_proj: int = 512      # 将 H* 从 d 投影到 d_model（cross-attn 所需）


# ----------------------------------------------------------------------------
# AnchorEncoder: 可学习的 PointNet 式 MLP + max-pool
# ----------------------------------------------------------------------------
class AnchorEncoder(nn.Module):
    """
    共享权重的局部 patch 编码器。
    输入: patches [B, k, m, in_ch]  (in_ch=4 含 xyz+conf, in_ch=3 仅 xyz)
    输出: tokens [B, k, d]
    """

    def __init__(self, d: int = 256, use_conf_channel: bool = True):
        super().__init__()
        in_ch = 4 if use_conf_channel else 3
        self.mlp = nn.Sequential(
            nn.Conv1d(in_ch, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, d, 1),
        )
        self.d = d

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: [B, k, m, in_ch]
        Returns:
            [B, k, d] 每锚点一个 token
        """
        B, k, m, _ = patches.shape
        x = patches.transpose(1, 2).contiguous()  # [B, k, in_ch, m]
        # 将 B×k 展平成 (B*k) 个序列，过共享权重 Conv1d
        x = x.view(B * k, -1, m)          # [B*k, in_ch, m]
        x = self.mlp(x)                   # [B*k, d, m]
        x = x.max(dim=-1).values          # [B*k, d]
        return x.view(B, k, -1)           # [B, k, d]


# ----------------------------------------------------------------------------
# Numpy 工具（阶段 A）
# ----------------------------------------------------------------------------
def _conf_filter(xyz, conf, floor):
    mask = conf >= floor
    return xyz[mask], conf[mask]


def _downsample(xyz, conf, M_ds, rng):
    if len(xyz) <= M_ds:
        return xyz, conf
    idx = rng.choice(len(xyz), M_ds, replace=False)
    return xyz[idx], conf[idx]


def _fps_np(xyz, n):
    M = xyz.shape[0]
    if M == 0:
        return np.array([], dtype=np.int64)
    if M <= n:
        rep = int(np.ceil(n / M))
        return np.tile(np.arange(M), rep)[:n]
    idx = np.zeros(n, dtype=np.int64)
    idx[0] = 0
    min_d = ((xyz - xyz[0]) ** 2).sum(1)
    for i in range(1, n):
        idx[i] = int(np.argmax(min_d))
        nd = ((xyz - xyz[idx[i]]) ** 2).sum(1)
        min_d = np.minimum(min_d, nd)
    return idx


def _conf_importance(c, floor, hand_high):
    s = np.where(c < hand_high,
                 (hand_high - c) / (hand_high - floor),
                 np.full_like(c, 0.2))
    return np.clip(s, 0.0, 1.0)


def _geom_saliency(xyz, C, k_nn):
    """几何显著性 g = λ2/(λ1+eps)，clamp[0,1]。使用 torch 加速 PCA。"""
    torch = __import__("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xyz_t = torch.tensor(xyz, device=device)
    C_t = torch.tensor(C, device=device)
    with torch.no_grad():
        # KNN
        dist = torch.cdist(C_t, xyz_t)  # [N, M]
        _, nb_idx = dist.topk(k_nn, dim=-1, largest=False)
        nb = xyz_t[nb_idx]              # [N, k_nn, 3]
        diff = nb - C_t.unsqueeze(1)    # [N, k_nn, 3]
        cov = torch.einsum("nki,nkj->nij", diff, diff) / k_nn
        e = torch.linalg.eigvalsh(cov)  # [N, 3] asc
        ratio = (e[:, 1] / (e[:, 2] + 1e-6)).clamp(0.0, 1.0)
    return ratio.cpu().numpy()


def _build_patches_np(P, c, A, cfg):
    """构建局部 patch [k, m, 4]，返回 numpy array。"""
    k = A.shape[0]
    m = cfg.m
    scene_scale = float((P.max(0) - P.min(0)).max())
    scene_scale = max(scene_scale, 1e-6)

    torch = __import__("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P_t = torch.tensor(P, device=device)
    A_t = torch.tensor(A, device=device)
    c_t = torch.tensor(c, device=device).unsqueeze(-1)

    with torch.no_grad():
        dist = torch.cdist(A_t, P_t)  # [k, M]
        _, nb_idx = dist.topk(m, dim=-1, largest=False)
        nb_xyz = P_t[nb_idx]          # [k, m, 3]
        xyz_loc = (nb_xyz - A_t.unsqueeze(1)) / scene_scale

    if cfg.use_conf_channel:
        nb_conf = c_t[nb_idx] / 5.0   # [k, m, 1]
        patch = torch.cat([xyz_loc, nb_conf], dim=-1)  # [k, m, 4]
    else:
        patch = xyz_loc

    patch = torch.where(torch.isfinite(patch), patch, torch.zeros_like(patch))
    return patch.cpu().numpy()


def _compute_anchors_np(xyz, conf, cfg, rng):
    """
    阶段 A: conf 过滤 → 降采样 → FPS → importance → top-k 锚点。
    返回 (A: [k,3], c_A: [k], P: [M',3], c: [M'], info: dict)
    """
    P, c = _conf_filter(xyz, conf, cfg.conf_floor)
    if len(P) == 0:
        return None, None, P, c, {"low_quality": True, "reason": "empty_after_filter"}

    P, c = _downsample(P, c, cfg.M_ds, rng)
    low_quality = len(P) < 2 * cfg.k

    if len(P) >= cfg.k:
        cand_idx = _fps_np(P, min(cfg.N_fps, len(P)))
        C = P[cand_idx]
        c_C = c[cand_idx]
        s_conf = _conf_importance(c_C, cfg.conf_floor, cfg.conf_hand_high)
        if cfg.w_struct > 0 and len(C) > cfg.k_struct_nn:
            g = _geom_saliency(P, C, cfg.k_struct_nn)
            s = cfg.w_hand * s_conf + cfg.w_struct * g
        else:
            s = s_conf
        order = np.argsort(-s)
        A = C[order[:cfg.k]]
    else:
        # 退化：点数不足 k，重复补齐
        rep = int(np.ceil(cfg.k / len(P)))
        A = np.tile(P, (rep, 1))[:cfg.k]

    return A, P, c, {
        "M_raw": int(len(xyz)),
        "M_prime": int(len(P)),
        "k": int(A.shape[0]),
        "low_quality": low_quality,
    }


# ----------------------------------------------------------------------------
# HOIHandler: 封装阶段 A + B
# ----------------------------------------------------------------------------
class HOIHandler(nn.Module):
    """
    从 H_raw 在线生成 H* 的完整管线。
    阶段 A（numpy）在 forward 中每 batch 执行一次（不可微，廉价）。
    阶段 B（AnchorEncoder）是 nn.Module，参与梯度回传。

    forward 输入支持两种模式:
      - 列表模式: h_raw_xyz = [np.ndarray(M_i,3), ...], conf = [np.ndarray(M_i), ...]
      - 张量模式: h_raw_xyz = Tensor([B, M_max, 3]), conf = Tensor([B, M_max])
                  + valid_mask = Tensor([B, M_max]) 布尔
    """

    def __init__(self, cfg: HStarCfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = AnchorEncoder(d=cfg.d, use_conf_channel=cfg.use_conf_channel)
        # 维度对齐: AnchorEncoder 输出 d → cross-attn 所需 d_model
        if cfg.d != cfg.hstar_proj:
            self.proj = nn.Sequential(
                nn.Linear(cfg.d, cfg.hstar_proj),
                nn.LayerNorm(cfg.hstar_proj),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, h_raw_xyz, h_raw_conf):
        """
        Args:
            h_raw_xyz: list of np.ndarray [(M_i, 3), ...] 或 Tensor [B, M_max, 3]
            h_raw_conf: list of np.ndarray [(M_i,), ...] 或 Tensor [B, M_max]
        Returns:
            H_star: [B, k, hstar_proj]  (已投影到 cross-attn 维度)
        """
        B = len(h_raw_xyz) if isinstance(h_raw_xyz, (list, tuple)) else h_raw_xyz.shape[0]
        device = self.encoder.mlp[0].weight.device

        # 逐样本执行阶段 A + B
        token_list = []
        for i in range(B):
            if isinstance(h_raw_xyz, (list, tuple)):
                xyz_i = np.asarray(h_raw_xyz[i], dtype=np.float32)
                conf_i = np.asarray(h_raw_conf[i], dtype=np.float32)
            else:
                # 张量模式：用 valid_mask 裁剪
                xyz_i = h_raw_xyz[i].cpu().numpy().astype(np.float32)
                conf_i = h_raw_conf[i].cpu().numpy().astype(np.float32)

            rng = np.random.default_rng(42 + i)
            A, P, c, info = _compute_anchors_np(xyz_i, conf_i, self.cfg, rng)

            if A is None or len(A) == 0:
                # 完全退化：返回零向量
                token_list.append(torch.zeros(self.cfg.k, self.cfg.hstar_proj, device=device))
                continue

            patches = _build_patches_np(P, c, A, self.cfg)  # [k, m, in_ch]
            patches_t = torch.tensor(patches, device=device, dtype=torch.float32).unsqueeze(0)  # [1, k, m, in_ch]

            with torch.no_grad() if not self.training else torch.enable_grad():
                tokens = self.encoder(patches_t)  # [1, k, d]
                tokens = self.proj(tokens)        # [1, k, hstar_proj]

            token_list.append(tokens.squeeze(0))   # [k, hstar_proj]

        # 统一 k 维度 stack
        H_star = torch.stack(token_list, dim=0)   # [B, k, hstar_proj]
        return H_star

    def extra_repr(self) -> str:
        return f"k={self.cfg.k} d={self.cfg.d} proj={self.cfg.hstar_proj}"
