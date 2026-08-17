import torch
import torch.nn as nn


class HOIFusion(nn.Module):
    """
    纯内容交叉注意力：query = 物体点特征 P~，key/value = HOI 锚点特征 H*。

    训练期: hoi_feat=[B,k,d] → cross-attn 检索交互先验 → P~_aug
    推理期: hoi_feat=None  → 使用可学习 null_token 退化（近恒等映射）

    参数量: 仅 cross-attn + FFN + 2×LayerNorm + 1 null_token，约 1~2M。
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 推理期 H*=None 时使用的可学习空 token（仅特征）
        self.null_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, obj_feat: torch.Tensor, hoi_feat: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            obj_feat: [B, N, d] 物体点云特征 P~（query）
            hoi_feat: [B, k, d] HOI 锚点特征 H*（key/value）；推理期为 None
        Returns:
            [B, N, d] 融合后的 P~_aug
        """
        if hoi_feat is None:
            hoi_feat = self.null_token.expand(obj_feat.size(0), -1, -1)

        attn_out, _ = self.cross(obj_feat, hoi_feat, hoi_feat)
        fused = self.norm1(obj_feat + attn_out)
        fused = self.norm2(fused + self.ffn(fused))
        return fused

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"
