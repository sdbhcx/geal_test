"""Region-level evaluation metrics for 3D affordance.

Complements the standard point-level IoU/AUC/SIM/MAE with geometry-aware
metrics required by the V1 evaluation protocol (§4.6 of the design doc):

recall_50：模型输出的 pred_score（0-1 之间的概率）在 阈值 0.5 处二值化后，预测为阳性的点占所有真实阳性点的比例
boundary_f1：衡量模型对 affordance 区域边界 的检出能力
recall_50 关注"区域内"覆盖，boundary_f1 关注"区域边缘"精度

- Small / Mid / Large affordance region binning and per-bucket aIoU
- Boundary F-score
- Small-region recall 
- Token-to-point average interpolation distance
- Token coverage (portion of points within each token's effective radius)
- Point downsampling curve
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

from sklearn.metrics import roc_auc_score

from utils.metrics import (
    _average_iou,
    _f1_from_masks,
    _iou_at_threshold,
    _knn_boundary,
    _recall_at_threshold,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _forward_with_token(model, question, point, device):
    """Run model forward with token_aux enabled; returns (pred, token_aux)."""
    old = getattr(model, "return_token_aux", False)
    model.return_token_aux = True
    try:
        output = model(question, point)
    finally:
        model.return_token_aux = old
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        return output[0], output[1]
    return output, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_region_metrics(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    tokenizer_enabled: bool = True,
    *,
    small_region_ratio: float = 0.05,
    boundary_k: int = 8,
) -> Dict[str, Dict[str, Any]]:
    """Run the evaluation loop and return region-aware metrics.

    Expects dataloader batches in the standard GEAL format::

        (point[B,3,N], cls[B], binary_label[B,N], question, aff_label[B], label[B,N])

    Returns a dict with keys:
      - ``by_bucket``: metrics grouped by small / mid / large region
      - ``overall``: overall boundary F1, small-region IoU, etc.
      - ``token`` (optional): token-to-point distance and coverage stats
      - ``records``: per-sample metric dict list (for further analysis)
    """
    records: List[Dict[str, Any]] = []
    ttp_distances: List[np.ndarray] = []
    token_coverages: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            (point, cls, binary_label, question, aff_label, label) = _unpack_batch(batch)

            point = point.float().to(device)
            label = label.float().to(device)

            if tokenizer_enabled and getattr(model, "use_local_tokenizer", False):
                pred, token_info = _forward_with_token(model, question, point, device)
            else:
                pred = model(question, point)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
                token_info = None

            pred_np = pred.cpu().numpy()
            label_np = label.cpu().numpy()
            xyz_np = point.cpu().numpy().transpose(0, 2, 1)  # [B, N, 3]

            for i in range(pred_np.shape[0]):
                rec = _sample_region_metrics(
                    pred_np[i], label_np[i], xyz_np[i],
                    small_region_ratio=small_region_ratio,
                    boundary_k=boundary_k,
                )
                if rec is not None:
                    records.append(rec)

            if token_info is not None:
                # Use original-space distance if available, otherwise normalized
                dist_key = "point_to_token_dist_orig" if "point_to_token_dist_orig" in token_info else "point_to_token_dist"
                dist_np = token_info[dist_key].cpu().numpy()
                centers_np = token_info["centers"].cpu().numpy()
                # Centers are in normalized space; scale to original for coverage
                centers_orig = centers_np * 0.5
                for i in range(dist_np.shape[0]):
                    ttp_distances.append(dist_np[i])
                    token_coverages.append(_compute_coverage(xyz_np[i], centers_orig[i]))

    if not records:
        return {"count": 0}

    by_bucket = _bin_by_region_size(records, small_region_ratio=small_region_ratio)
    overall = _compute_overall(records)
    result = {
        "count": len(records),
        "by_bucket": by_bucket,
        "overall": overall,
    }
    if token_coverages:
        result["token"] = _aggregate_token_stats(ttp_distances, token_coverages)
    result["records"] = records
    return result


def downsampling_curve(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    point_counts: Optional[List[int]] = None,
    repeats: int = 3,
    *,
    seed: int = 42,
    small_region_ratio: float = 0.05,
    boundary_k: int = 8,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate model metrics under controlled point-count subsampling.

    For each target point count, randomly subsamples points (with label
    alignment), runs the model, and computes per-sample region metrics across
    repeats. Returns per-point-count mean and std for key metrics.
    """
    point_counts = point_counts or [2048, 1536, 1024, 512]
    metric_keys = ["aIoU", "IoU_50", "auc", "recall_50", "boundary_f1", "fp_ratio"]

    results: Dict[str, Dict[str, Any]] = {}
    model.eval()

    with torch.no_grad():
        for pc in point_counts:
            repeat_records: Dict[int, List[Dict[str, Any]]] = {}
            for rep in range(repeats):
                repeat_records[rep] = []
                for batch_idx, batch in enumerate(dataloader):
                    point, cls, _, question, aff_label, label = _unpack_batch(batch)
                    total_points = point.shape[-1]
                    if pc >= total_points:
                        indices = torch.arange(total_points)
                    else:
                        gen = torch.Generator(device="cpu")
                        gen.manual_seed(seed + rep * 1_000_003 + batch_idx)
                        indices = torch.randperm(total_points, generator=gen)[:pc].sort().values

                    sampled_point = point[:, :, indices].float().to(device)
                    sampled_label = label[:, indices].float().to(device)
                    pred = model(question, sampled_point)
                    if isinstance(pred, (tuple, list)):
                        pred = pred[0]

                    pred_np = pred.cpu().numpy()
                    label_np = sampled_label.cpu().numpy()
                    xyz_np = sampled_point.cpu().numpy().transpose(0, 2, 1)

                    for i in range(pred_np.shape[0]):
                        rec = _sample_region_metrics(
                            pred_np[i], label_np[i], xyz_np[i],
                            small_region_ratio=small_region_ratio,
                            boundary_k=boundary_k,
                        )
                        if rec is not None:
                            repeat_records[rep].append(rec)

            pc_result: Dict[str, Any] = {
                "point_count": pc,
                "repeats": repeats,
                "repeat_details": [],
            }
            for rep in range(repeats):
                recs = repeat_records[rep]
                if not recs:
                    pc_result["repeat_details"].append({"count": 0})
                    continue
                detail = {"count": len(recs)}
                for mk in metric_keys:
                    detail[mk] = float(np.nanmean([r[mk] for r in recs]))
                pc_result["repeat_details"].append(detail)

            pc_result["mean"] = {}
            pc_result["std"] = {}
            for mk in metric_keys:
                vals = np.asarray(
                    [d.get(mk, float("nan")) for d in pc_result["repeat_details"] if mk in d],
                    dtype=np.float64,
                )
                pc_result["mean"][mk] = (
                    float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")
                )
                pc_result["std"][mk] = (
                    float(np.nanstd(vals)) if np.any(np.isfinite(vals)) else float("nan")
                )

            results[str(pc)] = pc_result

    return results


def _unpack_batch(batch) -> Tuple[torch.Tensor, ...]:
    """Handle both tuple and dict-style batches."""
    if isinstance(batch, dict):
        return (
            batch["point"],
            batch["cls"],
            batch.get("binary_label", batch["label"]),
            batch["question"],
            batch["aff_label"],
            batch["label"],
        )
    return batch


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------


def _sample_region_metrics(
    pred_score: np.ndarray,
    gt: np.ndarray,
    points: np.ndarray,
    *,
    small_region_ratio: float,
    boundary_k: int,
) -> Optional[Dict[str, Any]]:
    """Compute geometry-aware metrics for a single sample."""
    pred_score = np.asarray(pred_score, dtype=np.float32).reshape(-1)
    gt_mask = np.asarray(gt, dtype=np.float32).reshape(-1) >= 0.5

    n = gt_mask.size
    if n == 0:
        return None

    gt_positive = int(gt_mask.sum())
    if gt_positive == 0:
        return None

    gt_ratio = gt_positive / n
    pred_mask = pred_score >= 0.5

    # aIoU (mean IoU over 20 thresholds)
    a_iou = _average_iou(pred_score, gt_mask)
    iou_50 = _iou_at_threshold(pred_score, gt_mask, 0.5)
    recall_50 = _recall_at_threshold(pred_score, gt_mask, 0.5)

    # AUC (ROC-AUC)
    try:
        auc = float(roc_auc_score(gt_mask.astype(np.uint8), pred_score))
    except ValueError:
        auc = float("nan")

    # False-positive area ratio (point-count proxy under uniform sampling)
    fp_count = int(np.sum(pred_mask & ~gt_mask))
    fp_ratio = float(fp_count / max(n, 1))

    # Boundary detection via KNN
    gt_boundary = _knn_boundary(gt_mask, points, boundary_k)
    pred_boundary = _knn_boundary(pred_mask, points, boundary_k)
    boundary_f1 = _f1_from_masks(pred_boundary, gt_boundary)

    # Region bucket
    bucket = _bucket(gt_ratio, small_region_ratio)

    return {
        "aIoU": a_iou,
        "IoU_50": iou_50,
        "auc": auc,
        "recall_50": recall_50,
        "boundary_f1": boundary_f1,
        "fp_ratio": fp_ratio,
        "gt_ratio": gt_ratio,
        "gt_positive": gt_positive,
        "point_count": n,
        "bucket": bucket,
    }


def _bucket(gt_ratio: float, small_region_ratio: float) -> str:
    if gt_ratio <= small_region_ratio:
        return "small"
    if gt_ratio <= 0.20:
        return "mid"
    return "large"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _bin_by_region_size(
    records: List[Dict[str, Any]],
    *,
    small_region_ratio: float,
) -> Dict[str, Dict[str, Any]]:
    """Group records into small / mid / large and compute mean metrics."""
    buckets: Dict[str, List[Dict[str, Any]]] = {"small": [], "mid": [], "large": []}
    for r in records:
        buckets.setdefault(r["bucket"], []).append(r)

    result = {}
    for name, subset in buckets.items():
        if not subset:
            result[name] = {"count": 0}
            continue
        result[name] = {
            "count": len(subset),
            "aIoU": float(np.nanmean([r["aIoU"] for r in subset])),
            "IoU_50": float(np.nanmean([r["IoU_50"] for r in subset])),
            "auc": float(np.nanmean([r["auc"] for r in subset])),
            "recall_50": float(np.nanmean([r["recall_50"] for r in subset])),
            "boundary_f1": float(np.nanmean([r["boundary_f1"] for r in subset])),
            "fp_ratio": float(np.nanmean([r["fp_ratio"] for r in subset])),
            "mean_gt_ratio": float(np.mean([r["gt_ratio"] for r in subset])),
        }
    return result


def _compute_overall(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute overall region-level metrics."""
    # Overall recall: fraction of GT positive points predicted positive
    total_tp = sum(int(r["gt_positive"] * r["recall_50"]) for r in records)
    total_pos = sum(int(r["gt_positive"]) for r in records)
    overall_recall = total_tp / total_pos if total_pos > 0 else float("nan")

    # Boundary F1 aggregated by re-computing from per-sample prec/rec
    boundary_f1s = [r["boundary_f1"] for r in records]

    return {
        "small_region_aIoU": float(np.nanmean([
            r["aIoU"] for r in records if r["bucket"] == "small"
        ])),
        "overall_aIoU": float(np.nanmean([r["aIoU"] for r in records])),
        "overall_IoU_50": float(np.nanmean([r["IoU_50"] for r in records])),
        "overall_auc": float(np.nanmean([r["auc"] for r in records])),
        "overall_recall_50": float(overall_recall),
        "boundary_f1": float(np.nanmean(boundary_f1s)),
        "overall_fp_ratio": float(np.nanmean([r["fp_ratio"] for r in records])),
        "small_region_count": int(sum(1 for r in records if r["bucket"] == "small")),
        "mid_region_count": int(sum(1 for r in records if r["bucket"] == "mid")),
        "large_region_count": int(sum(1 for r in records if r["bucket"] == "large")),
    }


def _compute_coverage(points: np.ndarray, centers: np.ndarray) -> float:
    """Fraction of points within 5% of the point cloud diameter from nearest token center.

    With FPS sampling this should be close to 1.0; significant drops indicate
    token coverage gaps.
    """
    if len(centers) == 0 or len(points) == 0:
        return float("nan")
    points = points.reshape(-1, 3)
    centers = centers.reshape(-1, 3)

    if cKDTree is not None:
        _, idx = cKDTree(centers).query(points, k=1)
        dists = np.linalg.norm(points - centers[idx.flatten()], axis=1)
    else:
        dists = np.min(np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2), axis=1)

    # Effective radius: 5% of point cloud diameter
    diam = np.max(np.linalg.norm(points, axis=1)) - np.min(np.linalg.norm(points, axis=1))
    if diam < 1e-6:
        return 1.0
    threshold = 0.05 * (np.max(np.linalg.norm(points, axis=1)) * 2)
    return float(np.mean(dists <= threshold))


def _aggregate_token_stats(
    ttp_distances: List[np.ndarray],
    token_coverages: List[np.ndarray],
) -> Dict[str, float]:
    all_dist = np.concatenate([d.flatten() for d in ttp_distances])
    all_cov = np.array(token_coverages)

    return {
        "mean_token_to_point_dist": float(np.mean(all_dist)),
        "median_token_to_point_dist": float(np.median(all_dist)),
        "max_token_to_point_dist": float(np.max(all_dist)),
        "mean_token_coverage": float(np.mean(all_cov)),
        "min_token_coverage": float(np.min(all_cov)),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_region_metrics(metrics: Dict[str, Dict[str, Any]]) -> str:
    """Pretty-print region metrics for console and file output."""
    lines = ["===== Region-Level Metrics ====="]

    overall = metrics.get("overall", {})
    if overall:
        lines.append("")
        lines.append("Overall:")
        for k in ("overall_aIoU", "overall_IoU_50", "overall_auc", "overall_recall_50", "boundary_f1", "overall_fp_ratio"):
            v = overall.get(k, float("nan"))
            lines.append(f"  {k:25s}: {v:.4f}")
        lines.append(f"  {'small_region_count':25s}: {overall.get('small_region_count', 'N/A')}")
        lines.append(f"  {'mid_region_count':25s}: {overall.get('mid_region_count', 'N/A')}")
        lines.append(f"  {'large_region_count':25s}: {overall.get('large_region_count', 'N/A')}")

    by_bucket = metrics.get("by_bucket", {})
    if by_bucket:
        lines.append("")
        lines.append("By Region Bucket:")
        lines.append(f"  {'Bucket':<8s} {'Count':>6s} {'aIoU':>8s} {'IoU_50':>8s} {'AUC':>8s} {'Recall':>8s} {'BndF1':>8s} {'FPRatio':>8s} {'GT_Ratio':>9s}")
        for name in ("small", "mid", "large"):
            b = by_bucket.get(name, {})
            if b.get("count", 0) == 0:
                lines.append(f"  {name:<8s} {'0':>6s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>9s}")
                continue
            lines.append(
                f"  {name:<8s} {b['count']:>6d} {b['aIoU']:>8.4f} {b['IoU_50']:>8.4f} "
                f"{b['auc']:>8.4f} {b['recall_50']:>8.4f} {b['boundary_f1']:>8.4f} "
                f"{b['fp_ratio']:>8.4f} {b['mean_gt_ratio']:>9.4f}"
            )

    token = metrics.get("token", {})
    if token:
        lines.append("")
        lines.append("Token Statistics:")
        for k, v in token.items():
            lines.append(f"  {k:35s}: {v:.4f}")

    return "\n".join(lines)


def save_region_metrics(metrics: Dict[str, Dict[str, Any]], save_path: str) -> None:
    """Save region metrics to text file."""
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "w") as f:
        f.write(format_region_metrics(metrics))
        f.write("\n")