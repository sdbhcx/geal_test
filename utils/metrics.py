import time
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def evaluating(pred, label):

    mae = torch.sum(torch.abs(pred-label), dim=(0,1))
    points_num = pred.shape[0] * pred.shape[1]

    return mae, points_num

def evaluating_2d(pred, label):

    sim = cal_sim(pred, label)
    mae = (np.abs(pred-label)).mean()
    
    return sim, mae

def KLD(map1, map2, eps = 1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    kld = np.sum(map2*np.log( map2/(map1+eps) + eps))
    return kld
    
def cal_SIM_3d(map1, map2, eps=1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)

def cal_kl(pred: np.ndarray, gt: np.ndarray, eps=1e-12) -> np.ndarray:
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    kld = np.sum(map2 * np.log(map2 / (map1 + eps) + eps))
    return kld


def cal_sim(pred: np.ndarray, gt: np.ndarray, eps=1e-12) -> np.ndarray:
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    intersection = np.minimum(map1, map2)

    return np.sum(intersection)

def cal_nss(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred / 255.0
    gt = gt / 255.0
    std = np.std(pred)
    u = np.mean(pred)

    smap = (pred - u) / (std + 1e-12)
    fixation_map = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-12)
    fixation_map = image_binary(fixation_map, 0.1)

    nss = smap * fixation_map

    nss = np.sum(nss) / np.sum(fixation_map + 1e-12)

    return nss

def image_binary(image, threshold):
    output = np.zeros(image.size).reshape(image.shape)
    for xx in range(image.shape[0]):
        for yy in range(image.shape[1]):
            if (image[xx][yy] > threshold):
                output[xx][yy] = 1
    return output

def cal_nss_batch(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # Normalize predictions and ground truth
    pred = pred / 255.0
    gt = gt / 255.0
    
    # Calculate mean and std along the spatial dimensions for each image in the batch
    std = np.std(pred, axis=(1, 2), keepdims=True)
    u = np.mean(pred, axis=(1, 2), keepdims=True)
    
    # Normalize saliency map
    smap = (pred - u) / (std + 1e-12)
    
    # Normalize and binarize the fixation map
    fixation_map = (gt - np.min(gt, axis=(1, 2), keepdims=True)) / (np.max(gt, axis=(1, 2), keepdims=True) - np.min(gt, axis=(1, 2), keepdims=True) + 1e-12)
    fixation_map = (fixation_map > 0.1).astype(np.float32)  # Vectorized thresholding
    
    # Calculate NSS
    nss = smap * fixation_map
    nss = np.sum(nss, axis=(1, 2)) / (np.sum(fixation_map, axis=(1, 2)) + 1e-12)
    nss = nss.mean()
    
    return nss

def calculate_batch_iou_auc(pred, target):
    """
    Compute IoU and AUC for each instance in a batch.
    - IoU is averaged across multiple thresholds.
    - AUC is computed with ROC-AUC.
    """
    num_samples = pred.shape[0]
    iou, auc = np.zeros(num_samples), np.zeros(num_samples)
    thresholds = np.linspace(0, 1, 20)
    target = (target >= 0.5).astype(int)

    for i in range(num_samples):
        t_true = target[i]
        p_score = pred[i]

        if np.sum(t_true) == 0:
            # Skip samples without positive labels
            iou[i] = np.nan
            auc[i] = np.nan
            continue

        # Compute AUC safely
        try:
            auc[i] = roc_auc_score(t_true, p_score)
        except ValueError:
            auc[i] = np.nan

        # Compute averaged IoU across thresholds
        temp_iou = []
        for thr in thresholds:
            p_mask = (p_score >= thr).astype(int)
            intersect = np.sum(p_mask & t_true)
            union = np.sum(p_mask | t_true)
            temp_iou.append(1.0 * intersect / union if union > 0 else 0.0)

        iou[i] = np.mean(temp_iou)

    return iou, auc


def calculate_batch_sim(pred, target):
    """Compute histogram intersection similarity."""
    sim = np.minimum(
        pred / (np.sum(pred, axis=1, keepdims=True) + 1e-12),
        target / (np.sum(target, axis=1, keepdims=True) + 1e-12)
    )
    return sim.sum(-1)


def calculate_batch_mae(pred, target):
    """Compute mean absolute error."""
    return np.mean(np.abs(pred - target), axis=1)


def calculate_batch_iou(results: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute IoU for each instance in a batch by averaging over multiple thresholds.
    """
    iou = np.zeros(results.shape[0])
    IOU_thres = np.linspace(0, 1, 20)
    targets = (targets >= 0.5).astype(int)

    for i in range(results.shape[0]):
        t_true = targets[i]
        p_score = results[i]
        if np.sum(t_true) == 0:
            iou[i] = np.nan
            continue

        vals = []
        for thre in IOU_thres:
            p_mask = (p_score >= thre).astype(int)
            intersect = np.sum(p_mask & t_true)
            union = np.sum(p_mask | t_true)
            vals.append(0.0 if union == 0 else (1.0 * intersect / union))
        iou[i] = float(np.mean(vals))
    return iou


# ---------------------------------------------------------------------------
# Extended 3D affordance metrics
# ---------------------------------------------------------------------------

AREA_BUCKETS = (
    ("tiny", 0.0, 0.01),
    ("small", 0.01, 0.05),
    ("medium", 0.05, 0.20),
    ("large", 0.20, 1.01),
)


def _as_points(points: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 2 and points.shape == (3, n):
        points = points.T
    if points.ndim != 2 or points.shape[0] != n or points.shape[1] < 3:
        raise ValueError(f"points must have shape [N, 3] or [3, N], got {points.shape}")
    return points[:, :3]


def _knn_boundary(mask: np.ndarray, points: Optional[np.ndarray], k: int) -> np.ndarray:
    """Mark points whose k-neighborhood contains a different binary label."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    n = mask.size
    if n < 2:
        return np.zeros(n, dtype=bool)
    if points is None:
        # Fallback for callers without geometry: use a 1D neighborhood.
        radius = max(1, min(k // 2, n - 1))
        boundary = np.zeros(n, dtype=bool)
        for shift in range(1, radius + 1):
            boundary |= mask != np.roll(mask, shift)
            boundary |= mask != np.roll(mask, -shift)
        return boundary

    points = _as_points(points, n)
    k = max(1, min(int(k), n - 1))
    if cKDTree is not None:
        _, nn_idx = cKDTree(points).query(points, k=k + 1)
        nn_idx = np.asarray(nn_idx)[:, 1:]
    else:
        dist2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(dist2, np.inf)
        nn_idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
    return np.any(mask[nn_idx] != mask[:, None], axis=1)


def _f1_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    tp = float(np.sum(pred_mask & gt_mask))
    precision = tp / (float(np.sum(pred_mask)) + 1e-12)
    recall = tp / (float(np.sum(gt_mask)) + 1e-12)
    if precision + recall == 0:
        return 1.0 if not np.any(pred_mask) and not np.any(gt_mask) else 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _iou_at_threshold(pred_score: np.ndarray, gt_mask: np.ndarray, threshold: float) -> float:
    pred_mask = np.asarray(pred_score) >= threshold
    union = np.sum(pred_mask | gt_mask)
    return float(np.sum(pred_mask & gt_mask) / union) if union else 0.0


def _average_iou(pred_score: np.ndarray, gt_mask: np.ndarray, thresholds=None) -> float:
    """Match GEAL's original aIoU protocol: mean IoU over 20 thresholds."""
    thresholds = np.linspace(0, 1, 20) if thresholds is None else thresholds
    return float(np.mean([_iou_at_threshold(pred_score, gt_mask, thr) for thr in thresholds]))


def _recall_at_threshold(pred_score: np.ndarray, gt_mask: np.ndarray, threshold: float) -> float:
    positives = float(np.sum(gt_mask))
    if positives == 0:
        return np.nan
    return float(np.sum((np.asarray(pred_score) >= threshold) & gt_mask) / positives)


def _mean_nn_distance(points: Optional[np.ndarray]) -> float:
    if points is None or len(points) < 2:
        return np.nan
    points = _as_points(points, len(points))
    if cKDTree is not None:
        distances, _ = cKDTree(points).query(points, k=2)
        return float(np.asarray(distances)[:, 1].mean())
    dist2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    return float(np.sqrt(np.min(dist2, axis=1)).mean())


def extended_sample_metrics(
    pred_score: np.ndarray,
    target: np.ndarray,
    points: Optional[np.ndarray] = None,
    small_region_ratio: float = 0.05,
    boundary_k: int = 8,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute one sample's geometry-aware metrics.

    `false_positive_area` is a point-count proxy because this repository has
    point clouds but no triangle areas. It becomes an area approximation when
    sampling density is reasonably uniform.
    """
    pred_score = np.asarray(pred_score, dtype=np.float32).reshape(-1)
    gt_mask = np.asarray(target).reshape(-1) >= 0.5
    if pred_score.size != gt_mask.size:
        raise ValueError(f"prediction and target sizes differ: {pred_score.size} vs {gt_mask.size}")

    n = gt_mask.size
    gt_positive = int(gt_mask.sum())
    gt_ratio = float(gt_positive / max(n, 1))
    pred_mask = pred_score >= threshold
    fp_count = int(np.sum(pred_mask & ~gt_mask))
    fp_ratio = float(fp_count / max(n, 1))
    pred_boundary = _knn_boundary(pred_mask, points, boundary_k)
    gt_boundary = _knn_boundary(gt_mask, points, boundary_k)

    try:
        auc = float(roc_auc_score(gt_mask.astype(np.uint8), pred_score))
    except ValueError:
        auc = np.nan

    small = gt_positive > 0 and gt_ratio <= small_region_ratio
    bucket = "unknown"
    for name, lower, upper in AREA_BUCKETS:
        if lower <= gt_ratio < upper:
            bucket = name
            break

    return {
        "aIoU": _average_iou(pred_score, gt_mask),
        "IoU_50": _iou_at_threshold(pred_score, gt_mask, threshold),
        "AUC": auc,
        "small_region": bool(small),
        "small_region_aIoU": _average_iou(pred_score, gt_mask) if small else np.nan,
        "small_region_IoU_50": _iou_at_threshold(pred_score, gt_mask, threshold) if small else np.nan,
        "small_region_recall": _recall_at_threshold(pred_score, gt_mask, threshold) if small else np.nan,
        "boundary_fscore": _f1_from_masks(pred_boundary, gt_boundary),
        "false_positive_area": float(fp_count),
        "false_positive_area_ratio": fp_ratio,
        "gt_positive_count": gt_positive,
        "gt_area_ratio": gt_ratio,
        "area_bucket": bucket,
        "point_count": int(n),
        "mean_nn_distance": _mean_nn_distance(points),
    }


def _nanmean(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan


def aggregate_extended_metrics(records: Sequence[Dict[str, Any]], density_bins: int = 3) -> Dict[str, Any]:
    """Aggregate per-sample records and create area/density breakdowns."""
    records = list(records)
    if not records:
        return {"count": 0}

    def mean_key(key, subset=None):
        subset = records if subset is None else subset
        return _nanmean(r[key] for r in subset if key in r)

    result = {
        "count": len(records),
        "aIoU": mean_key("aIoU"),
        "IoU_50": mean_key("IoU_50"),
        "AUC": mean_key("AUC"),
        "small_region_count": int(sum(r["small_region"] for r in records)),
        "small_region_aIoU": mean_key("small_region_aIoU"),
        "small_region_recall": mean_key("small_region_recall"),
        "boundary_fscore": mean_key("boundary_fscore"),
        "false_positive_area": mean_key("false_positive_area"),
        "false_positive_area_ratio": mean_key("false_positive_area_ratio"),
        "mean_nn_distance": mean_key("mean_nn_distance"),
        "mean_point_count": mean_key("point_count"),
    }

    area_breakdown = {}
    for bucket, _, _ in AREA_BUCKETS:
        subset = [r for r in records if r.get("area_bucket") == bucket]
        area_breakdown[bucket] = {
            "count": len(subset),
            "aIoU": mean_key("aIoU", subset),
            "IoU_50": mean_key("IoU_50", subset),
            "AUC": mean_key("AUC", subset),
            "boundary_fscore": mean_key("boundary_fscore", subset),
            "false_positive_area_ratio": mean_key("false_positive_area_ratio", subset),
        }
    result["area_breakdown"] = area_breakdown

    density_values = np.asarray(
        [r["mean_nn_distance"] for r in records if np.isfinite(r.get("mean_nn_distance", np.nan))],
        dtype=np.float64,
    )
    density_breakdown = {}
    if density_values.size >= 2 and density_bins >= 2:
        quantiles = np.quantile(density_values, np.linspace(0, 1, density_bins + 1))
        for idx in range(density_bins):
            lower, upper = quantiles[idx], quantiles[idx + 1]
            if idx == density_bins - 1:
                subset = [r for r in records if lower <= r.get("mean_nn_distance", np.inf) <= upper]
            else:
                subset = [r for r in records if lower <= r.get("mean_nn_distance", np.inf) < upper]
            # Lower nearest-neighbor distance means denser sampling.
            name = ["dense", "medium", "sparse"][idx] if density_bins == 3 else f"density_bin_{idx}"
            density_breakdown[name] = {
                "count": len(subset),
                "mean_nn_distance": mean_key("mean_nn_distance", subset),
                "mean_point_count": mean_key("point_count", subset),
                "IoU_50": mean_key("IoU_50", subset),
                "AUC": mean_key("AUC", subset),
                "boundary_fscore": mean_key("boundary_fscore", subset),
                "false_positive_area_ratio": mean_key("false_positive_area_ratio", subset),
            }
    result["density_breakdown"] = density_breakdown
    return result


# ---------------------------------------------------------------------------
# Resource profiling
# ---------------------------------------------------------------------------


def _forward_prediction(model, text, points):
    output = model(text, points)
    return output[0] if isinstance(output, (tuple, list)) else output


def profile_model_resources(
    model,
    text,
    points: torch.Tensor,
    device: torch.device,
    warmup: int = 2,
    repeats: int = 10,
    profile_flops: bool = True,
) -> Dict[str, Any]:
    """Measure parameters, latency, CUDA peak memory and profiler FLOPs.

    FLOPs are reported only for operators recognized by torch.profiler. Custom
    CUDA extensions may be absent from the total, so the result is an estimate.
    """
    was_training = model.training
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _forward_prediction(model, text, points)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(max(1, repeats)):
            _forward_prediction(model, text, points)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    result = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "total_params_m": float(total_params / 1e6),
        "trainable_params_m": float(trainable_params / 1e6),
        "latency_ms": float(elapsed * 1000.0 / max(1, repeats)),
        "throughput_samples_per_sec": float(max(1, points.shape[0]) * max(1, repeats) / elapsed),
        "peak_cuda_memory_mb": np.nan,
        "flops": np.nan,
        "flops_g": np.nan,
        "flops_note": "torch.profiler estimate; unsupported custom operators may be omitted",
    }
    if device.type == "cuda":
        result["peak_cuda_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / 1024**2)

    if profile_flops:
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                with torch.no_grad():
                    _forward_prediction(model, text, points)
            flops = float(sum(getattr(evt, "flops", 0) or 0 for evt in prof.key_averages()))
            if flops > 0:
                result["flops"] = flops
                result["flops_g"] = flops / 1e9
        except Exception as exc:
            result["flops_note"] = f"torch.profiler unavailable: {type(exc).__name__}: {exc}"

    if was_training:
        model.train()
    return result
