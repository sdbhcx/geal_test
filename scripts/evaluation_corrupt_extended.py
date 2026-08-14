"""Extended corruption evaluation with severity curves.

Runs the seven GEAL corruption types at all five levels and saves each level's
extended metrics plus area-under-severity-curve (AUSC) summaries.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from dataset.corrupt import CorruptDataset
from model.branch_3d import Branch3D
from utils.metrics import aggregate_extended_metrics, extended_sample_metrics
from utils.utils import read_yaml, seed_torch

CORRUPT_TYPES = [
    "scale",
    "jitter",
    "rotate",
    "dropout_global",
    "dropout_local",
    "add_global",
    "add_local",
]


def unwrap_prediction(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def evaluate_one_level(model, dataloader, device, threshold, small_region_ratio, boundary_k, density_bins):
    records = []
    model.eval()
    with torch.no_grad():
        for point, _cls, label, question, _aff_label in dataloader:
            point = point.float().to(device)
            label = label.float().to(device)
            pred = unwrap_prediction(model(question, point))
            pred_np = pred.detach().cpu().numpy()
            label_np = label.detach().cpu().numpy()
            point_np = point.detach().cpu().numpy()
            for idx in range(pred_np.shape[0]):
                records.append(
                    extended_sample_metrics(
                        pred_score=pred_np[idx],
                        target=label_np[idx],
                        points=point_np[idx],
                        small_region_ratio=small_region_ratio,
                        boundary_k=boundary_k,
                        threshold=threshold,
                    )
                )
    return aggregate_extended_metrics(records, density_bins=density_bins)


def trapezoid_auc(levels, values):
    levels = np.asarray(levels, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.nan
    return float(np.trapz(values[valid], levels[valid]) / (levels[valid][-1] - levels[valid][0] + 1e-12))


def main():
    parser = argparse.ArgumentParser(description="Extended GEAL corruption evaluation")
    parser.add_argument("--config", type=str, default="config/evaluation_corrupt.yaml")
    parser.add_argument("--output", type=str, default="runs/result/")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--small_region_ratio", type=float, default=0.05)
    parser.add_argument("--boundary_k", type=int, default=8)
    parser.add_argument("--density_bins", type=int, default=3)
    parser.add_argument("--levels", type=int, default=5)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    seed_torch(cfg.get("seed", 42))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = Branch3D(cfg["model_3d"])
    ckpt = torch.load(cfg["ckpt"], map_location=device)
    status = model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)

    print(f"\nCheckpoint loaded: {cfg['ckpt']}")
    if status.missing_keys:
        print("Missing keys:", status.missing_keys)
    if status.unexpected_keys:
        print("Unexpected keys:", status.unexpected_keys)

    all_results = {}
    levels = list(range(args.levels))
    for corrupt_type in CORRUPT_TYPES:
        level_results = []
        for level in levels:
            dataset = CorruptDataset(
                corrupt_type=corrupt_type,
                level=level,
                data_root=cfg["data_root"],
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )
            metrics = evaluate_one_level(
                model,
                loader,
                device,
                threshold=args.threshold,
                small_region_ratio=args.small_region_ratio,
                boundary_k=args.boundary_k,
                density_bins=args.density_bins,
            )
            level_results.append(metrics)
            print(
                f"{corrupt_type} level={level}: "
                f"aIoU={metrics.get('aIoU', np.nan):.4f}, "
                f"boundary={metrics.get('boundary_fscore', np.nan):.4f}, "
                f"FP-ratio={metrics.get('false_positive_area_ratio', np.nan):.4f}"
            )

        curve = {
            "levels": levels,
            "aIoU": [m.get("aIoU", np.nan) for m in level_results],
            "IoU_50": [m.get("IoU_50", np.nan) for m in level_results],
            "AUC": [m.get("AUC", np.nan) for m in level_results],
            "small_region_aIoU": [m.get("small_region_aIoU", np.nan) for m in level_results],
            "small_region_recall": [m.get("small_region_recall", np.nan) for m in level_results],
            "boundary_fscore": [m.get("boundary_fscore", np.nan) for m in level_results],
            "false_positive_area_ratio": [m.get("false_positive_area_ratio", np.nan) for m in level_results],
        }
        curve["AUSC"] = {
            key: trapezoid_auc(levels, values)
            for key, values in curve.items()
            if key != "levels"
        }
        all_results[corrupt_type] = {
            "levels": level_results,
            "severity_curve": curve,
            "mean_over_levels": {
                key: float(np.nanmean(values))
                for key, values in curve.items()
                if key != "levels" and key != "AUSC"
            },
        }

    ckpt_name = os.path.splitext(os.path.basename(cfg["ckpt"]))[0]
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(
        args.output,
        f"{ckpt_name}_{cfg.get('dataset', 'corrupt')}_extended_corruption.json",
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, allow_nan=True)
    print(f"\nSaved severity curves to: {output_file}")


if __name__ == "__main__":
    main()
