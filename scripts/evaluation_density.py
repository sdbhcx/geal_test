"""Controlled point-density evaluation for GEAL.

Evaluates the same test samples after deterministic random subsampling to
multiple point counts. Labels are indexed with the same sampled point indices.
Use several repeats to estimate sensitivity to the sampled subset.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")

from dataset.laso import LasoDataset
from dataset.piad import PiadDataset
from model.branch_3d import Branch3D
from utils.metrics import aggregate_extended_metrics, extended_sample_metrics
from utils.utils import read_yaml, seed_torch


def unwrap_prediction(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def sample_indices(total_points, target_points, seed):
    if target_points > total_points:
        raise ValueError(f"target_points={target_points} exceeds input size {total_points}")
    if target_points == total_points:
        return torch.arange(total_points)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(total_points, generator=generator)[:target_points].sort().values


def evaluate_density(
    model,
    dataloader,
    device,
    target_points,
    repeat,
    base_seed,
    threshold,
    small_region_ratio,
    boundary_k,
):
    records = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (point, _cls, _binary, question, _aff, label) in enumerate(
            tqdm(dataloader, total=len(dataloader), ascii=True)
        ):
            total_points = point.shape[-1]
            indices = sample_indices(
                total_points,
                target_points,
                seed=base_seed + repeat * 1_000_003 + batch_idx,
            )
            sampled_point = point.index_select(-1, indices).float().to(device)
            sampled_label = label.index_select(-1, indices).float().to(device)
            pred = unwrap_prediction(model(question, sampled_point))

            pred_np = pred.detach().cpu().numpy()
            label_np = sampled_label.detach().cpu().numpy()
            point_np = sampled_point.detach().cpu().numpy()
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
    return aggregate_extended_metrics(records)


def nanmean_metric(repeat_results, key):
    values = np.asarray([r.get(key, np.nan) for r in repeat_results], dtype=np.float64)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan


def nanstd_metric(repeat_results, key):
    values = np.asarray([r.get(key, np.nan) for r in repeat_results], dtype=np.float64)
    return float(np.nanstd(values)) if np.any(np.isfinite(values)) else np.nan


def main():
    parser = argparse.ArgumentParser(description="GEAL point-density evaluation")
    parser.add_argument("--config", type=str, default="config/evaluation.yaml")
    parser.add_argument("--output", type=str, default="runs/result/")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--point_counts", type=str, default="2048,1536,1024,512")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--small_region_ratio", type=float, default=0.05)
    parser.add_argument("--boundary_k", type=int, default=8)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    seed = cfg.get("seed", 42)
    seed_torch(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = Branch3D(cfg["model_3d"])
    ckpt = torch.load(cfg["ckpt"], map_location=device)
    status = model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    if status.missing_keys:
        print("Missing keys:", status.missing_keys)
    if status.unexpected_keys:
        print("Unexpected keys:", status.unexpected_keys)

    if cfg["dataset"] == "laso":
        dataset = LasoDataset("test", cfg["setting"], data_root=cfg["data_root"])
    else:
        dataset = PiadDataset("test", cfg["setting"], data_root=cfg["data_root"])
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    point_counts = [int(value.strip()) for value in args.point_counts.split(",") if value.strip()]
    metric_keys = [
        "aIoU",
        "IoU_50",
        "AUC",
        "small_region_aIoU",
        "small_region_recall",
        "boundary_fscore",
        "false_positive_area_ratio",
        "mean_nn_distance",
    ]
    results = {}
    for point_count in point_counts:
        repeat_results = []
        for repeat in range(args.repeats):
            print(f"\nEvaluating point_count={point_count}, repeat={repeat}")
            repeat_results.append(
                evaluate_density(
                    model=model,
                    dataloader=loader,
                    device=device,
                    target_points=point_count,
                    repeat=repeat,
                    base_seed=seed,
                    threshold=args.threshold,
                    small_region_ratio=args.small_region_ratio,
                    boundary_k=args.boundary_k,
                )
            )
        results[str(point_count)] = {
            "repeats": repeat_results,
            "mean": {key: nanmean_metric(repeat_results, key) for key in metric_keys},
            "std": {key: nanstd_metric(repeat_results, key) for key in metric_keys},
        }

    ckpt_name = os.path.splitext(os.path.basename(cfg["ckpt"]))[0]
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(
        args.output,
        f"{ckpt_name}_{cfg['dataset']}_{cfg['setting']}_density.json",
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, allow_nan=True)
    print(f"\nSaved controlled density results to: {output_file}")


if __name__ == "__main__":
    main()
