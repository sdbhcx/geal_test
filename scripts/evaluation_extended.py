"""Extended evaluation for geometry-aware 3D affordance diagnostics.

Adds fixed-threshold aIoU, small-region metrics, point-cloud boundary F-score,
false-positive area proxies, affordance-area buckets, density buckets and model
resource profiling without changing the original evaluation.py protocol.
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
from utils.metrics import (
    aggregate_extended_metrics,
    extended_sample_metrics,
    profile_model_resources,
)
from utils.utils import read_yaml, seed_torch


def unwrap_prediction(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def evaluate_extended(
    model,
    dataloader,
    device,
    threshold=0.5,
    small_region_ratio=0.05,
    boundary_k=8,
    density_bins=3,
):
    records = []
    profile_batch = None
    model.eval()

    with torch.no_grad():
        for point, _cls, _binary_label, question, _aff_label, label in tqdm(
            dataloader, total=len(dataloader), ascii=True
        ):
            point = point.float().to(device)
            label = label.float().to(device)
            pred = unwrap_prediction(model(question, point))

            if profile_batch is None:
                profile_batch = (question, point.detach())

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

    return aggregate_extended_metrics(records, density_bins=density_bins), records, profile_batch


def main():
    parser = argparse.ArgumentParser(description="Extended GEAL evaluation")
    parser.add_argument("--config", type=str, default="config/evaluation.yaml")
    parser.add_argument("--output", type=str, default="runs/result/")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--small_region_ratio", type=float, default=0.05)
    parser.add_argument("--boundary_k", type=int, default=8)
    parser.add_argument("--density_bins", type=int, default=3)
    parser.add_argument("--profile_repeats", type=int, default=10)
    parser.add_argument("--skip_flops", action="store_true")
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

    if cfg["dataset"] == "laso":
        dataset = LasoDataset("test", cfg["setting"], data_root=cfg["data_root"])
    else:
        dataset = PiadDataset("test", cfg["setting"], data_root=cfg["data_root"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    summary, records, profile_batch = evaluate_extended(
        model=model,
        dataloader=loader,
        device=device,
        threshold=args.threshold,
        small_region_ratio=args.small_region_ratio,
        boundary_k=args.boundary_k,
        density_bins=args.density_bins,
    )

    if profile_batch is not None:
        questions, points = profile_batch
        summary["resources"] = profile_model_resources(
            model=model,
            text=questions,
            points=points,
            device=device,
            repeats=args.profile_repeats,
            profile_flops=not args.skip_flops,
        )

    summary["protocol"] = {
        "threshold": args.threshold,
        "small_region_ratio": args.small_region_ratio,
        "boundary_k": args.boundary_k,
        "density_bins": args.density_bins,
        "false_positive_area_definition": "number and ratio of false-positive points; an area proxy under uniform sampling",
        "density_definition": "mean nearest-neighbor distance after point-cloud normalization",
    }

    ckpt_name = os.path.splitext(os.path.basename(cfg["ckpt"]))[0]
    os.makedirs(args.output, exist_ok=True)
    summary_path = os.path.join(
        args.output,
        f"{ckpt_name}_{cfg['dataset']}_{cfg['setting']}_extended.json",
    )
    records_path = os.path.join(
        args.output,
        f"{ckpt_name}_{cfg['dataset']}_{cfg['setting']}_extended_samples.jsonl",
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    with open(records_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, allow_nan=True) + "\n")

    print("\n===== Extended Evaluation =====")
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"\nSummary saved to: {summary_path}")
    print(f"Per-sample records saved to: {records_path}")


if __name__ == "__main__":
    main()
