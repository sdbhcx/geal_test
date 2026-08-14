#!/usr/bin/env python3
"""Evaluate region-level metrics for 3D affordance models.

Usage::

    python scripts/evaluate_region_metrics.py --config config/evaluation_v1_tokenizer.yaml \\
        --output runs/result_tokenizer/ --name _region

This complements ``scripts/evaluation.py`` with:
  - Small / Mid / Large affordance region aIoU and recall
  - Boundary F-score
  - Token-to-point interpolation distance (when tokenizer is active)
  - Token coverage
  - Point downsampling curve

Requires ``return_token_aux: true`` in the model_3d config when tokenizer
metrics are needed; the script sets this automatically.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from dataset.data_utils import CLASSES, AFFORDANCES
from dataset.laso import LasoDataset
from dataset.piad import PiadDataset
from model.branch_3d import Branch3D
from model.local_3d_tokenizer import validate_local_tokenizer_checkpoint
from utils.region_metrics import (
    collect_region_metrics,
    downsampling_curve,
    format_region_metrics,
    save_region_metrics,
)
from utils.utils import read_yaml, seed_torch

# ---------------------------------------------------------------------------


def _load_model(cfg, device):
    model = Branch3D(cfg["model_3d"])
    ckpt = torch.load(cfg["ckpt"], map_location=device)
    status = model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)

    if model.use_local_tokenizer:
        validate_local_tokenizer_checkpoint(ckpt["model"])

    print(f"  Checkpoint: {cfg['ckpt']}")
    if status.missing_keys:
        print(f"  Missing keys: {status.missing_keys}")
    if status.unexpected_keys:
        print(f"  Unexpected keys: {status.unexpected_keys}")
    return model


def _load_dataloader(cfg, batch_size, num_workers):
    if cfg["dataset"] == "laso":
        dataset = LasoDataset("test", cfg["setting"], data_root=cfg["data_root"])
    else:
        dataset = PiadDataset("test", cfg["setting"], data_root=cfg["data_root"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate region-level 3D affordance metrics")
    parser.add_argument("--config", type=str, default="config/evaluation_v1_tokenizer.yaml")
    parser.add_argument("--output", type=str, default="runs/result_region/")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--name", type=str, default="_region")
    parser.add_argument(
        "--downsample",
        action="store_true",
        help="Run point-downsampling curve with controlled subsampling (slower)",
    )
    parser.add_argument(
        "--point_counts",
        type=str,
        default="2048,1536,1024,512",
        help="Comma-separated target point counts for downsampling (default: 2048,1536,1024,512)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per point count for downsampling (default: 3)",
    )
    parser.add_argument(
        "--skip_token",
        action="store_true",
        help="Skip tokenizer metrics (useful for baseline comparison)",
    )
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = read_yaml(args.config)
    seed_torch(cfg.get("seed", 42))

    # Enable token_aux return when needed
    if not args.skip_token:
        cfg["model_3d"].setdefault("return_token_aux", True)

    print("\nLoading model...")
    t0 = time.perf_counter()
    model = _load_model(cfg, device)
    tokenizer_enabled = getattr(model, "use_local_tokenizer", False) and not args.skip_token
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s  "
          f"(tokenizer={'ON' if tokenizer_enabled else 'OFF'})")

    print("Loading dataset...")
    loader = _load_dataloader(cfg, args.batch_size, args.num_workers)
    print(f"  {len(loader)} batches, {len(loader.dataset)} samples")

    # ---- Region metrics ----
    print("\nRunning region metrics...")
    t0 = time.perf_counter()
    metrics = collect_region_metrics(
        loader, model, device,
        tokenizer_enabled=tokenizer_enabled,
        small_region_ratio=0.05,
        boundary_k=8,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s ({metrics.get('count', 0)} valid samples)")

    print(format_region_metrics(metrics))

    # Save
    region_path = os.path.join(
        args.output,
        f"{cfg['dataset']}_{cfg['setting']}{args.name}_region.txt",
    )
    save_region_metrics(metrics, region_path)
    print(f"\nRegion metrics saved: {region_path}")

    # ---- Downsampling curve (optional) ----
    if args.downsample:
        print("\nRunning downsampling curve...")
        t0 = time.perf_counter()
        point_counts = [int(v.strip()) for v in args.point_counts.split(",") if v.strip()]
        ds = downsampling_curve(
            loader, model, device,
            point_counts=point_counts,
            repeats=args.repeats,
            seed=cfg.get("seed", 42),
        )
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s")

        metric_keys = ["aIoU", "IoU_50", "auc", "recall_50", "boundary_f1", "fp_ratio"]

        ds_path = os.path.join(
            args.output,
            f"{cfg['dataset']}_{cfg['setting']}{args.name}_downsample.txt",
        )
        with open(ds_path, "w") as f:
            f.write("Point Downsampling Curve (random subsampling with repeats):\n")
            f.write(f"  {'Pts':>6s}  " + "  ".join(f"{k:>10s}_mean  {k:>10s}_std" for k in metric_keys) + "\n")
            for pc_str, pc_data in ds.items():
                row = f"  {pc_str:>6s}  "
                for mk in metric_keys:
                    row += f"{pc_data['mean'][mk]:>10.4f}  {pc_data['std'][mk]:>10.4f}  "
                f.write(row + "\n")
        print(f"  Downsampling curve saved: {ds_path}")

        for pc_str, pc_data in ds.items():
            parts = [f"{pc_str}pts:"]
            for mk in metric_keys:
                parts.append(f"{mk}={pc_data['mean'][mk]:.4f}±{pc_data['std'][mk]:.4f}")
            print(f"    {'  '.join(parts)}")


if __name__ == "__main__":
    main()