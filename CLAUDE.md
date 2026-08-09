# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GEAL (Generalizable 3D Affordance Learning with Cross-Modal Consistency) — CVPR 2025 paper. Predicts per-point 3D affordance (18 affordance types across 23 object classes) from point clouds using a dual-branch architecture:

- **Branch 2D** (`model/branch_2d.py`): Renders point clouds to multi-view 2D images via differentiable Gaussian splatting (`renderer/gaussian_render.py`), extracts DINOv2 visual features, fuses with text (roberta-base affordance descriptions), outputs 2D activation maps.
- **Branch 3D** (`model/branch_3d.py`): PointNet++ hierarchical encoder with language-aware GAFM fusion blocks (`model/fusion_block.py`), predicts per-point 3D affordance scores.

Training is two-stage: Stage 1 trains the 2D branch (frozen DINOv2 + text encoder). Stage 2 trains the 3D branch with the 2D branch as a frozen teacher via KLD/MSE alignment loss.

Three architectural variants are supported via config flags:
1. **Baseline** (`train_stage2.yaml`): Standard KLD/MSE alignment.
2. **V1 Local 3D Tokenizer** (`train_stage2_v1_tokenizer.yaml`): Adds `Local3DTokenizer` + `TokenFusion` for geometry-only enrichment. Design doc at `docu/GEAL_V1_连续局部3D_Tokenizer_改动方案.md`.
3. **Lightweight/MIFAG** (`train_stage2_lightweight.yaml`): Uses PIAD interaction images, adds trainable `AffordanceProj` head, applies `L_invariant`, `L_3d2img`, and optional `L_contrastive` losses from `utils/affordance_loss.py`.

## Environment

Python 3.10, CUDA 11.8, PyTorch 2.1.0. Custom CUDA extension `diff-gaussian-rasterization` in `thirdparty/`. Text encoder uses roberta-base at `/home/junbo/wyn/model/roberta-base` (hardcoded in configs).

## Key Commands

### Training
```bash
# Stage 1: 2D branch
python scripts/train_stage1.py --config config/train_stage1.yaml

# Stage 2: 3D branch (baseline)
python scripts/train_stage2.py --config config/train_stage2.yaml

# Stage 2: with V1 Local 3D Tokenizer
python scripts/train_stage2.py --config config/train_stage2_v1_tokenizer.yaml

# Stage 2: Lightweight/MIFAG pipeline
python scripts/train_stage2.py --config config/train_stage2_lightweight.yaml
```
Outputs go to `runs/train/`. Key config fields: `category` (laso/piad), `setting` (seen/unseen), `data_root`, `pretrained_2d` (Stage 2).

### Evaluation
```bash
# Standard evaluation (IoU, AUC, SIM, MAE)
python scripts/evaluation.py --config config/evaluation.yaml --output runs/result/

# Region-level metrics (small/mid/large bins, boundary F1, token stats)
python scripts/evaluate_region_metrics.py --config config/evaluation_v1_tokenizer.yaml --output runs/result_region/

# With point downsampling curve
python scripts/evaluate_region_metrics.py --config config/evaluation_v1_tokenizer.yaml --output runs/result_region/ --downsample --point_counts 2048,1536,1024,512 --repeats 3

# Use --skip_token to disable tokenizer metrics for baseline comparison
python scripts/evaluate_region_metrics.py --config config/evaluation.yaml --output runs/result_region/ --skip_token

# Robustness evaluation (7 corruption types x 5 severity levels)
python scripts/evaluation_corrupt.py --config config/evaluation_corrupt.yaml

# Extended corruption with severity curves
python scripts/evaluation_corrupt_extended.py --config config/evaluation_corrupt.yaml --output runs/result/
```

### Visualization
```bash
# Export top-N per (affordance, class) as colored PLY
python visualization/export_point_cloud.py --config config/evaluation.yaml --top_n 10

# Mitsuba image rendering pipeline
python visualization/render_image.py --mode full --input_txt runs/ply/ply_paths.txt --xml_dir runs/xml_file --exr_dir runs/exr_file --jpg_dir runs/jpg_file

# Mitsuba rotating GIF
python visualization/render_video.py --input runs/ply/ply_paths.txt --out_dir runs/video --frames 200 --radius 3.5 --fps 24
```

### Data Preprocessing (PIAD)
```bash
python dataset/piad_process.py
```
Also copy `Affordance-Question.csv` from LASO root into PIAD root.

## Core Metrics

Defined in `utils/metrics.py` and `utils/region_metrics.py`:
- **aIoU**: Mean IoU over 20 uniform thresholds [0,1] — primary metric
- **IoU_50**: IoU at fixed threshold 0.5
- **AUC**: ROC-AUC, threshold-independent ranking quality
- **recall_50**: TP recall at threshold 0.5
- **boundary_f1**: F1 on boundary points detected via KNN(k=8) label changes
- **fp_ratio**: False positive ratio at threshold 0.5
- **Small/Mid/Large binning**: By GT positive point ratio (≤5%, 5%-20%, >20%)
- **Token stats** (V1 only): `token_to_point_dist`, `token_coverage`

## Loss Functions

In `utils/loss.py`: `HM_Loss` (Focal + Dice), `CosineLoss`, `SIM_Loss`, `CenterLoss`.
In `utils/affordance_loss.py`: `loss_invariant`, `loss_3d2img`, `loss_contrastive` (Lightweight/MIFAG only).

## Dataset Structure

- `dataset/laso.py`, `dataset/piad.py`: Dataset loaders
- `dataset/data_utils.py`: `CLASSES` (23), `AFFORDANCES` (18), `VIEWPOINTS`, normalization constants
- `dataset/corrupt.py`: Corruption dataset (7 types, 5 levels)

## Key Files for Model Changes

- `model/branch_3d.py` — Main 3D network; entry point for most 3D-side changes
- `model/local_3d_tokenizer.py` — V1 tokenizer: `Local3DTokenizer`, `TokenToPointInterpolator`, `TokenFusion`
- `model/fusion_block.py` — `GAFMBlock`: granularity-adaptive fusion
- `model/gaf_conv.py` — `GafConv`: multi-level feature gating
- `model/layers.py` — Utility layers: `Mlp`, `SmallUpsampleNet`, `FeatureUpsampler`, `PointFeatureDownsampler`, `PointEncoder`
- `model/attention.py` — `TransformerDecoder`, `TransformerDecoderLayer`
- `scripts/train_stage2.py` — Training loop; understand the loss composition and variant dispatch logic before modifying

## Development Workflow Rules (改造流程)

Every code modification must follow these five phases in order:

### 1. 想法阶段 (Idea Phase)

Before any code change, write a design document under `docu/`, named after the change point (改造点). The document must include:
- **改造原因**: Why this change is needed, what problem it solves
- **改造原理**: The technical approach and mechanism
- **论文出处**: Cited paper and the specific module/method being referenced
- **改造方案**: Concrete implementation plan
- **风险点**: Potential risks, failure modes, or side effects

### 2. 改造阶段 (Implementation Phase)

- New module code must be **completely decoupled** from the baseline — never modify baseline code in place
- Create a **new config file** for the variant, so the new module can be toggled on/off via config flags
- The baseline behavior must remain unchanged when the new module is disabled

### 3. 训练阶段 (Training Phase)

- Training logs go under `runs/`
- Log directory name is controlled by the config field `log_name`
- `log_name` **must** be named after the change point (改造点名称)

### 4. 评估阶段 (Evaluation Phase)

After training, run both evaluation scripts on the resulting weights:
```bash
python scripts/evaluation.py --config <new_eval_config.yaml> --output runs/<改造点名称>/
python scripts/evaluate_region_metrics.py --config <new_eval_config.yaml> --output runs/<改造点名称>/
```
Results must be placed under `runs/<改造点名称>/`.

### 5. 分析阶段 (Analysis Phase)

Add an **analysis section** to the design document, covering:
- Metric comparison against the baseline at `runs/my_result/`
- Analysis of why metrics improved or regressed for each metric
- Per-region (Small/Mid/Large) and per-affordance breakdown if available

## Notes

- Config files use YAML; training scripts load via `utils/utils.py:read_yaml`.
- All script entry points accept `--config` pointing to a YAML file.
- The lightweight/unseen variant uses `config/evaluation_lightweight_unseen.yaml`.
- `utils/logger.py` sets up logging; `utils/region_metrics.py` has the region-metric computation logic.