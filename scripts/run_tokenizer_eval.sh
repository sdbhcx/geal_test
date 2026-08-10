#!/bin/bash
# Run 4 tokenizer weight evaluations (evaluation.py + evaluate_region_metrics.py) in parallel
# Results: runs/result_tokenizer/

set -euo pipefail

OUTPUT_DIR="runs/result_tokenizer"
mkdir -p "$OUTPUT_DIR"

CONFIGS=(
  "config/evaluation_tokenizer_seed100_seen.yaml:_seed100_seen"
  "config/evaluation_tokenizer_seed100_unseen.yaml:_seed100_unseen"
  "config/evaluation_tokenizer_seed2024_seen.yaml:_seed2024_seen"
  "config/evaluation_tokenizer_seed2024_unseen.yaml:_seed2024_unseen"
)

PIDS=()

for pair in "${CONFIGS[@]}"; do
  cfg="${pair%%:*}"
  name="${pair##*:}"
  (
    echo "=========================================="
    echo "[$(date '+%H:%M:%S')] Starting: $cfg | $name"
    echo "=========================================="
    python scripts/evaluation.py \
      --config "$cfg" \
      --output "$OUTPUT_DIR" \
      --name "$name"
    python scripts/evaluate_region_metrics.py \
      --config "$cfg" \
      --output "$OUTPUT_DIR" \
      --name "$name"
    echo "[$(date '+%H:%M:%S')] Done: $cfg | $name"
  ) &
  PIDS+=($!)
done

echo "All jobs launched, PIDs: ${PIDS[*]}"

# Wait for all background jobs
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAILED=1
    echo "Job $pid failed!"
  fi
done

if [ "$FAILED" -eq 0 ]; then
  echo "All evaluations completed successfully."
else
  echo "Some evaluations failed. Check output above."
fi