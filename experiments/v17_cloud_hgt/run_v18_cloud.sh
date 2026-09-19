#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/work/v17_cloud
CODE="$ROOT/code/v17_cloud_hgt"
PRED="$ROOT/outputs/v18_rank_predictions"
REPORT="$ROOT/outputs/v18_rank_report"
CONFIG="$CODE/config_v18_rank.json"

mkdir -p "$PRED" "$REPORT"
cd "$CODE"
for seed in 20260803 20260817 20260831; do
  for fold in 0 1 2 3 4; do
    python train_fold.py \
      --data-root "$ROOT/data" \
      --output "$PRED" \
      --config "$CONFIG" \
      --fold "$fold" \
      --seed "$seed" \
      --device auto
  done
done

python analyze_rank_only.py \
  --data-root "$ROOT/data" \
  --predictions "$PRED" \
  --output "$REPORT" \
  --config "$CONFIG"

printf 'complete\n' > "$REPORT/V18_COMPLETE"
