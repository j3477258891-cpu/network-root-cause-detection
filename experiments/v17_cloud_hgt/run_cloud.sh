#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${V17_DATA_ROOT:?set V17_DATA_ROOT to the mounted rootcause-v17-v1 dataset}"
OUTPUT_ROOT="${V17_OUTPUT_ROOT:-/tmp/v17_outputs}"
SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"

python "${SCRIPT_ROOT}/run_pipeline.py" \
  --data-root "${DATA_ROOT}" \
  --output "${OUTPUT_ROOT}" \
  --device auto \
  --resume
