#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/experiment.sh"

DATASET=${DATASET:-closed_5tab}
CHECKPOINT=${CHECKPOINT:-checkpoints/closed_5tab/seed2024.pt}

conditions=(
  "packet-loss 0.10 loss10"
  "packet-loss 0.20 loss20"
  "packet-loss 0.30 loss30"
  "latency-scale 1.25 scale125"
  "latency-scale 1.50 scale150"
  "latency-scale 2.00 scale200"
  "midpoint-delay-offset 100 offset100ms"
  "midpoint-delay-offset 300 offset300ms"
  "midpoint-delay-offset 500 offset500ms"
)

for specification in "${conditions[@]}"; do
  read -r condition value suffix <<< "$specification"
  output="$DATA_ROOT/$DATASET/hg8_test_${suffix}.npz"
  (
    cd "$PRISMWF_ROOT"
    "$PYTHON_BIN" tools/generate_robustness_features.py \
      --input "$DATA_ROOT/$DATASET/test.npz" \
      --output "$output" \
      --condition "$condition" \
      --value "$value" \
      --seed 2024
  )
  run_prismwf_evaluate \
    "$DATASET" "$CHECKPOINT" 5 2024 "results/robustness/${suffix}" \
    --split "test_${suffix}"
done
