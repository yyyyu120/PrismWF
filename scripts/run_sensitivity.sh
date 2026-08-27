#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/experiment.sh"

DATASET=${DATASET:-closed_5tab}
SEEDS=(2024 2025 2026)

run_setting() {
  local slot_ms=$1
  local window_s=$2
  local prefix="hg8_i${slot_ms}_w${window_s}"
  for split in train valid test; do
    run_prismwf_extract \
      "$DATA_ROOT/$DATASET/${split}.npz" \
      "$DATA_ROOT/$DATASET/${prefix}_${split}.npz" \
      --slot-ms "$slot_ms" \
      --max-loading-seconds "$window_s"
  done
  for seed in "${SEEDS[@]}"; do
    checkpoint="checkpoints/sensitivity/${prefix}_seed${seed}.pt"
    run_prismwf_train "$checkpoint" 5 "$seed" \
      --datasets "$DATASET" \
      --feature-prefix "$prefix"
    run_prismwf_evaluate \
      "$DATASET" "$checkpoint" 5 "$seed" \
      "results/sensitivity/${prefix}_seed${seed}" \
      --feature-prefix "$prefix"
  done
}

for window in 80 160 240 320; do
  run_setting 20 "$window"
done
for interval in 10 40 80; do
  run_setting "$interval" 160
done
