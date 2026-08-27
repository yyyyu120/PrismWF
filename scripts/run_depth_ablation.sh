#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/experiment.sh"

DATASET=${DATASET:-closed_2tab}
TABS=${TABS:-2}
SEEDS=(2024 2025 2026)

for blocks in 1 2 3 4 5; do
  for seed in "${SEEDS[@]}"; do
    checkpoint="checkpoints/depth/${DATASET}_blocks${blocks}_seed${seed}.pt"
    run_prismwf_train "$checkpoint" "$TABS" "$seed" \
      --datasets "$DATASET" \
      --num-layers "$blocks"
    run_prismwf_evaluate \
      "$DATASET" "$checkpoint" "$TABS" "$seed" \
      "results/depth/${DATASET}_blocks${blocks}_seed${seed}" \
      --num-layers "$blocks"
  done
done
