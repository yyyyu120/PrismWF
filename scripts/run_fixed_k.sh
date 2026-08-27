#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/experiment.sh"

SEEDS=(2024 2025 2026)

for scenario in closed open; do
  for tabs in 2 3 4 5; do
    dataset="${scenario}_${tabs}tab"
    for seed in "${SEEDS[@]}"; do
      checkpoint="checkpoints/${dataset}/seed${seed}.pt"
      run_prismwf_train "$checkpoint" "$tabs" "$seed" --datasets "$dataset"
      run_prismwf_evaluate \
        "$dataset" "$checkpoint" "$tabs" "$seed" \
        "results/${dataset}/seed${seed}"
    done
    (
      cd "$PRISMWF_ROOT"
      "$PYTHON_BIN" tools/aggregate_seeds.py \
        "results/${dataset}/seed2024.json" \
        "results/${dataset}/seed2025.json" \
        "results/${dataset}/seed2026.json" \
        --output "results/${dataset}/summary.json"
    )
  done
done
