#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/experiment.sh"

SEEDS=(2024 2025 2026)
GROUPS=(all packet-count transition-count transition-interval)

for tabs in 2 5; do
  dataset="closed_${tabs}tab"
  for group in "${GROUPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      checkpoint="checkpoints/feature_ablation/${dataset}_${group}_seed${seed}.pt"
      run_prismwf_train "$checkpoint" "$tabs" "$seed" \
        --datasets "$dataset" \
        --feature-group "$group"
      run_prismwf_evaluate \
        "$dataset" "$checkpoint" "$tabs" "$seed" \
        "results/feature_ablation/${dataset}_${group}_seed${seed}" \
        --feature-group "$group"
    done
  done
done
