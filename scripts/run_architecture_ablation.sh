#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/experiment.sh"

SEEDS=(2024 2025 2026)
VARIANTS=(full no-router no-router-no-cross single-granularity)

for tabs in 2 3 4 5; do
  dataset="closed_${tabs}tab"
  for variant in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      checkpoint="checkpoints/ablation/${dataset}_${variant}_seed${seed}.pt"
      run_prismwf_train "$checkpoint" "$tabs" "$seed" \
        --datasets "$dataset" \
        --ablation "$variant"
      run_prismwf_evaluate \
        "$dataset" "$checkpoint" "$tabs" "$seed" \
        "results/ablation/${dataset}_${variant}_seed${seed}" \
        --ablation "$variant"
    done
  done
done
