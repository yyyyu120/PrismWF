#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/experiment.sh"

SEEDS=(2024 2025 2026)
DEFENSES=(wtfpad front regulator)

for tabs in 2 5; do
  for defense in "${DEFENSES[@]}"; do
    dataset="${defense}_${tabs}tab"
    for seed in "${SEEDS[@]}"; do
      checkpoint="checkpoints/defenses/${dataset}_seed${seed}.pt"
      run_prismwf_train "$checkpoint" "$tabs" "$seed" --datasets "$dataset"
      run_prismwf_evaluate \
        "$dataset" "$checkpoint" "$tabs" "$seed" \
        "results/defenses/${dataset}_seed${seed}"
    done
  done
done
