#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/experiment.sh"

SCENARIO=${SCENARIO:-closed}
SEEDS=(2024 2025 2026)
DATASETS=("${SCENARIO}_2tab" "${SCENARIO}_3tab" "${SCENARIO}_4tab" "${SCENARIO}_5tab")

for seed in "${SEEDS[@]}"; do
  checkpoint="checkpoints/${SCENARIO}_mixed/seed${seed}.pt"
  run_prismwf_train "$checkpoint" 5 "$seed" \
    --datasets "${DATASETS[@]}" \
    --sample-ratio 0.3

  validation=()
  test=()
  for tabs in 2 3 4 5; do
    dataset="${SCENARIO}_${tabs}tab"
    for split in valid test; do
      output="results/${SCENARIO}_mixed/seed${seed}_${split}_${tabs}tab"
      run_prismwf_evaluate \
        "$dataset" "$checkpoint" "$tabs" "$seed" "$output" \
        --split "$split"
      if [[ "$split" == valid ]]; then
        validation+=("${output}.npz")
      else
        test+=("${output}.npz")
      fi
    done
  done

  (
    cd "$PRISMWF_ROOT"
    "$PYTHON_BIN" tools/evaluate_unknown_k.py \
      --validation "${validation[@]}" \
      --test "${test[@]}" \
      --output "results/${SCENARIO}_mixed/unknown_k_seed${seed}.json"
  )
done
