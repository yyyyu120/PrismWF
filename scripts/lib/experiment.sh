#!/usr/bin/env bash

PRISMWF_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-$PRISMWF_ROOT/datasets}
PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-cuda}

run_prismwf_train() {
  local checkpoint=$1
  local checkpoint_k=$2
  local seed=$3
  shift 3

  (
    cd "$PRISMWF_ROOT"
    "$PYTHON_BIN" train.py \
      --data-root "$DATA_ROOT" \
      --checkpoint "$checkpoint" \
      --checkpoint-k "$checkpoint_k" \
      --seed "$seed" \
      --device "$DEVICE" \
      "$@"
  )
}

run_prismwf_evaluate() {
  local dataset=$1
  local checkpoint=$2
  local tabs=$3
  local seed=$4
  local output=$5
  shift 5

  (
    cd "$PRISMWF_ROOT"
    "$PYTHON_BIN" evaluate.py \
      --data-root "$DATA_ROOT" \
      --dataset "$dataset" \
      --checkpoint "$checkpoint" \
      --tabs "$tabs" \
      --seed "$seed" \
      --device "$DEVICE" \
      --output "$output" \
      "$@"
  )
}

run_prismwf_extract() {
  local input=$1
  local output=$2
  shift 2

  (
    cd "$PRISMWF_ROOT"
    "$PYTHON_BIN" extract_features.py \
      --input "$input" \
      --output "$output" \
      "$@"
  )
}
