#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <dataset> <tabs> <seed>" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/experiment.sh"

DATASET=$1
TABS=$2
SEED=$3
FEATURE_WORKERS=${FEATURE_WORKERS:-8}
TRAIN_WORKERS=${TRAIN_WORKERS:-10}

cd "$PRISMWF_ROOT"
echo "[$(date --iso-8601=seconds)] $DATASET seed $SEED pipeline started"

for split in train valid test; do
  input="$DATA_ROOT/$DATASET/${split}.npz"
  output="$DATA_ROOT/$DATASET/hg8_${split}.npz"
  if [[ -s "$output" ]]; then
    echo "[$(date --iso-8601=seconds)] feature exists: $output"
    continue
  fi
  echo "[$(date --iso-8601=seconds)] extracting six-channel robust trace features: $split"
  run_prismwf_extract "$input" "$output" \
    --raw-length 10000 \
    --slot-ms 20 \
    --max-loading-seconds 160 \
    --num-workers "$FEATURE_WORKERS"
done

checkpoint="checkpoints/$DATASET/seed${SEED}.pt"
result="results/$DATASET/seed${SEED}"

echo "[$(date --iso-8601=seconds)] training $DATASET seed $SEED"
run_prismwf_train "$checkpoint" "$TABS" "$SEED" \
  --datasets "$DATASET" \
  --epochs 80 \
  --batch-size 256 \
  --learning-rate 5e-4 \
  --num-workers "$TRAIN_WORKERS"

echo "[$(date --iso-8601=seconds)] evaluating $DATASET seed $SEED"
run_prismwf_evaluate \
  "$DATASET" "$checkpoint" "$TABS" "$SEED" "$result" \
  --batch-size 256 \
  --num-workers "$TRAIN_WORKERS"

sha256sum "$checkpoint" > "${checkpoint%.pt}.sha256"
echo "[$(date --iso-8601=seconds)] pipeline complete"
