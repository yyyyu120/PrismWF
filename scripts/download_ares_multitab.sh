#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-datasets}
CACHE_ROOT=${CACHE_ROOT:-downloads/ares}
ZENODO_RECORD=${ZENODO_RECORD:-13732130}
DOWNLOAD_CONNECTIONS=${DOWNLOAD_CONNECTIONS:-8}

datasets=(
  closed_2tab closed_3tab closed_4tab closed_5tab
  open_2tab open_3tab open_4tab open_5tab
)
md5s=(
  6c66e244544ae9e4316ed9927242215b
  b4296520242b5b4f9b2fe67b2b9ae64d
  9f479bbd452240d944bd65d65373efd1
  8e6f8c675f253e2c2538c3d3005c526e
  dba781164a049f4fd8043d35ce932a15
  3262e9a9461ff56b70b717e1295a8a90
  c7b5bf33bd4498cc5b3163669754092c
  92b3d6143d9c740122b5e8cc10fb5d1b
)

for command in unzip md5sum python; do
  command -v "$command" >/dev/null || {
    echo "Required command is unavailable: $command" >&2
    exit 1
  }
done

mkdir -p "$DATA_ROOT" "$CACHE_ROOT"

for index in "${!datasets[@]}"; do
  dataset=${datasets[$index]}
  expected_md5=${md5s[$index]}
  archive="$CACHE_ROOT/${dataset}.npz.zip"
  partial="${archive}.part"
  source_npz="$DATA_ROOT/${dataset}.npz"

  if [[ ! -f "$archive" ]]; then
    echo "Downloading $dataset"
    python tools/download_with_ranges.py \
      --url "https://zenodo.org/api/records/${ZENODO_RECORD}/files/${dataset}.npz.zip/content" \
      --output "$partial" \
      --connections "$DOWNLOAD_CONNECTIONS"
    mv "$partial" "$archive"
  fi

  echo "${expected_md5}  ${archive}" | md5sum --check -
  if [[ -f "$source_npz" \
        && -f "$DATA_ROOT/$dataset/train.npz" \
        && -f "$DATA_ROOT/$dataset/valid.npz" \
        && -f "$DATA_ROOT/$dataset/test.npz" \
        && -f "$DATA_ROOT/$dataset/split_indices.npz" \
        && -f "$DATA_ROOT/$dataset/split_metadata.json" ]]; then
    echo "$dataset has already been extracted and split; skipping preparation."
    continue
  fi
  unzip -jo "$archive" "${dataset}.npz" -d "$DATA_ROOT"

  python tools/split_ares_dataset.py \
    --input "$source_npz" \
    --output-dir "$DATA_ROOT/$dataset" \
    --seed 2024
done

echo "ARES closed/open 2--5-tab datasets are ready under $DATA_ROOT."
