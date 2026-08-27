#!/usr/bin/env python3
"""Verify a clean WFDef index and a reproducible sample of raw CSV traces."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--include-trace-ids", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def verify_csv(item: dict[str, Any]) -> None:
    metadata = json.loads(Path(item["metadata_path"]).read_text(encoding="utf-8"))
    traffic = metadata["traffic"]
    timestamps: list[int] = []
    real_bytes = 0
    dummy_bytes = 0
    outgoing_records = 0
    incoming_records = 0

    with Path(item["trace_path"]).open(newline="", encoding="ascii") as source:
        reader = csv.DictReader(source)
        expected_fields = ["timestamp_ns", "real_bytes", "dummy_bytes"]
        if reader.fieldnames != expected_fields:
            raise RuntimeError(
                f"{item['trace_id']}: unexpected CSV header {reader.fieldnames}"
            )
        for row in reader:
            timestamp = int(row["timestamp_ns"])
            real = int(row["real_bytes"])
            dummy = int(row["dummy_bytes"])
            timestamps.append(timestamp)
            real_bytes += abs(real)
            dummy_bytes += abs(dummy)
            outgoing_records += real >= 0
            incoming_records += real < 0

    checks = {
        "packet_records": len(timestamps),
        "real_bytes": real_bytes,
        "dummy_bytes": dummy_bytes,
        "outgoing_records": outgoing_records,
        "incoming_records": incoming_records,
    }
    for field, actual in checks.items():
        if actual != traffic[field]:
            raise RuntimeError(
                f"{item['trace_id']}: {field} is {actual}, metadata has "
                f"{traffic[field]}"
            )
    if timestamps != sorted(timestamps):
        raise RuntimeError(f"{item['trace_id']}: timestamps are not monotonic")
    if not timestamps:
        raise RuntimeError(f"{item['trace_id']}: trace has no rows")
    if timestamps[0] != traffic["first_timestamp_ns"]:
        raise RuntimeError(f"{item['trace_id']}: first timestamp mismatch")
    if timestamps[-1] != traffic["last_timestamp_ns"]:
        raise RuntimeError(f"{item['trace_id']}: last timestamp mismatch")
    if not (
        metadata["trace_window_start_ns"]
        <= timestamps[0]
        <= timestamps[-1]
        <= metadata["trace_window_end_ns"]
    ):
        raise RuntimeError(f"{item['trace_id']}: timestamps outside trace window")


def main() -> int:
    args = parse_args()
    clean = read_jsonl(args.dataset_dir / "clean_index.jsonl")
    excluded = read_jsonl(args.dataset_dir / "excluded_traces.jsonl")
    queue = read_jsonl(args.dataset_dir / "queue_snapshot.jsonl")

    clean_by_id = {item["trace_id"]: item for item in clean}
    excluded_ids = {item["trace_id"] for item in excluded}
    complete_ids = {
        item["trace_id"] for item in queue if item["status"] == "complete"
    }
    if len(clean_by_id) != len(clean):
        raise RuntimeError("clean index contains duplicate trace IDs")
    if len(excluded_ids) != len(excluded):
        raise RuntimeError("excluded index contains duplicate trace IDs")
    if clean_by_id.keys() & excluded_ids:
        raise RuntimeError("clean and excluded indexes overlap")
    if set(clean_by_id) | excluded_ids != complete_ids:
        raise RuntimeError("clean/excluded partition does not match queue snapshot")

    labels = {label for item in clean for label in item["labels"]}
    if labels != set(range(len(labels))):
        raise RuntimeError("clean labels are not contiguous from zero")

    sample_size = min(args.sample_size, len(clean))
    sample_ids = {
        item["trace_id"]
        for item in random.Random(args.seed).sample(clean, sample_size)
    }
    forced_ids: set[str] = set()
    if args.include_trace_ids:
        forced_ids = set(json.loads(args.include_trace_ids.read_text(encoding="utf-8")))
        missing_forced = forced_ids - set(clean_by_id)
        if missing_forced:
            raise RuntimeError(
                f"forced trace IDs are not clean: {sorted(missing_forced)[:10]}"
            )
        sample_ids.update(forced_ids)

    for trace_id in sorted(sample_ids):
        verify_csv(clean_by_id[trace_id])

    result = {
        "queue_statuses": dict(sorted(Counter(x["status"] for x in queue).items())),
        "partition_ok": True,
        "clean_traces": len(clean),
        "excluded_traces": len(excluded),
        "classes": len(labels),
        "csv_traces_checked": len(sample_ids),
        "random_sample_size": sample_size,
        "random_seed": args.seed,
        "forced_traces_checked": len(forced_ids),
        "csv_rows_bytes_directions_match": True,
        "timestamps_monotonic_and_in_window": True,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
