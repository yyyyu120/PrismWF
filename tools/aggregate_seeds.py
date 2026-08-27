#!/usr/bin/env python3
"""Aggregate metric JSON files as mean and sample standard deviation."""

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    keys = records[0].keys()
    if any(record.keys() != records[0].keys() for record in records):
        raise ValueError("Metric files have different keys")
    summary = {
        key: {
            "mean": statistics.mean(record[key] for record in records),
            "sample_std": statistics.stdev(record[key] for record in records),
        }
        for key in keys
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
