#!/usr/bin/env python3
"""Generate PrismWF features from signed timestamp traces."""

import argparse
from pathlib import Path

import numpy as np

from prismwf.features import align_raw_traces, encode_traces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Raw NPZ containing X and y")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-length", type=int, default=10000)
    parser.add_argument("--slot-ms", type=int, default=20)
    parser.add_argument("--max-loading-seconds", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--strict-window",
        action="store_true",
        help="Discard events after the configured window instead of reproducing the legacy tail slot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.input)
    traces = align_raw_traces(data["X"], args.raw_length)
    features = encode_traces(
        traces,
        slot_ms=args.slot_ms,
        max_loading_seconds=args.max_loading_seconds,
        num_workers=args.num_workers,
        include_tail_in_last_slot=not args.strict_window,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=features, y=data["y"])
    print(f"Saved {features.shape} to {args.output}")


if __name__ == "__main__":
    main()
