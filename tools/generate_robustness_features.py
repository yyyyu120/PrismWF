#!/usr/bin/env python3
"""Generate deterministic network-perturbed robust trace features."""

import argparse
from pathlib import Path

import numpy as np

from prismwf.features import align_raw_traces, encode_traces


def perturb_trace(
    trace: np.ndarray,
    condition: str,
    value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float32)
    nonzero = trace[trace != 0].copy()
    if not len(nonzero):
        return trace.copy()

    signs = np.sign(nonzero)
    times = np.abs(nonzero)
    if condition == "packet-loss":
        keep = rng.random(len(nonzero)) >= value
        keep[0] = True
        changed = nonzero[keep]
    elif condition == "latency-scale":
        start = times[0]
        changed = signs * (start + (times - start) * value)
    elif condition == "midpoint-delay-offset":
        times[len(nonzero) // 2 :] += value / 1000.0
        changed = signs * times
    else:
        raise ValueError(condition)

    output = np.zeros_like(trace)
    output[: len(changed)] = changed
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=["packet-loss", "latency-scale", "midpoint-delay-offset"],
        required=True,
    )
    parser.add_argument("--value", type=float, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--raw-length", type=int, default=10000)
    parser.add_argument("--slot-ms", type=int, default=20)
    parser.add_argument("--max-loading-seconds", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--strict-window", action="store_true")
    args = parser.parse_args()

    with np.load(args.input) as data:
        raw = align_raw_traces(data["X"], args.raw_length)
        labels = data["y"]
    seeds = np.random.SeedSequence(args.seed).spawn(len(raw))
    perturbed = np.empty_like(raw, dtype=np.float32)
    for index, child_seed in enumerate(seeds):
        perturbed[index] = perturb_trace(
            raw[index],
            args.condition,
            args.value,
            np.random.default_rng(child_seed),
        )
    features = encode_traces(
        perturbed,
        slot_ms=args.slot_ms,
        max_loading_seconds=args.max_loading_seconds,
        num_workers=args.num_workers,
        include_tail_in_last_slot=not args.strict_window,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=features, y=labels)
    print(f"Saved {features.shape} to {args.output}")


if __name__ == "__main__":
    main()
