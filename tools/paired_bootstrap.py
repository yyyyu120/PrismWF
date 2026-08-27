#!/usr/bin/env python3
"""Paired percentile-bootstrap confidence intervals over test traces."""

import argparse
import json
from pathlib import Path

import numpy as np


def per_trace_scores(y: np.ndarray, score: np.ndarray, k: int) -> dict[str, np.ndarray]:
    ranking = np.argsort(score, axis=1)[:, ::-1]
    prefixes = []
    for cutoff in range(1, k + 1):
        selected = ranking[:, :cutoff]
        hits = np.take_along_axis(y, selected, axis=1).sum(axis=1)
        prefixes.append(hits / cutoff)
    return {f"P@{k}": prefixes[-1], f"MAP@{k}": np.mean(prefixes, axis=0)}


def confidence_interval(diff: np.ndarray, repeats: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 250):
        count = min(250, repeats - start)
        indices = rng.integers(0, len(diff), size=(count, len(diff)))
        estimates[start : start + count] = diff[indices].mean(axis=1)
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "mean_difference": float(diff.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prismwf", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--tabs", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prismwf = np.load(args.prismwf)
    baseline = np.load(args.baseline)
    if not np.array_equal(prismwf["y"], baseline["y"]):
        raise ValueError("Prediction files do not have identical labels and sample order")
    first = per_trace_scores(prismwf["y"], prismwf["score"], args.tabs)
    second = per_trace_scores(baseline["y"], baseline["score"], args.tabs)
    results = {
        metric: confidence_interval(first[metric] - second[metric], args.repeats, args.seed + i)
        for i, metric in enumerate(first)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
