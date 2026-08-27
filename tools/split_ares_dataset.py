#!/usr/bin/env python3
"""Reproduce the WFlib instance-level train/validation/test split."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def split_indices(
    sample_count: int,
    seed: int = 2024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the WFlib 81/9/10 split indices for a multi-label dataset."""
    indices = np.arange(sample_count, dtype=np.int64)
    train_valid, test = train_test_split(
        indices,
        train_size=0.9,
        random_state=seed,
        shuffle=True,
    )
    train, valid = train_test_split(
        train_valid,
        train_size=0.9,
        random_state=seed,
        shuffle=True,
    )
    return train, valid, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split one released ARES multi-tab NPZ using the WFlib protocol."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.input) as data:
        if "X" not in data or "y" not in data:
            raise KeyError(f"{args.input} must contain X and y arrays")
        x = data["X"]
        y = data["y"]

    if len(x) != len(y):
        raise ValueError(f"X/y sample counts differ: {len(x)} != {len(y)}")
    if len(x) < 10:
        raise ValueError("At least 10 samples are required for an 81/9/10 split")

    train, valid, test = split_indices(len(x), args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": train, "valid": valid, "test": test}
    for name, selected in splits.items():
        np.savez(args.output_dir / f"{name}.npz", X=x[selected], y=y[selected])

    index_path = args.output_dir / "split_indices.npz"
    np.savez(index_path, **splits)
    metadata = {
        "source": str(args.input),
        "seed": args.seed,
        "protocol": "WFlib instance-level split: 90/10 followed by 90/10",
        "sample_count": len(x),
        "counts": {name: len(selected) for name, selected in splits.items()},
        "index_file": index_path.name,
    }
    (args.output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
