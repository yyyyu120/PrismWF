#!/usr/bin/env python3
"""Select a validation threshold and evaluate without access to the true K."""

import argparse
import json
from pathlib import Path

import numpy as np

from prismwf.metrics import select_threshold, unknown_k_metrics


def load_predictions(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    items = [np.load(path) for path in paths]
    return np.concatenate([item["y"] for item in items]), np.concatenate(
        [item["score"] for item in items]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", nargs="+", type=Path, required=True)
    parser.add_argument("--test", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    valid_y, valid_score = load_predictions(args.validation)
    threshold, validation_metrics = select_threshold(valid_y, valid_score)
    results = {
        "threshold": threshold,
        "selection": "maximum pooled-validation micro-F1",
        "validation": validation_metrics,
        "test": {},
    }
    for path in args.test:
        data = np.load(path)
        results["test"][path.stem] = unknown_k_metrics(
            data["y"], data["score"], threshold
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
