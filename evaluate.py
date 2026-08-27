#!/usr/bin/env python3
"""Evaluate a PrismWF checkpoint with the fixed-K protocol."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prismwf import build_prismwf
from prismwf.data import load_feature_file, make_loader
from prismwf.engine import evaluate_fixed_k, load_checkpoint
from prismwf.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature-prefix", default="hg8")
    parser.add_argument(
        "--feature-group",
        choices=["all", "packet-count", "transition-count", "transition-interval"],
        default="all",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tabs", type=int, choices=[2, 3, 4, 5], required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--ablation",
        choices=["full", "no-router", "no-router-no-cross", "single-granularity"],
        default="full",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    x, y = load_feature_file(
        args.data_root / args.dataset / f"{args.feature_prefix}_{args.split}.npz",
        feature_group=args.feature_group,
    )
    loader = make_loader(x, y, args.batch_size, args.seed, False, args.num_workers)
    model = load_checkpoint(
        build_prismwf(
            y.shape[1],
            num_layers=args.num_layers,
            ablation=args.ablation,
        ),
        args.checkpoint,
        device,
    )
    metrics, labels, scores = evaluate_fixed_k(model, loader, device, args.tabs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savez_compressed(args.output.with_suffix(".npz"), y=labels, score=scores)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
