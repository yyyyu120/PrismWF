#!/usr/bin/env python3
"""Train PrismWF on one dataset or a pooled mixed-tab training set."""

import argparse
import json
from pathlib import Path

import torch

from prismwf import build_prismwf
from prismwf.data import load_feature_file, make_loader
from prismwf.engine import train
from prismwf.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--feature-prefix", default="hg8")
    parser.add_argument(
        "--feature-group",
        choices=["all", "packet-count", "transition-count", "transition-interval"],
        default="all",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[2024, 2025, 2026], default=2024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--checkpoint-k", type=int, required=True)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--ablation",
        choices=["full", "no-router", "no-router-no-cross", "single-granularity"],
        default="full",
    )
    return parser.parse_args()


def load_split(
    args: argparse.Namespace,
    split: str,
    sampling_generator: torch.Generator,
    saved_indices: dict[str, list[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[int]]]:
    features, labels = [], []
    used_indices: dict[str, list[int]] = {}
    for dataset in args.datasets:
        x, y = load_feature_file(
            args.data_root / dataset / f"{args.feature_prefix}_{split}.npz",
            feature_group=args.feature_group,
        )
        if split == "train" and args.sample_ratio < 1.0:
            if saved_indices is not None:
                selected = torch.as_tensor(saved_indices[dataset], dtype=torch.long)
            else:
                selected = torch.randperm(len(x), generator=sampling_generator)[
                    : int(len(x) * args.sample_ratio)
                ]
            if len(selected) == 0 or int(selected.max()) >= len(x):
                raise ValueError(f"Invalid saved sampling indices for {dataset}")
            used_indices[dataset] = selected.tolist()
            x, y = x[selected], y[selected]
        features.append(x)
        labels.append(y)
    return torch.cat(features), torch.cat(labels), used_indices


def main() -> None:
    args = parse_args()
    if not 0 < args.sample_ratio <= 1:
        raise ValueError("sample-ratio must be in (0, 1]")
    set_seed(args.seed)
    device = torch.device(args.device)
    sampling_path = args.checkpoint.with_suffix(".sampling.json")
    saved_indices = None
    if sampling_path.exists():
        saved = json.loads(sampling_path.read_text(encoding="utf-8"))
        if (
            saved["datasets"] != args.datasets
            or saved["seed"] != args.seed
            or saved["sample_ratio"] != args.sample_ratio
        ):
            raise ValueError(f"Sampling metadata does not match this run: {sampling_path}")
        saved_indices = saved["indices"]
    sampling_generator = torch.Generator().manual_seed(args.seed)
    train_x, train_y, used_indices = load_split(
        args, "train", sampling_generator, saved_indices
    )
    valid_x, valid_y, _ = load_split(args, "valid", sampling_generator)
    if used_indices and saved_indices is None:
        sampling_path.parent.mkdir(parents=True, exist_ok=True)
        sampling_path.write_text(
            json.dumps(
                {
                    "datasets": args.datasets,
                    "seed": args.seed,
                    "sample_ratio": args.sample_ratio,
                    "indices": used_indices,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    train_loader = make_loader(
        train_x, train_y, args.batch_size, args.seed, True, args.num_workers
    )
    valid_loader = make_loader(
        valid_x, valid_y, args.batch_size, args.seed, False, args.num_workers
    )
    model = build_prismwf(
        num_classes=train_y.shape[1],
        num_layers=args.num_layers,
        ablation=args.ablation,
    )
    history = train(
        model,
        train_loader,
        valid_loader,
        device,
        args.checkpoint,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_k=args.checkpoint_k,
    )
    history_path = args.checkpoint.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    run_path = args.checkpoint.with_suffix(".run.json")
    run_path.write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Checkpoint: {args.checkpoint}")
    print(f"History: {history_path}")
    print(f"Run configuration: {run_path}")


if __name__ == "__main__":
    main()
