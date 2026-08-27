"""Training and inference routines shared by the command-line tools."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import fixed_k_metrics, map_at_k


def predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    labels, scores = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            labels.append(y.numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores)


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    checkpoint: Path,
    epochs: int = 80,
    learning_rate: float = 5e-4,
    checkpoint_k: int = 2,
) -> list[dict[str, float]]:
    """Train PrismWF and select the checkpoint by validation MAP@K."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.74)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.to(device)

    best_score = float("-inf")
    history = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(x)
            sample_count += len(x)

        y_valid, score_valid = predict(model, valid_loader, device)
        validation_map = map_at_k(y_valid, score_valid, checkpoint_k)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            f"validation_MAP@{checkpoint_k}": validation_map,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_map > best_score:
            best_score = validation_map
            torch.save(model.state_dict(), checkpoint)
        scheduler.step()
    return history


def evaluate_fixed_k(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    k: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    labels, scores = predict(model, loader, device)
    return fixed_k_metrics(labels, scores, k), labels, scores


def load_checkpoint(
    model: torch.nn.Module, checkpoint: str | Path, device: torch.device
) -> torch.nn.Module:
    state = torch.load(Path(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model.to(device)
