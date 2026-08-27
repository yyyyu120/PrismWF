#!/usr/bin/env python3
"""Profile PrismWF parameters, FLOPs, latency, throughput, and peak memory."""

import argparse
import io
import json
import statistics
import time
from pathlib import Path

import torch

from prismwf import PrismWF


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(model: torch.nn.Module, x: torch.Tensor, repeats: int) -> list[float]:
    device = x.device
    with torch.inference_mode():
        for _ in range(20):
            model(x)
        synchronize(device)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--throughput-batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = PrismWF(args.num_classes, num_layers=args.num_layers).to(device).eval()
    single = torch.randn(1, 6, 8000, device=device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode(), torch.profiler.profile(activities=activities, with_flops=True) as prof:
        model(single)
        synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            model(single)
        synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_memory = None

    latency = timed(model, single, args.repeats)
    batch = torch.randn(args.throughput_batch_size, 6, 8000, device=device)
    batch_latency = timed(model, batch, max(20, args.repeats // 4))
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    results = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_size_mb": len(buffer.getbuffer()) / (1024**2),
        "profiled_flops": sum(event.flops or 0 for event in prof.key_averages()),
        "latency_mean_ms": statistics.mean(latency),
        "latency_std_ms": statistics.stdev(latency),
        "throughput_traces_per_second": args.throughput_batch_size
        / (statistics.mean(batch_latency) / 1000),
        "peak_memory_mb_batch1": peak_memory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
