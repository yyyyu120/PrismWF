#!/usr/bin/env python3
"""Generate PrismWF robust trace features and ARES MTAF from full trace CSVs."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np


_HG8_MEMMAP = None
_MTAF_MEMMAP = None
_INTERVAL_MS = None
_MAX_SLOTS = None
_EVENT_LIMIT = None
_FRAME_BYTES = None
_CELL_BYTES = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-ms", type=float, default=20.0)
    parser.add_argument("--max-slots", type=int, default=8_000)
    parser.add_argument(
        "--event-limit",
        type=int,
        default=0,
        help="Maximum events per trace. Zero keeps the complete trace.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--frame-bytes",
        type=int,
        default=536,
        help="Require a fixed signed record size. Zero accepts native variable-size records.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--cell-bytes",
        type=int,
        default=536,
        help="Reconstruct fixed-size Tor cells from signed byte records.",
    )
    return parser.parse_args()


def load_index(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def init_worker(
    hg8_path: str,
    mtaf_path: str,
    samples: int,
    interval_ms: float,
    max_slots: int,
    event_limit: int,
    frame_bytes: int,
    cell_bytes: int,
) -> None:
    global _HG8_MEMMAP, _MTAF_MEMMAP
    global _INTERVAL_MS, _MAX_SLOTS, _EVENT_LIMIT, _FRAME_BYTES, _CELL_BYTES
    _HG8_MEMMAP = np.memmap(
        hg8_path, mode="r+", dtype=np.float32, shape=(samples, 6, max_slots)
    )
    _MTAF_MEMMAP = np.memmap(
        mtaf_path, mode="r+", dtype=np.float32, shape=(samples, 8, max_slots)
    )
    _INTERVAL_MS = interval_ms
    _MAX_SLOTS = max_slots
    _EVENT_LIMIT = event_limit
    _FRAME_BYTES = frame_bytes
    _CELL_BYTES = cell_bytes


def flush_slot(
    hg8: np.ndarray,
    mtaf: np.ndarray,
    slot: int,
    state: dict,
) -> None:
    if slot < 0 or slot >= hg8.shape[1] or state["events"] == 0:
        return

    pos_count = state["pos_count"]
    neg_count = state["neg_count"]
    pos_to_neg = state["pos_to_neg"]
    neg_to_pos = state["neg_to_pos"]

    hg8[0, slot] = pos_count
    hg8[1, slot] = neg_count
    hg8[2, slot] = pos_to_neg
    hg8[3, slot] = neg_to_pos
    if pos_to_neg:
        hg8[4, slot] = state["pos_to_neg_gap_sum"] / pos_to_neg
    if neg_to_pos:
        hg8[5, slot] = state["neg_to_pos_gap_sum"] / neg_to_pos

    pos_bursts = state["pos_bursts"]
    neg_bursts = state["neg_bursts"]
    if state["run_dir"] > 0:
        pos_bursts += 1
    elif state["run_dir"] < 0:
        neg_bursts += 1

    mtaf[0, slot] = pos_count
    mtaf[1, slot] = neg_count
    if state["pos_first"] is not None:
        mtaf[2, slot] = state["pos_last"] - state["pos_first"]
    if state["neg_first"] is not None:
        mtaf[3, slot] = state["neg_last"] - state["neg_first"]
    mtaf[4, slot] = pos_bursts
    mtaf[5, slot] = neg_bursts
    if pos_bursts:
        mtaf[6, slot] = pos_count / pos_bursts
    if neg_bursts:
        mtaf[7, slot] = neg_count / neg_bursts


def empty_state() -> dict:
    return {
        "events": 0,
        "pos_count": 0,
        "neg_count": 0,
        "pos_first": None,
        "pos_last": None,
        "neg_first": None,
        "neg_last": None,
        "previous_dir": 0,
        "previous_time": 0.0,
        "pos_to_neg": 0,
        "neg_to_pos": 0,
        "pos_to_neg_gap_sum": 0.0,
        "neg_to_pos_gap_sum": 0.0,
        "run_dir": 0,
        "pos_bursts": 0,
        "neg_bursts": 0,
    }


def update_state(state: dict, direction: int, time_ms: float) -> None:
    state["events"] += 1
    if direction > 0:
        state["pos_count"] += 1
        if state["pos_first"] is None:
            state["pos_first"] = time_ms
        state["pos_last"] = time_ms
    else:
        state["neg_count"] += 1
        if state["neg_first"] is None:
            state["neg_first"] = time_ms
        state["neg_last"] = time_ms

    previous_dir = state["previous_dir"]
    if previous_dir and direction != previous_dir:
        gap = time_ms - state["previous_time"]
        if previous_dir > 0:
            state["pos_to_neg"] += 1
            state["pos_to_neg_gap_sum"] += gap
            state["pos_bursts"] += 1
        else:
            state["neg_to_pos"] += 1
            state["neg_to_pos_gap_sum"] += gap
            state["neg_bursts"] += 1
        state["run_dir"] = direction
    elif not previous_dir:
        state["run_dir"] = direction

    state["previous_dir"] = direction
    state["previous_time"] = time_ms


def convert_one(task: tuple[int, dict]) -> tuple[int, int, int, float, int, int, int, int]:
    row_index, record = task
    trace_path = Path(record["trace_path"])
    hg8 = np.zeros((6, _MAX_SLOTS), dtype=np.float32)
    mtaf = np.zeros((8, _MAX_SLOTS), dtype=np.float32)

    first_timestamp = None
    previous_timestamp = None
    first_time_ms = float(np.float32(1e-6) * np.float32(1000.0))
    current_slot = -1
    state = empty_state()
    events = 0
    last_relative_ms = 0.0
    direction_remainders = {1: 0, -1: 0}
    direction_bytes = {1: 0, -1: 0}

    with trace_path.open(encoding="utf-8") as handle:
        header = next(handle, "").strip()
        if header != "timestamp_ns,real_bytes,dummy_bytes":
            raise ValueError(f"Unexpected header in {trace_path}: {header!r}")

        for line in handle:
            timestamp, real_bytes, dummy_bytes = map(int, line.rstrip().split(","))
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(f"Non-monotonic timestamp in {trace_path}")
            previous_timestamp = timestamp

            if real_bytes and dummy_bytes and (real_bytes > 0) != (dummy_bytes > 0):
                raise ValueError(f"Conflicting directions in {trace_path}")
            total_bytes = real_bytes + dummy_bytes
            if total_bytes == 0:
                raise ValueError(f"Zero-byte record in {trace_path}")
            if _FRAME_BYTES and abs(total_bytes) != _FRAME_BYTES:
                raise ValueError(
                    f"Unexpected frame size {total_bytes} in {trace_path}; "
                    f"expected +/-{_FRAME_BYTES}"
                )

            if first_timestamp is None:
                first_timestamp = timestamp
                relative_ms = first_time_ms
            else:
                relative_seconds = np.float32((timestamp - first_timestamp) * 1e-9)
                relative_ms = float(relative_seconds * np.float32(1000.0))

            slot = int((relative_ms - first_time_ms) // _INTERVAL_MS)
            slot = max(slot, 0)
            if slot >= _MAX_SLOTS:
                break
            if slot != current_slot:
                flush_slot(hg8, mtaf, current_slot, state)
                current_slot = slot
                state = empty_state()

            direction = 1 if total_bytes > 0 else -1
            direction_bytes[direction] += abs(total_bytes)
            available = direction_remainders[direction] + abs(total_bytes)
            cells, direction_remainders[direction] = divmod(available, _CELL_BYTES)
            if _EVENT_LIMIT:
                cells = min(cells, _EVENT_LIMIT - events)
            for _ in range(cells):
                update_state(state, direction, relative_ms)
            events += cells
            last_relative_ms = relative_ms
            if _EVENT_LIMIT and events >= _EVENT_LIMIT:
                break

    flush_slot(hg8, mtaf, current_slot, state)
    if events == 0:
        raise ValueError(f"Empty trace: {trace_path}")

    _HG8_MEMMAP[row_index] = hg8
    _MTAF_MEMMAP[row_index] = mtaf
    expected_events = (
        direction_bytes[1] // _CELL_BYTES + direction_bytes[-1] // _CELL_BYTES
    )
    if not _EVENT_LIMIT and events != expected_events:
        raise ValueError(
            f"Cell conservation failed for {trace_path}: {events} != {expected_events}"
        )
    return (
        row_index,
        events,
        current_slot + 1,
        last_relative_ms * 1e-3,
        direction_bytes[1],
        direction_bytes[-1],
        direction_remainders[1],
        direction_remainders[-1],
    )


def write_split(
    split: str,
    records: list[dict],
    output_dir: Path,
    args: argparse.Namespace,
    num_classes: int,
) -> dict:
    samples = len(records)
    hg8_memmap_path = output_dir / f".{split}.hg8.float32.dat"
    mtaf_memmap_path = output_dir / f".{split}.mtaf.float32.dat"
    hg8_shape = (samples, 6, args.max_slots)
    mtaf_shape = (samples, 8, args.max_slots)

    for path, shape in ((hg8_memmap_path, hg8_shape), (mtaf_memmap_path, mtaf_shape)):
        array = np.memmap(path, mode="w+", dtype=np.float32, shape=shape)
        array[:] = 0
        array.flush()
        del array

    labels = np.zeros((samples, num_classes), dtype=np.int8)
    for index, record in enumerate(records):
        labels[index, record["labels"]] = 1

    event_counts = np.empty(samples, dtype=np.int32)
    occupied_slots = np.empty(samples, dtype=np.int32)
    observed_seconds = np.empty(samples, dtype=np.float32)
    positive_bytes = np.empty(samples, dtype=np.int64)
    negative_bytes = np.empty(samples, dtype=np.int64)
    positive_remainders = np.empty(samples, dtype=np.int32)
    negative_remainders = np.empty(samples, dtype=np.int32)
    context = mp.get_context("fork")
    with context.Pool(
        processes=min(args.workers, samples),
        initializer=init_worker,
        initargs=(
            str(hg8_memmap_path),
            str(mtaf_memmap_path),
            samples,
            args.interval_ms,
            args.max_slots,
            args.event_limit,
            args.frame_bytes,
            args.cell_bytes,
        ),
    ) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(convert_one, enumerate(records), chunksize=4), start=1
        ):
            (
                index,
                events,
                slots,
                seconds,
                pos_bytes,
                neg_bytes,
                pos_remainder,
                neg_remainder,
            ) = result
            event_counts[index] = events
            occupied_slots[index] = slots
            observed_seconds[index] = seconds
            positive_bytes[index] = pos_bytes
            negative_bytes[index] = neg_bytes
            positive_remainders[index] = pos_remainder
            negative_remainders[index] = neg_remainder
            if completed % 500 == 0 or completed == samples:
                print(f"[{split}] {completed}/{samples} traces", flush=True)

    outputs = {}
    for feature, path, shape in (
        ("hg8", hg8_memmap_path, hg8_shape),
        ("mtaf", mtaf_memmap_path, mtaf_shape),
    ):
        array = np.memmap(path, mode="r", dtype=np.float32, shape=shape)
        output_path = output_dir / f"{feature}_{split}.npz"
        np.savez(output_path, X=array, y=labels)
        outputs[feature] = str(output_path.resolve())
        del array
        path.unlink()

    return {
        "samples": samples,
        "events_min": int(event_counts.min()),
        "events_median": float(np.median(event_counts)),
        "events_max": int(event_counts.max()),
        "occupied_slots_median": float(np.median(occupied_slots)),
        "observed_seconds_median": float(np.median(observed_seconds)),
        "cell_conservation_all_passed": bool(np.all(
            event_counts
            == positive_bytes // args.cell_bytes + negative_bytes // args.cell_bytes
        )),
        "positive_residual_bytes_max": int(positive_remainders.max()),
        "negative_residual_bytes_max": int(negative_remainders.max()),
        "outputs": outputs,
    }


def main() -> None:
    args = parse_args()
    if args.interval_ms <= 0 or args.max_slots <= 0 or args.event_limit < 0:
        raise ValueError("Invalid interval, max-slots, or event-limit")

    records = load_index(args.index)
    num_classes = max(label for record in records for label in record["labels"]) + 1
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    summary = {
        "source_index": str(args.index.resolve()),
        "num_classes": num_classes,
        "interval_ms": args.interval_ms,
        "max_slots": args.max_slots,
        "window_seconds": args.interval_ms * args.max_slots / 1000.0,
        "event_limit": args.event_limit,
        "frame_bytes": args.frame_bytes,
        "cell_bytes": args.cell_bytes,
        "record_normalization": "per-direction byte carry reconstructed into fixed-size cells",
        "splits": {},
    }
    split_names = {"train": "train", "valid": "validation", "test": "test"}
    for output_split, source_split in split_names.items():
        split_records = [record for record in records if record["split"] == source_split]
        split_records.sort(
            key=lambda record: (
                record.get("combination_id", record.get("pair_id")),
                record["repetition"],
            )
        )
        summary["splits"][output_split] = write_split(
            output_split, split_records, args.output_dir, args, num_classes
        )

    summary_path = args.output_dir / "fulltrace_feature_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
