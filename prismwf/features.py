"""Six-channel robust trace representation used by PrismWF."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
from tqdm import tqdm


def align_raw_traces(traces: np.ndarray, length: int = 10000) -> np.ndarray:
    """Crop or zero-pad raw traces to the event length used in the paper."""
    if traces.shape[-1] > length:
        return traces[..., :length]
    if traces.shape[-1] < length:
        widths = [(0, 0)] * traces.ndim
        widths[-1] = (0, length - traces.shape[-1])
        return np.pad(traces, widths, mode="constant")
    return traces


def slot_features(packets: np.ndarray) -> np.ndarray:
    directions = np.sign(packets)
    if np.any(directions == 0):
        raise ValueError("Packet sequences must not contain zero-valued events")
    transitions = np.diff(directions)
    times = np.abs(packets)

    pos_to_neg = np.flatnonzero(transitions < 0)
    neg_to_pos = np.flatnonzero(transitions > 0)
    pn_interval = (
        np.mean(times[pos_to_neg + 1] - times[pos_to_neg]) if len(pos_to_neg) else 0.0
    )
    np_interval = (
        np.mean(times[neg_to_pos + 1] - times[neg_to_pos]) if len(neg_to_pos) else 0.0
    )
    return np.asarray(
        [
            np.sum(packets > 0),
            np.sum(packets < 0),
            len(pos_to_neg),
            len(neg_to_pos),
            pn_interval,
            np_interval,
        ],
        dtype=np.float32,
    )


def _encode_one(
    index: int,
    sequence: np.ndarray,
    slot_ms: int,
    max_slots: int,
    include_tail_in_last_slot: bool = True,
) -> tuple[int, np.ndarray]:
    packets = np.trim_zeros(np.asarray(sequence).squeeze(), "fb")
    encoded = np.zeros((6, max_slots), dtype=np.float32)
    if not len(packets):
        return index, encoded

    absolute_times = np.abs(packets)
    start_time = absolute_times[0]
    start_position = 0
    for slot in range(max_slots):
        if slot == max_slots - 1 and include_tail_in_last_slot:
            end_position = len(packets)
        else:
            end_position = np.searchsorted(
                absolute_times, start_time + (slot + 1) * slot_ms
            )
        if start_position < end_position:
            encoded[:, slot] = slot_features(packets[start_position:end_position])
        start_position = end_position
        if start_position >= len(packets):
            break
    return index, encoded


def encode_traces(
    sequences: np.ndarray,
    slot_ms: int = 20,
    max_loading_seconds: int = 160,
    num_workers: int = 8,
    include_tail_in_last_slot: bool = True,
) -> np.ndarray:
    """Encode signed timestamp traces into an `(N, 6, L)` feature tensor."""
    traces_ms = np.asarray(sequences).copy()
    traces_ms *= 1000
    max_slots = int(max_loading_seconds * 1000 / slot_ms)
    output = np.zeros((len(traces_ms), 6, max_slots), dtype=np.float32)
    workers = max(1, min(num_workers, len(traces_ms)))
    next_index = 0
    pending = set()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        while next_index < len(traces_ms) and len(pending) < workers * 2:
            pending.add(
                executor.submit(
                    _encode_one,
                    next_index,
                    traces_ms[next_index],
                    slot_ms,
                    max_slots,
                    include_tail_in_last_slot,
                )
            )
            next_index += 1

        with tqdm(total=len(traces_ms), desc="Encoding traces") as progress:
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    index, encoded = future.result()
                    output[index] = encoded
                    progress.update(1)
                    if next_index < len(traces_ms):
                        pending.add(
                            executor.submit(
                                _encode_one,
                                next_index,
                                traces_ms[next_index],
                                slot_ms,
                                max_slots,
                                include_tail_in_last_slot,
                            )
                        )
                        next_index += 1
    return output
