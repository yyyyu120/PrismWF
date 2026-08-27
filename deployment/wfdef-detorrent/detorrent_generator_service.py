#!/usr/bin/env python3
"""Causal inference service for a generator reproduced with DeTorrent's artifact."""

import argparse
import json
import os
import socketserver
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch


NUM_BINS = 256
NOISE_DIM = 32
SESSION_TTL_SECONDS = 600


class GeneratorRuntime:
    def __init__(
        self, artifact_root, checkpoint, calibration_hist, seed, samples, audit_log=None
    ):
        sys.path.insert(0, str(artifact_root))
        from models import LSTM_ATTACK

        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.model = LSTM_ATTACK(NOISE_DIM, 128)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model.eval()
        self.registry_lock = threading.Lock()
        self.rng_lock = threading.Lock()
        self.audit_lock = threading.Lock()
        self.sessions = {}
        self.audit_log = audit_log
        self.mean_raw_total = self._calibrate(calibration_hist, samples)

    def _expire_sessions(self):
        cutoff = time.monotonic() - SESSION_TTL_SECONDS
        expired = [
            session_id
            for session_id, session in self.sessions.items()
            if session["updated"] < cutoff
        ]
        for session_id in expired:
            self.sessions.pop(session_id, None)
            self._audit({"op": "expire", "session_id": session_id})

    def _audit(self, event):
        if self.audit_log is None:
            return
        record = {"timestamp": time.time(), **event}
        with self.audit_lock:
            with self.audit_log.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")

    def _new_input(self):
        model_input = self.rng.standard_normal((1, NUM_BINS, NOISE_DIM)).astype(
            np.float32
        )
        model_input[0, :, -1] = 0
        model_input[0, :, 5] = np.arange(NUM_BINS, dtype=np.float32) / 10
        return model_input

    def _infer(self, model_input):
        with torch.inference_mode():
            return self.model(torch.from_numpy(model_input))[0, :, 0].numpy()

    def _calibrate(self, histogram_path, samples):
        histograms = np.load(histogram_path, mmap_mode="r")
        if histograms.ndim not in (2, 3) or histograms.shape[1] != NUM_BINS:
            raise ValueError("calibration histogram must have shape [N, 256] or [N, 256, 1]")

        sample_count = min(samples, len(histograms))
        indices = self.rng.choice(len(histograms), sample_count, replace=False)
        totals = []
        for index in indices:
            model_input = self._new_input()
            model_input[0, :, -1] = np.asarray(histograms[index]).reshape(NUM_BINS)
            totals.append(float(self._infer(model_input).sum()))

        mean_total = float(np.mean(totals))
        if not np.isfinite(mean_total) or mean_total <= 0:
            raise ValueError("generator calibration produced a non-positive scale")
        return mean_total

    def start(self, budget):
        session_id = uuid.uuid4().hex
        with self.rng_lock:
            model_input = self._new_input()
            # A zero padding budget is the cell-framed Null control. Keep the
            # introductory bin empty so the control emits no dummy packets.
            first_bin = 0 if budget == 0 else int(self.rng.integers(0, 20))
        with self.registry_lock:
            self._expire_sessions()
            self.sessions[session_id] = {
                "input": model_input,
                "next_bin": 1,
                "budget": float(budget),
                "created": time.monotonic(),
                "updated": time.monotonic(),
                "lock": threading.Lock(),
            }
        self._audit(
            {
                "op": "start",
                "session_id": session_id,
                "budget": int(budget),
                "bin": 0,
                "dummy_packets": first_bin,
                "mean_raw_total": self.mean_raw_total,
            }
        )
        return session_id, first_bin

    def step(self, session_id, previous_bin, real_download_packets):
        if previous_bin < 0 or previous_bin >= NUM_BINS - 1:
            raise ValueError("previous_bin must be in [0, 254]")
        if real_download_packets < 0:
            raise ValueError("real_download_packets must be non-negative")

        with self.registry_lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise KeyError("unknown session")
        with session["lock"]:
            if session["next_bin"] != previous_bin + 1:
                raise ValueError(
                    "out-of-order step: expected previous_bin "
                    f"{session['next_bin'] - 1}, received {previous_bin}"
                )

            session["input"][0, previous_bin, -1] = float(real_download_packets)
            session["updated"] = time.monotonic()
            raw = float(self._infer(session["input"])[previous_bin])
            dummy_packets = int(
                np.rint(raw * session["budget"] / self.mean_raw_total)
            )
            session["next_bin"] += 1
            dummy_packets = max(0, dummy_packets)
        self._audit(
            {
                "op": "step",
                "session_id": session_id,
                "previous_bin": previous_bin,
                "real_download_packets": real_download_packets,
                "dummy_packets": dummy_packets,
            }
        )
        return dummy_packets

    def close(self, session_id):
        with self.registry_lock:
            closed = self.sessions.pop(session_id, None) is not None
        self._audit(
            {"op": "close", "session_id": session_id, "closed": closed}
        )
        return closed

    def schedule(self, real_download_bins, budget):
        """Compatibility endpoint used only for offline smoke tests."""
        real = np.asarray(real_download_bins, dtype=np.float32)
        if real.shape != (NUM_BINS,):
            raise ValueError("real_download_bins must contain exactly 256 values")

        with self.rng_lock:
            model_input = self._new_input()
            model_input[0, :, -1] = real
            weights = self._infer(model_input)
            schedule = np.rint(weights * float(budget) / self.mean_raw_total).astype(
                np.int64
            )
            schedule = np.roll(schedule, 1)
            schedule[0] = int(self.rng.integers(0, 20))
        return schedule.tolist()

    def status(self):
        with self.registry_lock:
            self._expire_sessions()
            return {
                "active_sessions": len(self.sessions),
                "mean_raw_total": self.mean_raw_total,
            }


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            try:
                request = json.loads(line)
                operation = request.get("op", "schedule")
                if operation == "start":
                    session_id, first_bin = self.server.runtime.start(
                        request.get("budget", 3000)
                    )
                    response = {
                        "session_id": session_id,
                        "bin": 0,
                        "dummy_packets": first_bin,
                    }
                elif operation == "step":
                    dummy_packets = self.server.runtime.step(
                        request["session_id"],
                        int(request["previous_bin"]),
                        int(request["real_download_packets"]),
                    )
                    response = {
                        "bin": int(request["previous_bin"]) + 1,
                        "dummy_packets": dummy_packets,
                    }
                elif operation == "close":
                    response = {
                        "closed": self.server.runtime.close(request["session_id"])
                    }
                elif operation == "status":
                    response = self.server.runtime.status()
                elif operation == "schedule":
                    response = {
                        "padding_bins": self.server.runtime.schedule(
                            request["real_download_bins"], request.get("budget", 3000)
                        )
                    }
                else:
                    raise ValueError(f"unsupported operation: {operation}")
            except Exception as error:
                response = {"error": str(error)}
            self.wfile.write((json.dumps(response) + "\n").encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-hist", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=1000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19991)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--audit-log", type=Path)
    args = parser.parse_args()

    # Small recurrent inference is faster and more predictable with a bounded
    # CPU pool than with PyTorch's host-wide default thread count.
    torch_threads = max(1, int(os.environ.get("DETORRENT_TORCH_THREADS", "1")))
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)

    runtime = GeneratorRuntime(
        args.artifact_root,
        args.checkpoint,
        args.calibration_hist,
        args.seed,
        args.calibration_samples,
        args.audit_log,
    )
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((args.host, args.port), RequestHandler) as server:
        server.daemon_threads = True
        server.runtime = runtime
        server.serve_forever()


if __name__ == "__main__":
    main()
