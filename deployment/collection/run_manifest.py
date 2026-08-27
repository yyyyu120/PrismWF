#!/usr/bin/env python3
"""Run a collection manifest across isolated Tor/WFDefProxy lanes."""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--lanes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--browser-binary", type=Path, required=True)
    parser.add_argument("--default-profile", type=Path, required=True)
    parser.add_argument("--geckodriver", type=Path, required=True)
    parser.add_argument("--capture-seconds", type=float, default=160.0)
    parser.add_argument("--tail-seconds", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--skip-tor-check", action="store_true")
    parser.add_argument("--skip-newnym", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    lanes = json.loads(args.lanes.read_text(encoding="utf-8"))
    if not records or not lanes:
        raise ValueError("The manifest and lane list must be non-empty")
    args.output_root.mkdir(parents=True, exist_ok=True)

    work: queue.Queue[dict] = queue.Queue()
    for record in records:
        work.put(record)
    lock = threading.Lock()
    completed: list[str] = []
    failed: list[dict] = []
    collector = Path(__file__).with_name("collect_multitab.py")

    def collect_lane(lane: dict) -> None:
        while True:
            try:
                record = work.get_nowait()
            except queue.Empty:
                return
            trace_id = record["trace_id"]
            command = [
                sys.executable,
                str(collector),
                "--trace-id",
                trace_id,
                "--output-root",
                str(args.output_root),
                "--browser-binary",
                str(args.browser_binary),
                "--default-profile",
                str(args.default_profile),
                "--geckodriver",
                str(args.geckodriver),
                "--pt-log",
                lane["pt_log"],
                "--control-host",
                lane.get("control_host", "127.0.0.1"),
                "--control-port",
                str(lane["control_port"]),
                "--control-cookie",
                lane["control_cookie"],
                "--socks-host",
                lane.get("socks_host", "127.0.0.1"),
                "--socks-port",
                str(lane["socks_port"]),
                "--capture-seconds",
                str(args.capture_seconds),
                "--tail-seconds",
                str(args.tail_seconds),
                "--defense",
                record.get("defense", "null (undefended)"),
                "--defense-parameters-json",
                json.dumps(record.get("defense_parameters", {})),
            ]
            for url in record["urls"]:
                command.extend(["--url", url])
            for delay in record.get("inter_tab_delays_seconds", []):
                command.extend(["--inter-tab-delay-seconds", str(delay)])
            if args.skip_tor_check:
                command.append("--skip-tor-check")
            if args.skip_newnym:
                command.append("--skip-newnym")

            error = None
            output_dir = args.output_root / f"{len(record['urls'])}tab" / trace_id
            for attempt in range(args.retries + 1):
                result = subprocess.run(command, text=True, capture_output=True)
                if result.returncode == 0:
                    metadata_path = output_dir / "metadata.json"
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata["manifest"] = record
                    metadata_path.write_text(
                        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    (output_dir / "runner.log").write_text(
                        result.stdout + result.stderr, encoding="utf-8"
                    )
                    error = None
                    break
                error = result.stderr[-4000:]
                if output_dir.exists():
                    archived = output_dir.with_name(
                        f"{trace_id}.failed-{int(time.time())}-{attempt + 1}"
                    )
                    output_dir.rename(archived)

            with lock:
                if error is None:
                    completed.append(trace_id)
                else:
                    failed.append(
                        {"trace_id": trace_id, "lane": lane["name"], "error": error}
                    )
                print(
                    json.dumps(
                        {
                            "complete": len(completed),
                            "failed": len(failed),
                            "remaining": work.qsize(),
                        }
                    ),
                    flush=True,
                )
            work.task_done()

    threads = [threading.Thread(target=collect_lane, args=(lane,)) for lane in lanes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    summary = {"requested": len(records), "completed": len(completed), "failed": failed}
    (args.output_root / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
