#!/usr/bin/env python3
"""Download a large HTTP file with resumable parallel range requests."""

import argparse
import concurrent.futures
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(url: str, start: int | None = None, end: int | None = None):
    headers = {"User-Agent": "PrismWF-artifact/1.0"}
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end}"
    return urllib.request.Request(url, headers=headers)


def content_length(url: str) -> int:
    head = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "PrismWF-artifact/1.0"}
    )
    with urllib.request.urlopen(head, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download_part(
    url: str,
    part_path: Path,
    start: int,
    end: int,
    retries: int,
) -> None:
    expected = end - start + 1
    existing = part_path.stat().st_size if part_path.exists() else 0
    if existing == expected:
        return
    if existing > expected:
        part_path.unlink()
        existing = 0

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(
                request(url, start + existing, end), timeout=120
            ) as response:
                if response.status != 206:
                    raise RuntimeError(
                        f"Server did not honor range request: HTTP {response.status}"
                    )
                with part_path.open("ab") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            existing = part_path.stat().st_size
            if existing == expected:
                return
            if existing > expected:
                raise RuntimeError(f"Range exceeded expected size: {part_path}")
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed range {start}-{end}: {exc}") from exc
            time.sleep(min(5 * attempt, 30))
            existing = part_path.stat().st_size if part_path.exists() else 0
    raise RuntimeError(
        f"Incomplete range {start}-{end}: downloaded {existing}/{expected} bytes"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--retries", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.connections < 1:
        raise ValueError("connections must be positive")

    total = content_length(args.url)
    if args.output.exists() and args.output.stat().st_size == total:
        print(f"Already complete: {args.output} ({total} bytes)")
        return

    parts_dir = args.output.with_name(f".{args.output.name}.parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunk = (total + args.connections - 1) // args.connections
    jobs = []
    for index in range(args.connections):
        start = index * chunk
        if start >= total:
            break
        end = min(start + chunk - 1, total - 1)
        jobs.append((index, start, end))

    print(
        f"Downloading {total} bytes with {len(jobs)} range connections to {args.output}",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(
                download_part,
                args.url,
                parts_dir / f"part-{index:03d}",
                start,
                end,
                args.retries,
            ): index
            for index, start, end in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()
            print(f"Completed range {futures[future] + 1}/{len(jobs)}", flush=True)

    temporary = args.output.with_suffix(args.output.suffix + ".assembling")
    with temporary.open("wb") as output:
        for index, _, _ in jobs:
            with (parts_dir / f"part-{index:03d}").open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if temporary.stat().st_size != total:
        raise RuntimeError(f"Assembled size mismatch: {temporary.stat().st_size} != {total}")
    temporary.replace(args.output)
    shutil.rmtree(parts_dir)
    print(f"Completed: {args.output} ({total} bytes)", flush=True)


if __name__ == "__main__":
    main()
