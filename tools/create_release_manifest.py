#!/usr/bin/env python3
"""Create checksums and an environment snapshot for an artifact release."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import sklearn
import timm
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def tracked_files(root: Path) -> list[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"],
        stderr=subprocess.DEVNULL,
    )
    return [root / item.decode("utf-8") for item in output.split(b"\0") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("release_manifest.json"))
    parser.add_argument("--include-large-files", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output_path = args.output.resolve()
    files = []
    for path in tracked_files(root):
        if not path.is_file():
            continue
        if path.resolve() == output_path:
            continue
        if not args.include_large_files and path.stat().st_size > 100 * 1024 * 1024:
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    manifest = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "timm": timm.__version__,
            "nvidia_driver": command_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ),
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        },
        "files": files,
    }
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} with {len(files)} checksums")


if __name__ == "__main__":
    main()
