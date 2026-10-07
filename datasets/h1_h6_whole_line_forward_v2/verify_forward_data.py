#!/usr/bin/env python3
"""Verify the six published whole-line forward arrays and geometry files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "forward_manifest.json").read_text())
    failures = []
    for record in manifest["lines"]:
        line = record["line"]
        for relative, expected in record["files"].items():
            path = ROOT / relative
            if not path.is_file():
                failures.append(f"missing: {path}")
                continue
            actual = sha256(path)
            if actual != expected["sha256"]:
                failures.append(f"sha256: {path}: {actual}")
        path = ROOT / record["forward_path"]
        if path.is_file():
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != record["shape"]:
                failures.append(f"shape: {path}: {list(array.shape)}")
            if str(array.dtype) != record["dtype"]:
                failures.append(f"dtype: {path}: {array.dtype}")
        print(f"VERIFIED {line}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"ALL_VERIFIED {len(manifest['lines'])}")


if __name__ == "__main__":
    main()
