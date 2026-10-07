#!/usr/bin/env python3
"""Reassemble the GitHub-compatible checkpoint parts without data loss."""
from pathlib import Path


directory = Path(__file__).resolve().parent / "checkpoints_16tiles"
target = directory / "tiles16_batch4_lr0.0005_tikh0.6_best.pt"
parts = sorted(directory.glob(target.name + ".part-*"))
if not parts:
    raise SystemExit(f"No checkpoint parts found for {target.name}")

with target.open("wb") as output:
    for part in parts:
        output.write(part.read_bytes())

print(f"Reassembled {target} ({target.stat().st_size} bytes)")
