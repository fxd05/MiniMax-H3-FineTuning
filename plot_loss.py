"""Plot loss values emitted by train.py."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    pattern = re.compile(r"step=(\d+)\s+loss=([0-9.eE+-]+)")
    points = []
    for line in args.log.read_text().splitlines():
        match = pattern.search(line)
        if match:
            points.append((int(match.group(1)), float(match.group(2))))
    if not points:
        raise RuntimeError(f"No loss records found in {args.log}")
    steps, losses = zip(*points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker="o", linewidth=1.8)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("MiniMax-H3 training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"saved {args.output} ({len(points)} points)")


if __name__ == "__main__":
    main()
