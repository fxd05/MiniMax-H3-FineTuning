"""Build a deterministic, diverse 1000-sample H3 Ref2VA manifest."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def category(row):
    ni = len(row.get("reference_images") or [])
    na = len(row.get("reference_audios") or [])
    nv = len(row.get("reference_videos") or [])
    return f"v{min(nv, 2)}_i{min(ni, 9)}_a{min(na, 3)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("manifests/train_1000.jsonl"))
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.input.open() if x.strip()]
    groups = defaultdict(list)
    for row in rows:
        if row.get("target_video") or row.get("video"):
            groups[category(row)].append(row)
    rng = random.Random(args.seed)
    for values in groups.values():
        rng.shuffle(values)

    # Keep every available video-reference row, then allocate the remainder
    # round-robin over image/audio cardinality strata.
    selected, seen = [], set()
    for key in sorted(groups):
        if key.startswith("v") and key.split("_")[0] != "v0":
            for row in groups[key]:
                if len(selected) >= 1000:
                    break
                selected.append(row); seen.add(row.get("id") or row.get("sample_id"))
    keys = sorted(groups)
    cursor = 0
    while len(selected) < 1000 and keys:
        key = keys[cursor % len(keys)]
        cursor += 1
        if groups[key]:
            row = groups[key].pop()
            ident = row.get("id") or row.get("sample_id")
            if ident not in seen:
                selected.append(row); seen.add(ident)
        if cursor > len(keys) * 10000:
            break
    if len(selected) != 1000:
        raise RuntimeError(f"Only selected {len(selected)} rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "count": len(selected), "categories": {k: sum(category(r) == k for r in selected) for k in sorted({category(r) for r in selected})}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
