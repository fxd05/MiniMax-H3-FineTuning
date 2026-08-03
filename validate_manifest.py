"""Validate the public H3 JSONL manifest format without loading media."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PICTURE = re.compile(r"<Picture (\d+)>")
LIST_FIELDS = ("reference_images", "reference_videos", "reference_audios")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--allow-reference-only", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    seen: set[str] = set()
    count = 0
    for line_number, raw in enumerate(args.manifest.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        count += 1
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
            continue
        sample_id = str(row.get("id", ""))
        if not sample_id:
            errors.append(f"line {line_number}: missing id")
        elif sample_id in seen:
            errors.append(f"line {line_number}: duplicate id {sample_id!r}")
        seen.add(sample_id)
        if not row.get("caption"):
            errors.append(f"line {line_number}: missing caption")
        target = row.get("target_video") or row.get("video")
        reference_only = args.allow_reference_only and row.get("example_type") == "reference_only"
        if not target and not reference_only:
            errors.append(f"line {line_number}: missing target_video")
        paths: list[tuple[str, str]] = [("target_video", target)] if target else []
        if row.get("target_audio"):
            paths.append(("target_audio", row["target_audio"]))
        for field in LIST_FIELDS:
            value = row.get(field, [])
            if not isinstance(value, list):
                errors.append(f"line {line_number}: {field} must be a list")
                continue
            paths.extend((field, path) for path in value if isinstance(path, str))
            if any(not isinstance(path, str) or not path for path in value):
                errors.append(f"line {line_number}: {field} contains an invalid path")
        if args.check_files:
            for field, path in paths:
                if not Path(path).is_file():
                    errors.append(f"line {line_number}: {field} does not exist: {path}")
        max_picture = max((int(i) for i in PICTURE.findall(str(row.get("caption", "")))), default=0)
        if max_picture > len(row.get("reference_images", [])):
            errors.append(f"line {line_number}: caption references <Picture {max_picture}>, but reference_images has only {len(row.get('reference_images', []))}")
    if errors:
        raise SystemExit("Manifest validation failed:\n" + "\n".join(errors[:50]))
    print(f"valid manifest: {count} samples")


if __name__ == "__main__":
    main()
