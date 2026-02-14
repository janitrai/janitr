#!/usr/bin/env python3
"""
Check data split leakage via ID overlap.

By default checks pairwise overlap among train/valid/calib/holdout JSONL files.
Optionally checks holdout overlap against one or more previous split files
(JSONL with {"id": ...} rows or plain-text ID lists).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"

DEFAULT_TRAIN = DATA_DIR / "train.jsonl"
DEFAULT_VALID = DATA_DIR / "valid.jsonl"
DEFAULT_CALIB = DATA_DIR / "calib.jsonl"
DEFAULT_HOLDOUT = DATA_DIR / "holdout.jsonl"


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        handle.seek(0)

        is_jsonl = first.lstrip().startswith("{")

        if is_jsonl:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                value = obj.get("id")
                if value is None:
                    continue
                ids.add(str(value))
        else:
            for line in handle:
                value = line.strip()
                if value:
                    ids.add(value)

    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument(
        "--forbid-holdout-overlap-with",
        type=Path,
        action="append",
        default=[],
        help="Previous split file(s); any overlap with holdout fails.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    args = parser.parse_args()

    split_paths = {
        "train": args.train,
        "valid": args.valid,
        "calib": args.calib,
        "holdout": args.holdout,
    }

    split_ids = {name: load_ids(path) for name, path in split_paths.items()}

    pairs = [
        ("train", "valid"),
        ("train", "calib"),
        ("train", "holdout"),
        ("valid", "calib"),
        ("valid", "holdout"),
        ("calib", "holdout"),
    ]

    overlap_counts: dict[str, int] = {}
    has_error = False

    for left, right in pairs:
        key = f"{left}_{right}"
        count = len(split_ids[left] & split_ids[right])
        overlap_counts[key] = count
        if count > 0:
            has_error = True

    forbidden_reports: list[dict[str, object]] = []
    holdout_ids = split_ids["holdout"]
    for path in args.forbid_holdout_overlap_with:
        ids = load_ids(path)
        overlap = holdout_ids & ids
        report = {
            "path": str(path),
            "rows": len(ids),
            "holdout_overlap": len(overlap),
        }
        forbidden_reports.append(report)
        if overlap:
            has_error = True

    print("Split row counts:")
    for name in ("train", "valid", "calib", "holdout"):
        print(f"  {name:7s}: {len(split_ids[name])}")

    print("\nPairwise overlaps:")
    for key in overlap_counts:
        print(f"  {key:16s}: {overlap_counts[key]}")

    if forbidden_reports:
        print("\nHoldout overlap checks:")
        for item in forbidden_reports:
            print(
                f"  {item['path']}: overlap={item['holdout_overlap']} rows={item['rows']}"
            )

    report = {
        "split_counts": {name: len(ids) for name, ids in split_ids.items()},
        "pairwise_overlaps": overlap_counts,
        "holdout_forbidden_checks": forbidden_reports,
        "ok": not has_error,
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nWrote report to {args.out}")

    if has_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
