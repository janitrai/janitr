#!/usr/bin/env python3
"""
Create a deterministic time-based holdout split from data/sample.jsonl.

Outputs:
  - data/holdout.jsonl (original records)
  - data/holdout.txt (fastText format)
  - data/holdout_meta.json (split metadata)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "sample.jsonl"
DEFAULT_HOLDOUT_JSONL = REPO_ROOT / "data" / "holdout.jsonl"
DEFAULT_HOLDOUT_TXT = REPO_ROOT / "data" / "holdout.txt"
DEFAULT_META = REPO_ROOT / "data" / "holdout_meta.json"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from prepare_data import clean_text, extract_labels  # type: ignore


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        text = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def write_fasttext(path: Path, rows: list[tuple[list[str], str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for labels, text in rows:
            if not labels:
                continue
            prefix = " ".join(f"__label__{label}" for label in labels)
            f.write(f"{prefix} {text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic holdout split")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL")
    parser.add_argument("--ratio", type=float, default=0.1, help="Holdout ratio (time-based)")
    parser.add_argument(
        "--cutoff",
        type=str,
        default="",
        help="ISO timestamp cutoff; rows >= cutoff go to holdout",
    )
    parser.add_argument("--holdout-jsonl", type=Path, default=DEFAULT_HOLDOUT_JSONL)
    parser.add_argument("--holdout-txt", type=Path, default=DEFAULT_HOLDOUT_TXT)
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META)
    parser.add_argument("--strip-urls", action="store_true", help="Remove URLs from text")
    parser.add_argument("--no-normalize", action="store_true", help="Disable Unicode normalization")
    parser.add_argument("--no-lowercase", action="store_true", help="Disable lowercasing")
    args = parser.parse_args()

    if args.ratio <= 0 or args.ratio >= 1:
        raise SystemExit("--ratio must be > 0 and < 1")

    cutoff_dt = parse_time(args.cutoff) if args.cutoff else None

    records: list[tuple[datetime | None, str, dict, list[str], str]] = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            labels = extract_labels(obj)
            if not labels:
                continue
            text = obj.get("text") or obj.get("raw_text") or ""
            cleaned = clean_text(
                text,
                normalize=not args.no_normalize,
                lowercase=not args.no_lowercase,
                strip_urls=args.strip_urls,
            )
            if not cleaned:
                continue
            ts = parse_time(obj.get("collected_at"))
            id_ = str(obj.get("id", ""))
            records.append((ts, id_, obj, labels, cleaned))

    if not records:
        raise SystemExit("No valid records found.")

    min_dt = datetime.min.replace(tzinfo=timezone.utc)
    records.sort(key=lambda row: (row[0] or min_dt, row[1]))

    if cutoff_dt:
        holdout = [row for row in records if row[0] and row[0] >= cutoff_dt]
        train = [row for row in records if row not in holdout]
    else:
        split_idx = int(len(records) * (1 - args.ratio))
        train = records[:split_idx]
        holdout = records[split_idx:]

    if not holdout:
        raise SystemExit("Holdout split is empty. Adjust --ratio or --cutoff.")

    holdout_rows: list[tuple[list[str], str]] = []
    holdout_objs: list[dict] = []
    for _, _, obj, labels, cleaned in holdout:
        clone = dict(obj)
        clone["labels"] = labels
        clone.pop("label", None)
        holdout_objs.append(clone)
        holdout_rows.append((labels, cleaned))

    args.holdout_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.holdout_jsonl.open("w", encoding="utf-8") as f:
        for obj in holdout_objs:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

    write_fasttext(args.holdout_txt, holdout_rows)

    first_holdout = holdout[0][0].isoformat() if holdout[0][0] else None
    last_train = train[-1][0].isoformat() if train and train[-1][0] else None
    meta = {
        "input": str(args.input),
        "holdout_jsonl": str(args.holdout_jsonl),
        "holdout_txt": str(args.holdout_txt),
        "ratio": args.ratio,
        "cutoff": args.cutoff or None,
        "total": len(records),
        "train_count": len(train),
        "holdout_count": len(holdout),
        "last_train_collected_at": last_train,
        "first_holdout_collected_at": first_holdout,
    }
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote holdout JSONL to {args.holdout_jsonl}")
    print(f"Wrote holdout TXT to {args.holdout_txt}")
    print(f"Wrote holdout metadata to {args.meta_out}")
    print(f"Holdout size: {len(holdout)} / {len(records)}")
    if last_train or first_holdout:
        print(f"Last train collected_at: {last_train}")
        print(f"First holdout collected_at: {first_holdout}")


if __name__ == "__main__":
    main()
