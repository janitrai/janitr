#!/usr/bin/env python3
"""
Prepare data for fastText training.

Converts data/sample.jsonl to data/train.txt and data/valid.txt
with fastText labels:
  __label__scam <text>
  __label__clean <text>
"""

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "sample.jsonl"
DEFAULT_TRAIN = REPO_ROOT / "data" / "train.txt"
DEFAULT_VALID = REPO_ROOT / "data" / "valid.txt"

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str, *, normalize: bool, lowercase: bool, strip_urls: bool) -> str:
    if text is None:
        return ""
    if normalize:
        text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH_RE.sub("", text)
    if strip_urls:
        text = URL_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def map_label(label: str) -> str | None:
    if label == "clean":
        return "clean"
    if label in {"crypto_scam", "scam"}:
        return "scam"
    return None


def load_samples(path: Path) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_fasttext(path: Path, rows: list[tuple[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for label, text in rows:
            f.write(f"__label__{label} {text}\n")


def count_labels(rows: list[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label, _ in rows:
        counts[label] = counts.get(label, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fastText data")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL file")
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN, help="Train output file")
    parser.add_argument("--valid-out", type=Path, default=DEFAULT_VALID, help="Validation output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--valid-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--strip-urls", action="store_true", help="Remove URLs from text")
    parser.add_argument("--no-normalize", action="store_true", help="Disable Unicode normalization")
    parser.add_argument("--no-lowercase", action="store_true", help="Disable lowercasing")
    args = parser.parse_args()

    samples = load_samples(args.input)
    print(f"Loaded {len(samples)} samples from {args.input}")

    rows: list[tuple[str, str]] = []
    skipped_unknown = 0
    skipped_empty = 0

    for sample in samples:
        label = map_label(sample.get("label", ""))
        if label is None:
            skipped_unknown += 1
            continue

        text = sample.get("text") or sample.get("raw_text") or ""
        cleaned = clean_text(
            text,
            normalize=not args.no_normalize,
            lowercase=not args.no_lowercase,
            strip_urls=args.strip_urls,
        )

        if not cleaned:
            skipped_empty += 1
            continue

        rows.append((label, cleaned))

    if not rows:
        raise SystemExit("No valid rows found after cleaning.")

    random.Random(args.seed).shuffle(rows)

    split_idx = int(len(rows) * (1 - args.valid_ratio))
    train_rows = rows[:split_idx]
    valid_rows = rows[split_idx:]

    if len(valid_rows) == 0 and len(train_rows) > 1:
        valid_rows = [train_rows.pop()]
    if len(train_rows) == 0 and len(valid_rows) > 1:
        train_rows = [valid_rows.pop()]

    write_fasttext(args.train_out, train_rows)
    write_fasttext(args.valid_out, valid_rows)

    print(f"\nWrote {len(train_rows)} train rows to {args.train_out}")
    print(f"Wrote {len(valid_rows)} valid rows to {args.valid_out}")

    print("\nLabel distribution:")
    print(f"  train: {count_labels(train_rows)}")
    print(f"  valid: {count_labels(valid_rows)}")

    if skipped_unknown or skipped_empty:
        print("\nSkipped rows:")
        print(f"  unknown label: {skipped_unknown}")
        print(f"  empty after cleaning: {skipped_empty}")


if __name__ == "__main__":
    main()
