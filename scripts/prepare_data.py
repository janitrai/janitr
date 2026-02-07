#!/usr/bin/env python3
"""
Prepare data for fastText training.

Converts data/sample.jsonl to data/train.txt and data/valid.txt
with consolidated fastText labels (single training label per line):
  __label__scam <text>
  __label__topic_crypto <text>
  __label__clean <text>

Raw JSONL labels are preserved; consolidation happens only here during training
data preparation.
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


TRAINING_CLASSES = ["clean", "topic_crypto", "scam"]

# Consolidate a detailed label taxonomy into 3 training classes.
# Priority: scam > topic_crypto > clean.
SCAM_RAW_LABELS = {
    # User-provided scam bucket
    "phishing",
    "malware",
    "fake_support",
    "recovery_scam",
    "job_scam",
    "romance_scam",
    "impersonation",
    "account_compromise",
    "spam",
    "reply_spam",
    "dm_spam",
    "promo",
    "affiliate",
    "lead_gen",
    "engagement_bait",
    "follow_train",
    "giveaway",
    "bot",
    # Common/legacy labels
    "scam",
    "crypto_scam",
}


def _normalize_label(label: str) -> str:
    return label.strip().lower()


def extract_raw_labels(sample: dict) -> list[str]:
    """Extract raw labels from a JSONL sample without consolidation."""
    raw: list[str] = []
    if isinstance(sample.get("labels"), list):
        for item in sample["labels"]:
            if not isinstance(item, str):
                continue
            normalized = _normalize_label(item)
            if normalized:
                raw.append(normalized)
        return raw

    label = sample.get("label")
    if isinstance(label, str):
        normalized = _normalize_label(label)
        if normalized:
            raw.append(normalized)
    return raw


def consolidate_training_label(raw_labels: list[str]) -> str:
    """Map raw labels to a single training class."""
    raw_set = set(raw_labels)
    if raw_set & SCAM_RAW_LABELS:
        return "scam"
    if "topic_crypto" in raw_set or "crypto" in raw_set:
        return "topic_crypto"
    return "clean"


def extract_labels(sample: dict) -> list[str]:
    """Extract consolidated training labels (always one label)."""
    raw_labels = extract_raw_labels(sample)
    return [consolidate_training_label(raw_labels)]


def map_label(label: str | None) -> str:
    """Legacy helper for scripts that expect a single consolidated label."""
    if not label:
        return "clean"
    normalized = _normalize_label(label)
    if not normalized:
        return "clean"
    return consolidate_training_label([normalized])


def load_samples(path: Path) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_fasttext(path: Path, rows: list[tuple[list[str], str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for labels, text in rows:
            if not labels:
                continue
            label_prefix = " ".join(f"__label__{label}" for label in labels)
            f.write(f"{label_prefix} {text}\n")


def count_labels(rows: list[tuple[list[str], str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for labels, _ in rows:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fastText data")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL file"
    )
    parser.add_argument(
        "--train-out", type=Path, default=DEFAULT_TRAIN, help="Train output file"
    )
    parser.add_argument(
        "--valid-out", type=Path, default=DEFAULT_VALID, help="Validation output file"
    )
    parser.add_argument(
        "--hard-negatives",
        type=Path,
        default=None,
        help="JSONL of clean hard negatives to oversample into training",
    )
    parser.add_argument(
        "--hard-negatives-mult",
        type=int,
        default=3,
        help="Times to repeat each hard negative in training",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--valid-ratio", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--strip-urls", action="store_true", help="Remove URLs from text"
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Disable Unicode normalization"
    )
    parser.add_argument(
        "--no-lowercase", action="store_true", help="Disable lowercasing"
    )
    args = parser.parse_args()
    if args.hard_negatives_mult < 1:
        raise SystemExit("--hard-negatives-mult must be >= 1")

    samples = load_samples(args.input)
    print(f"Loaded {len(samples)} samples from {args.input}")

    rows: list[tuple[list[str], str]] = []
    skipped_empty = 0
    mapped_clean_unlabeled = 0
    mapped_clean_other = 0

    for sample in samples:
        raw_labels = extract_raw_labels(sample)
        training_label = consolidate_training_label(raw_labels)
        if not raw_labels:
            mapped_clean_unlabeled += 1
        elif training_label == "clean":
            mapped_clean_other += 1

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

        rows.append(([training_label], cleaned))

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

    hard_added = 0
    hard_skipped_unknown = 0
    hard_skipped_empty = 0

    if args.hard_negatives is not None:
        if not args.hard_negatives.exists():
            raise SystemExit(f"Hard negatives file not found: {args.hard_negatives}")
        hard_samples = load_samples(args.hard_negatives)
        for sample in hard_samples:
            labels = extract_labels(sample)
            if "clean" not in labels or len(labels) != 1:
                hard_skipped_unknown += 1
                continue
            text = sample.get("text") or sample.get("raw_text") or ""
            cleaned = clean_text(
                text,
                normalize=not args.no_normalize,
                lowercase=not args.no_lowercase,
                strip_urls=args.strip_urls,
            )
            if not cleaned:
                hard_skipped_empty += 1
                continue
            for _ in range(args.hard_negatives_mult):
                train_rows.append((["clean"], cleaned))
                hard_added += 1

        random.Random(args.seed).shuffle(train_rows)

    write_fasttext(args.train_out, train_rows)
    write_fasttext(args.valid_out, valid_rows)

    print(f"\nWrote {len(train_rows)} train rows to {args.train_out}")
    print(f"Wrote {len(valid_rows)} valid rows to {args.valid_out}")

    print("\nLabel distribution:")
    print(f"  train: {count_labels(train_rows)}")
    print(f"  valid: {count_labels(valid_rows)}")

    if args.hard_negatives is not None:
        print("\nHard negatives:")
        print(f"  added to train: {hard_added}")
        if hard_skipped_unknown or hard_skipped_empty:
            print("  skipped:")
            print(f"    non-clean label: {hard_skipped_unknown}")
            print(f"    empty after cleaning: {hard_skipped_empty}")

    if mapped_clean_unlabeled or mapped_clean_other:
        print("\nLabel notes:")
        print(f"  unlabeled -> clean: {mapped_clean_unlabeled}")
        print(f"  other labels -> clean: {mapped_clean_other}")

    if skipped_empty:
        print("\nSkipped rows:")
        print(f"  empty after cleaning: {skipped_empty}")


if __name__ == "__main__":
    main()
