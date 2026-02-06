#!/usr/bin/env python3
"""
Prepare dataset for Hugging Face upload.

Converts data/sample.jsonl to dataset/train.jsonl + dataset/test.jsonl
with proper formatting and train/test split.

Usage:
    python scripts/prepare_hf.py [--test-ratio 0.2] [--seed 42]
"""

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
INPUT_FILE = REPO_ROOT / "data" / "sample.jsonl"
OUTPUT_DIR = REPO_ROOT / "dataset"

LABEL_TO_ID = {
    "clean": 0,
    "topic_crypto": 1,
    "scam": 2,
    "ai_generated_reply": 3,
    "promo": 4,
}


def load_samples(path: Path) -> list[dict]:
    """Load JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def convert_to_hf_format(sample: dict) -> dict:
    """Convert internal format to HF format."""
    labels = sample.get("labels")
    if labels is None and "label" in sample:
        labels = [sample["label"]]
    labels = labels or []
    seen = set()
    deduped = []
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    label_ids = [LABEL_TO_ID[label] for label in deduped if label in LABEL_TO_ID]
    return {
        "id": sample["id"],
        "text": sample["text"],
        "labels": deduped,
        "label_ids": label_ids,
        "platform": sample.get("platform", "x"),
        "source_id": sample.get("source_id", ""),
        "source_url": sample.get("source_url", ""),
        "collected_at": sample.get("collected_at", ""),
    }


def write_jsonl(samples: list[dict], path: Path):
    """Write samples to JSONL file."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare HF dataset")
    parser.add_argument(
        "--test-ratio", type=float, default=0.2, help="Test split ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load samples
    samples = load_samples(INPUT_FILE)
    print(f"Loaded {len(samples)} samples from {INPUT_FILE}")

    # Convert to HF format
    hf_samples = [convert_to_hf_format(s) for s in samples]

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(hf_samples)

    split_idx = int(len(hf_samples) * (1 - args.test_ratio))
    train_samples = hf_samples[:split_idx]
    test_samples = hf_samples[split_idx:]

    # Ensure we have at least 1 in each split
    if len(test_samples) == 0 and len(train_samples) > 1:
        test_samples = [train_samples.pop()]
    if len(train_samples) == 0 and len(test_samples) > 1:
        train_samples = [test_samples.pop()]

    # Write output
    OUTPUT_DIR.mkdir(exist_ok=True)
    write_jsonl(train_samples, OUTPUT_DIR / "train.jsonl")
    write_jsonl(test_samples, OUTPUT_DIR / "test.jsonl")

    # Print stats
    print(f"\nOutput written to {OUTPUT_DIR}/")
    print(f"  train.jsonl: {len(train_samples)} samples")
    print(f"  test.jsonl:  {len(test_samples)} samples")

    # Label distribution
    print("\nLabel distribution:")
    for split_name, split_samples in [("train", train_samples), ("test", test_samples)]:
        counts = {}
        for s in split_samples:
            for label in s.get("labels", []):
                counts[label] = counts.get(label, 0) + 1
        print(f"  {split_name}: {counts}")


if __name__ == "__main__":
    main()
