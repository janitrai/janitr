#!/usr/bin/env python3
"""
Train a fastText supervised model for multi-label scam/topic_crypto classification.
"""

import argparse
import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_TRAIN = REPO_ROOT / "data" / "train.txt"
DEFAULT_MODEL = REPO_ROOT / "models" / "scam_detector.bin"


def filter_training_file(path: Path, exclude_labels: set[str]) -> tuple[Path, int]:
    removed = 0
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        encoding="utf-8",
        prefix="train_filtered_",
        suffix=".txt",
    ) as temp:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or not line.startswith("__label__"):
                    continue
                parts = line.split()
                labels: list[str] = []
                idx = 0
                for token in parts:
                    if token.startswith("__label__"):
                        labels.append(token.replace("__label__", "", 1))
                        idx += 1
                    else:
                        break
                if not labels:
                    continue
                filtered = [label for label in labels if label not in exclude_labels]
                if not filtered:
                    removed += 1
                    continue
                if len(filtered) != len(labels):
                    removed += 1
                text = " ".join(parts[idx:]) if idx < len(parts) else ""
                prefix = " ".join(f"__label__{label}" for label in filtered)
                temp.write(f"{prefix} {text}\n")
    return Path(temp.name), removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fastText model")
    parser.add_argument(
        "--train", type=Path, default=DEFAULT_TRAIN, help="Training data file"
    )
    parser.add_argument(
        "--model-out", type=Path, default=DEFAULT_MODEL, help="Output model file"
    )
    parser.add_argument("--word-ngrams", type=int, default=2, help="Word n-grams")
    parser.add_argument(
        "--minn", type=int, default=2, help="Min length of char n-grams (0 disables)"
    )
    parser.add_argument(
        "--maxn", type=int, default=5, help="Max length of char n-grams (0 disables)"
    )
    parser.add_argument("--bucket", type=int, default=2000000, help="Number of buckets")
    parser.add_argument("--dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--epoch", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    parser.add_argument(
        "--loss",
        type=str,
        default="ova",
        choices=["softmax", "ova", "hs"],
        help="Loss function (use 'ova' for multi-label)",
    )
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=["ai_generated_reply"],
        help="Label to exclude from training data (repeatable)",
    )
    parser.add_argument(
        "--no-exclude",
        action="store_true",
        help="Disable label exclusion filtering",
    )
    args = parser.parse_args()

    if not args.train.exists():
        raise SystemExit(f"Training file not found: {args.train}")

    try:
        import fasttext  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "fasttext is not installed. Install Python deps with: cd scripts && uv sync"
        ) from exc

    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    exclude_labels = set()
    if not args.no_exclude:
        exclude_labels = {label for label in (args.exclude_label or []) if label}

    filtered_path: Path = args.train
    temp_path: Path | None = None
    if exclude_labels:
        filtered_path, removed = filter_training_file(args.train, exclude_labels)
        temp_path = filtered_path
        if removed:
            print(
                f"Filtered {removed} training rows containing excluded labels "
                f"({', '.join(sorted(exclude_labels))})."
            )

    try:
        model = fasttext.train_supervised(
            input=str(filtered_path),
            wordNgrams=args.word_ngrams,
            minn=args.minn,
            maxn=args.maxn,
            bucket=args.bucket,
            dim=args.dim,
            epoch=args.epoch,
            lr=args.lr,
            loss=args.loss,
        )
    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    model.save_model(str(args.model_out))
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
