#!/usr/bin/env python3
"""
Train a fastText supervised model for scam detection.
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_TRAIN = REPO_ROOT / "data" / "train.txt"
DEFAULT_MODEL = REPO_ROOT / "models" / "scam_detector.bin"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fastText model")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="Training data file")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL, help="Output model file")
    parser.add_argument("--word-ngrams", type=int, default=2, help="Word n-grams")
    parser.add_argument("--minn", type=int, default=0, help="Min length of char n-grams (0 disables)")
    parser.add_argument("--maxn", type=int, default=0, help="Max length of char n-grams (0 disables)")
    parser.add_argument("--bucket", type=int, default=2000000, help="Number of buckets")
    parser.add_argument("--dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--epoch", type=int, default=25, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    args = parser.parse_args()

    if not args.train.exists():
        raise SystemExit(f"Training file not found: {args.train}")

    try:
        import fasttext  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "fasttext is not installed. Install with: pip install fasttext-wheel"
        ) from exc

    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    model = fasttext.train_supervised(
        input=str(args.train),
        wordNgrams=args.word_ngrams,
        minn=args.minn,
        maxn=args.maxn,
        bucket=args.bucket,
        dim=args.dim,
        epoch=args.epoch,
        lr=args.lr,
    )

    model.save_model(str(args.model_out))
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
