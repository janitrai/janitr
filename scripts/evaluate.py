#!/usr/bin/env python3
"""
Evaluate a fastText model and print metrics.
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "scam_detector.bin"
DEFAULT_VALID = REPO_ROOT / "data" / "valid.txt"

CLASSES = ["clean", "scam"]


def parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line:
        return None
    if not line.startswith("__label__"):
        return None
    parts = line.split(" ", 1)
    label_token = parts[0]
    text = parts[1] if len(parts) > 1 else ""
    label = label_token.replace("__label__", "", 1)
    return label, text


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fastText model")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model file")
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID, help="Validation file")
    parser.add_argument("--fp-limit", type=int, default=200, help="Max false positives to print")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}")
    if not args.valid.exists():
        raise SystemExit(f"Validation file not found: {args.valid}")

    try:
        import fasttext  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "fasttext is not installed. Install with: pip install fasttext-wheel"
        ) from exc

    model = fasttext.load_model(str(args.model))

    confusion = {actual: {pred: 0 for pred in CLASSES} for actual in CLASSES}
    false_positives: list[str] = []
    total = 0

    with open(args.valid, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            actual, text = parsed
            if actual not in CLASSES:
                continue
            labels, _ = model.predict(text, k=1)
            pred = labels[0].replace("__label__", "", 1)
            if pred not in CLASSES:
                continue

            confusion[actual][pred] += 1
            total += 1

            if actual == "clean" and pred == "scam":
                false_positives.append(text)

    if total == 0:
        raise SystemExit("No valid rows found in validation file.")

    print(f"Evaluated {total} samples")
    print("\nConfusion matrix (rows=actual, cols=predicted)")
    header = f"{'':12s} {'pred_clean':>10s} {'pred_scam':>10s}"
    print(header)
    for actual in CLASSES:
        row = confusion[actual]
        print(f"{('actual_' + actual):12s} {row['clean']:10d} {row['scam']:10d}")

    print("\nPer-class metrics:")
    for cls in CLASSES:
        tp = confusion[cls][cls]
        fp = sum(confusion[a][cls] for a in CLASSES if a != cls)
        fn = sum(confusion[cls][p] for p in CLASSES if p != cls)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        print(
            f"  {cls:5s} | precision {precision:.3f} | recall {recall:.3f} | f1 {f1:.3f}"
        )

    print(f"\nFalse positives (clean -> scam): {len(false_positives)}")
    if false_positives:
        limit = max(args.fp_limit, 0)
        for idx, text in enumerate(false_positives[:limit], start=1):
            print(f"  {idx:03d}. {text}")
        if len(false_positives) > limit:
            print(f"  ... {len(false_positives) - limit} more not shown")


if __name__ == "__main__":
    main()
