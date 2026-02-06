#!/usr/bin/env python3
"""
Production inference for multi-label scam/topic_crypto classification.
"""

import argparse
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "scam_detector.bin"

DEFAULT_THRESHOLDS = REPO_ROOT / "config" / "thresholds.json"
DEFAULT_GLOBAL_THRESHOLD = 0.5

CLASSES = ["clean", "topic_crypto", "scam", "promo"]


def load_model(model_path: Path):
    """Load the fastText model."""
    try:
        import fasttext  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "fasttext is not installed. Install Python deps with: cd scripts && uv sync"
        ) from exc

    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    return fasttext.load_model(str(model_path))


def load_thresholds(path: Path | None) -> dict[str, float] | None:
    if path is None or not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("thresholds") if isinstance(payload, dict) else None
    if raw is None and isinstance(payload, dict):
        raw = payload
    if not isinstance(raw, dict):
        return None
    thresholds: dict[str, float] = {}
    for cls in CLASSES:
        if cls in raw:
            thresholds[cls] = float(raw[cls])
    return thresholds or None


def build_thresholds(
    per_label: dict[str, float] | None, global_threshold: float
) -> dict[str, float]:
    thresholds = {cls: global_threshold for cls in CLASSES}
    if per_label:
        for cls, value in per_label.items():
            if cls in thresholds:
                thresholds[cls] = float(value)
    return thresholds


def predict(
    model,
    text: str,
    *,
    thresholds: dict[str, float] | None = None,
    threshold: float = DEFAULT_GLOBAL_THRESHOLD,
    allow_empty: bool = False,
) -> dict:
    labels, probs = model.predict(text.replace("\n", " "), k=len(CLASSES))

    scores = {cls: 0.0 for cls in CLASSES}
    for label, prob in zip(labels, probs):
        cls = label.replace("__label__", "", 1)
        if cls in scores:
            scores[cls] = float(prob)

    applied_thresholds = build_thresholds(thresholds, threshold)
    predicted = {
        cls for cls in CLASSES if scores.get(cls, 0.0) >= applied_thresholds[cls]
    }
    if not predicted and not allow_empty:
        predicted = {"clean"} if "clean" in CLASSES else {max(scores, key=scores.get)}
    if "clean" in predicted and any(cls != "clean" for cls in predicted):
        predicted.discard("clean")

    ordered_labels = [cls for cls in CLASSES if cls in predicted]
    p_scam = scores.get("scam", 0.0)
    is_scam = "scam" in predicted

    # Confidence bands (scam-specific)
    if p_scam >= 0.95 or p_scam <= 0.05:
        confidence = "high"
    elif p_scam >= 0.80 or p_scam <= 0.20:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "labels": ordered_labels,
        "is_scam": is_scam,
        "probability": p_scam,
        "confidence": confidence,
        "label": "scam" if is_scam else "clean",
        "scores": scores,
        "thresholds": applied_thresholds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict labels for text using fastText model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/inference.py "🚀 FREE AIRDROP! Connect wallet now!"
    python scripts/inference.py --threshold 0.50 "suspicious text"
    python scripts/inference.py --thresholds config/thresholds.json "text to check"
    echo "check this" | python scripts/inference.py --stdin
    python scripts/inference.py --json "text to check"
        """,
    )
    parser.add_argument("text", nargs="?", help="Text to classify")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Global threshold for all labels (overrides thresholds file)",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=DEFAULT_THRESHOLDS,
        help="JSON file with per-label thresholds",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow empty predictions when nothing meets threshold",
    )
    parser.add_argument("--stdin", action="store_true", help="Read text from stdin")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--batch", action="store_true", help="Process multiple lines from stdin"
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.stdin or args.batch:
        texts = [line.strip() for line in sys.stdin if line.strip()]
    elif args.text:
        texts = [args.text]
    else:
        parser.print_help()
        raise SystemExit(1)

    per_label = None
    global_threshold = DEFAULT_GLOBAL_THRESHOLD
    if args.threshold is not None:
        global_threshold = args.threshold
    else:
        per_label = load_thresholds(args.thresholds)

    results = []
    for text in texts:
        result = predict(
            model,
            text,
            thresholds=per_label,
            threshold=global_threshold,
            allow_empty=args.allow_empty,
        )
        result["text"] = text[:100] + "..." if len(text) > 100 else text
        results.append(result)

    if args.json:
        if len(results) == 1:
            print(json.dumps(results[0], indent=2))
        else:
            print(json.dumps(results, indent=2))
    else:
        for r in results:
            emoji = "🚨" if r["is_scam"] else "✅"
            labels = ",".join(r["labels"]) if r["labels"] else "none"
            print(
                f"{emoji} {labels.upper()} (p_scam={r['probability']:.3f}, {r['confidence']} confidence)"
            )
            if len(results) > 1:
                print(f"   {r['text']}")


if __name__ == "__main__":
    main()
