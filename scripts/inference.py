#!/usr/bin/env python3
"""
Production inference for scam detection.

Uses the trained fastText model with the optimized threshold (0.90)
that achieves ~4.7% FPR with 75.8% recall.

Usage:
    python scripts/inference.py "check this text for scams"
    echo "some tweet" | python scripts/inference.py --stdin
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "scam_detector.bin"

# Optimized threshold: 4.7% FPR, 75.8% recall on validation set
# Lower threshold = more aggressive (catches more scams, more false positives)
# Higher threshold = more conservative (fewer false positives, misses some scams)
PRODUCTION_THRESHOLD = 0.90


def load_model(model_path: Path):
    """Load the fastText model."""
    try:
        import fasttext  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "fasttext is not installed. Install with: pip install fasttext-wheel"
        ) from exc
    
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    
    return fasttext.load_model(str(model_path))


def predict(model, text: str, threshold: float = PRODUCTION_THRESHOLD) -> dict:
    """
    Predict if text is a scam.
    
    Returns:
        dict with keys:
            - is_scam: bool (True if scam probability >= threshold)
            - probability: float (scam probability 0-1)
            - confidence: str ("high", "medium", "low")
            - label: str ("scam" or "clean")
    """
    labels, probs = model.predict(text.replace("\n", " "), k=3)
    
    scores = {}
    for label, prob in zip(labels, probs):
        cls = label.replace("__label__", "", 1)
        scores[cls] = float(prob)
    
    p_scam = scores.get("scam", 0.0)
    is_scam = p_scam >= threshold
    
    # Confidence bands
    if p_scam >= 0.95 or p_scam <= 0.05:
        confidence = "high"
    elif p_scam >= 0.80 or p_scam <= 0.20:
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        "is_scam": is_scam,
        "probability": p_scam,
        "confidence": confidence,
        "label": "scam" if is_scam else "clean",
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect scams in text using fastText model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/inference.py "🚀 FREE AIRDROP! Connect wallet now!"
    python scripts/inference.py --threshold 0.50 "suspicious text"
    echo "check this" | python scripts/inference.py --stdin
    python scripts/inference.py --json "text to check"
        """,
    )
    parser.add_argument("text", nargs="?", help="Text to classify")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=PRODUCTION_THRESHOLD,
        help=f"Scam threshold (default: {PRODUCTION_THRESHOLD})",
    )
    parser.add_argument("--stdin", action="store_true", help="Read text from stdin")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--batch", action="store_true", help="Process multiple lines from stdin")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.stdin or args.batch:
        texts = [line.strip() for line in sys.stdin if line.strip()]
    elif args.text:
        texts = [args.text]
    else:
        parser.print_help()
        raise SystemExit(1)

    results = []
    for text in texts:
        result = predict(model, text, args.threshold)
        result["text"] = text[:100] + "..." if len(text) > 100 else text
        results.append(result)

    if args.json:
        import json
        if len(results) == 1:
            print(json.dumps(results[0], indent=2))
        else:
            print(json.dumps(results, indent=2))
    else:
        for r in results:
            emoji = "🚨" if r["is_scam"] else "✅"
            print(f"{emoji} {r['label'].upper()} (p={r['probability']:.3f}, {r['confidence']} confidence)")
            if len(results) > 1:
                print(f"   {r['text']}")


if __name__ == "__main__":
    main()
