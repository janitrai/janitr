#!/usr/bin/env python3
"""
Tune per-label thresholds to maximize recall under an FPR constraint.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "scam_detector.bin"
DEFAULT_DATA = REPO_ROOT / "data" / "calib.txt"
DEFAULT_OUT = REPO_ROOT / "config" / "thresholds.json"

CLASSES = ["clean", "topic_crypto", "scam", "promo"]

from evaluate import get_scores, parse_line  # type: ignore


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def load_rows(path: Path) -> list[tuple[set[str], str]]:
    rows: list[tuple[set[str], str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            labels, text = parsed
            if labels:
                rows.append((labels, text))
    return rows


def best_threshold_for_label(
    points: list[tuple[float, bool]], target_fpr: float
) -> tuple[float, float, float, float]:
    candidates = sorted({score for score, _ in points})
    candidates.append(1.0)

    best = None
    for thr in candidates:
        tp = fp = fn = tn = 0
        for score, gold in points:
            pred = score >= thr
            if pred and gold:
                tp += 1
            elif pred and not gold:
                fp += 1
            elif (not pred) and gold:
                fn += 1
            else:
                tn += 1
        fpr = safe_div(fp, fp + tn)
        if fpr > target_fpr:
            continue
        recall = safe_div(tp, tp + fn)
        precision = safe_div(tp, tp + fp)
        if best is None:
            best = (thr, fpr, recall, precision)
        else:
            _, _, best_recall, _ = best
            if recall > best_recall or (recall == best_recall and thr > best[0]):
                best = (thr, fpr, recall, precision)

    if best is None:
        best = (1.0, 1.0, 0.0, 0.0)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model file")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Calibration data (fastText txt)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output thresholds JSON"
    )
    parser.add_argument(
        "--target-fpr", type=float, default=0.02, help="Target FPR per label"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="topic_crypto,scam,promo",
        help="Comma-separated labels to tune",
    )
    parser.add_argument("--clean-threshold", type=float, default=0.1)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}")
    if not args.data.exists():
        raise SystemExit(f"Data file not found: {args.data}")

    try:
        import fasttext  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "fasttext is not installed. Install Python deps with: cd scripts && uv sync"
        ) from exc

    rows = load_rows(args.data)
    if not rows:
        raise SystemExit("No valid rows found in calibration data.")

    label_list = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not label_list:
        raise SystemExit("No labels provided.")

    model = fasttext.load_model(str(args.model))

    scored_rows = []
    for labels, text in rows:
        scores = get_scores(model, text)
        scored_rows.append((labels, scores))

    thresholds: dict[str, float] = {label: 1.0 for label in CLASSES}
    thresholds["clean"] = float(args.clean_threshold)

    print(f"Calibrating on {len(rows)} samples (target FPR <= {args.target_fpr:.2%})")
    print(f"{'label':10s} {'thr':>7s} {'fpr':>7s} {'recall':>7s} {'prec':>7s}")

    stats_out: dict[str, dict[str, float]] = {}
    for label in label_list:
        points = [
            (scores.get(label, 0.0), (label in labels))
            for labels, scores in scored_rows
        ]
        thr, fpr, recall, precision = best_threshold_for_label(points, args.target_fpr)
        thresholds[label] = float(thr)
        stats_out[label] = {
            "threshold": float(thr),
            "fpr": float(fpr),
            "recall": float(recall),
            "precision": float(precision),
        }
        print(f"{label:10s} {thr:7.4f} {fpr:7.4f} {recall:7.4f} {precision:7.4f}")

    payload = {
        "version": 1,
        "classes": CLASSES,
        "thresholds": thresholds,
        "tune_target_fpr": args.target_fpr,
        "tuned_on": str(args.data),
        "tuned_at": date.today().isoformat(),
        "label_stats": stats_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\nWrote thresholds to {args.out}")


if __name__ == "__main__":
    main()
