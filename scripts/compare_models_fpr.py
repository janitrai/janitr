#!/usr/bin/env python3
"""
Compare reduced fastText models under an FPR constraint.

For each model and label, find the highest-recall threshold such that
FPR <= target. Prints a per-label table and optionally writes CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_HOLDOUT = REPO_ROOT / "data" / "holdout.txt"
DEFAULT_MODELS_GLOB = "models/reduced/quant-*.ftz"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluate as eval_mod  # type: ignore


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def load_rows(path: Path) -> list[tuple[set[str], str]]:
    rows: list[tuple[set[str], str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = eval_mod.parse_line(line)
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
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS_GLOB)
    parser.add_argument("--target-fpr", type=float, default=0.02)
    parser.add_argument(
        "--labels",
        type=str,
        default="topic_crypto,scam,promo",
        help="Comma-separated labels to evaluate",
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="Optional CSV output path"
    )
    args = parser.parse_args()

    if not args.holdout.exists():
        raise SystemExit(f"Holdout file not found: {args.holdout}")

    try:
        import fasttext  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "fasttext is not installed. Install Python deps with: cd scripts && uv sync"
        ) from exc

    rows = load_rows(args.holdout)
    if not rows:
        raise SystemExit("No valid rows found in holdout file.")

    label_list = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not label_list:
        raise SystemExit("No labels provided.")

    model_paths = sorted(Path(REPO_ROOT / ".").glob(args.models))
    if not model_paths:
        raise SystemExit(f"No models matched: {args.models}")

    results: dict[str, dict[str, dict[str, float]]] = {}

    for model_path in model_paths:
        model = fasttext.load_model(str(model_path))
        size_mb = model_path.stat().st_size / (1024 * 1024)
        per_label: dict[str, dict[str, float]] = {}
        # Precompute scores for all rows
        scored_rows = []
        for labels, text in rows:
            scores = eval_mod.get_scores(model, text)
            scored_rows.append((labels, scores))
        for label in label_list:
            points = [
                (scores.get(label, 0.0), (label in labels))
                for labels, scores in scored_rows
            ]
            thr, fpr, recall, precision = best_threshold_for_label(
                points, args.target_fpr
            )
            per_label[label] = {
                "threshold": float(thr),
                "fpr": float(fpr),
                "recall": float(recall),
                "precision": float(precision),
            }
        per_label["_size_mb"] = {
            "threshold": size_mb,
            "fpr": 0.0,
            "recall": 0.0,
            "precision": 0.0,
        }
        results[model_path.name] = per_label

    for label in label_list:
        print(f"\nLabel: {label} (target FPR <= {args.target_fpr:.2%})")
        print(
            f"{'model':24s} {'sizeMB':>7s} {'thr':>7s} {'fpr':>7s} {'recall':>7s} {'prec':>7s}"
        )
        for model_name, per_label in results.items():
            size_mb = per_label["_size_mb"]["threshold"]
            stats = per_label[label]
            print(
                f"{model_name:24s} "
                f"{size_mb:7.2f} "
                f"{stats['threshold']:7.4f} "
                f"{stats['fpr']:7.4f} "
                f"{stats['recall']:7.4f} "
                f"{stats['precision']:7.4f}"
            )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["model", "size_mb", "label", "threshold", "fpr", "recall", "precision"]
            )
            for model_name, per_label in results.items():
                size_mb = per_label["_size_mb"]["threshold"]
                for label in label_list:
                    stats = per_label[label]
                    writer.writerow(
                        [
                            model_name,
                            f"{size_mb:.4f}",
                            label,
                            f"{stats['threshold']:.6f}",
                            f"{stats['fpr']:.6f}",
                            f"{stats['recall']:.6f}",
                            f"{stats['precision']:.6f}",
                        ]
                    )
        print(f"\nWrote CSV to {args.csv}")


if __name__ == "__main__":
    main()
