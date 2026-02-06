#!/usr/bin/env python3
"""
Evaluate a fastText model with multi-label metrics.
"""

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "scam_detector.bin"
DEFAULT_VALID = REPO_ROOT / "data" / "valid.txt"
DEFAULT_THRESHOLDS = REPO_ROOT / "config" / "thresholds.json"

CLASSES = ["clean", "topic_crypto", "scam", "promo"]
DEFAULT_GLOBAL_THRESHOLD = 0.5


def parse_line(line: str) -> tuple[set[str], str] | None:
    line = line.strip()
    if not line or not line.startswith("__label__"):
        return None
    parts = line.split()
    labels: list[str] = []
    idx = 0
    for token in parts:
        if token.startswith("__label__"):
            labels.append(token.replace("__label__", "", 1))
            idx += 1
        else:
            break
    filtered = {label for label in labels if label in CLASSES}
    if not filtered:
        return None
    text = " ".join(parts[idx:]) if idx < len(parts) else ""
    return filtered, text


def get_scores(model, text: str) -> dict[str, float]:
    labels, probs = model.predict(text, k=len(CLASSES))
    scores: dict[str, float] = {cls: 0.0 for cls in CLASSES}
    for label, prob in zip(labels, probs):
        cls = label.replace("__label__", "", 1)
        if cls in scores:
            scores[cls] = float(prob)
    return scores


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


def predict_labels(
    scores: dict[str, float],
    thresholds: dict[str, float],
    *,
    allow_empty: bool,
) -> set[str]:
    predicted = {cls for cls in CLASSES if scores.get(cls, 0.0) >= thresholds[cls]}
    if not predicted and not allow_empty:
        predicted = {"clean"} if "clean" in CLASSES else {max(scores, key=scores.get)}
    if "clean" in predicted and any(cls != "clean" for cls in predicted):
        predicted.discard("clean")
    return predicted


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def evaluate(
    rows: list[tuple[set[str], str]],
    model,
    thresholds: dict[str, float],
    *,
    allow_empty: bool,
) -> dict:
    tp = Counter()
    fp = Counter()
    fn = Counter()
    support = Counter()
    exact = 0

    for gold, text in rows:
        scores = get_scores(model, text)
        pred = predict_labels(scores, thresholds, allow_empty=allow_empty)
        if pred == gold:
            exact += 1
        for cls in CLASSES:
            if cls in gold:
                support[cls] += 1
            if cls in pred and cls in gold:
                tp[cls] += 1
            elif cls in pred and cls not in gold:
                fp[cls] += 1
            elif cls not in pred and cls in gold:
                fn[cls] += 1

    metrics: dict[str, dict[str, float]] = {}
    total = len(rows)
    for cls in CLASSES:
        precision = safe_div(tp[cls], tp[cls] + fp[cls])
        recall = safe_div(tp[cls], tp[cls] + fn[cls])
        f1 = safe_div(2 * precision * recall, precision + recall)
        negatives = total - support[cls]
        fpr = safe_div(fp[cls], negatives)
        fnr = safe_div(fn[cls], support[cls])
        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "support": float(support[cls]),
        }

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_precision = safe_div(total_tp, total_tp + total_fp)
    micro_recall = safe_div(total_tp, total_tp + total_fn)
    micro_f1 = safe_div(
        2 * micro_precision * micro_recall, micro_precision + micro_recall
    )

    macro_precision = safe_div(
        sum(m["precision"] for m in metrics.values()), len(CLASSES)
    )
    macro_recall = safe_div(sum(m["recall"] for m in metrics.values()), len(CLASSES))
    macro_f1 = safe_div(sum(m["f1"] for m in metrics.values()), len(CLASSES))

    return {
        "metrics": metrics,
        "exact_match": safe_div(exact, len(rows)),
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
    }


def tune_thresholds(
    rows: list[tuple[set[str], str]],
    model,
    *,
    step: float,
) -> dict[str, float]:
    if step <= 0 or step >= 1:
        raise ValueError("tune step must be > 0 and < 1")
    grid: list[float] = []
    thr = step
    while thr < 1.0:
        grid.append(round(thr, 6))
        thr += step

    scores_by_label: dict[str, list[float]] = {cls: [] for cls in CLASSES}
    gold_by_label: dict[str, list[int]] = {cls: [] for cls in CLASSES}

    for gold, text in rows:
        scores = get_scores(model, text)
        for cls in CLASSES:
            scores_by_label[cls].append(scores.get(cls, 0.0))
            gold_by_label[cls].append(1 if cls in gold else 0)

    best: dict[str, float] = {}
    for cls in CLASSES:
        best_f1 = -1.0
        best_thr = DEFAULT_GLOBAL_THRESHOLD
        y_true = gold_by_label[cls]
        y_score = scores_by_label[cls]
        for thr in grid:
            tp = fp = fn = 0
            for score, gold in zip(y_score, y_true):
                pred = score >= thr
                if pred and gold:
                    tp += 1
                elif pred and not gold:
                    fp += 1
                elif (not pred) and gold:
                    fn += 1
            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            f1 = safe_div(2 * precision * recall, precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        best[cls] = best_thr
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fastText model (multi-label)"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model file")
    parser.add_argument(
        "--valid", type=Path, default=DEFAULT_VALID, help="Validation file"
    )
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
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune per-label thresholds to maximize F1 on validation set",
    )
    parser.add_argument(
        "--tune-step",
        type=float,
        default=0.05,
        help="Step size for threshold tuning grid",
    )
    parser.add_argument(
        "--save-thresholds",
        type=Path,
        default=None,
        help="Write tuned thresholds JSON to this path",
    )
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

    rows: list[tuple[set[str], str]] = []
    with open(args.valid, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            labels, text = parsed
            if labels:
                rows.append((labels, text))

    if not rows:
        raise SystemExit("No valid rows found in validation file.")

    model = fasttext.load_model(str(args.model))

    per_label = None
    global_threshold = DEFAULT_GLOBAL_THRESHOLD

    if args.tune:
        per_label = tune_thresholds(rows, model, step=args.tune_step)
        if args.save_thresholds:
            try:
                tuned_on = str(args.valid.relative_to(REPO_ROOT))
            except ValueError:
                tuned_on = str(args.valid)
            payload = {
                "version": 1,
                "classes": CLASSES,
                "thresholds": per_label,
                "tuned_on": tuned_on,
                "tuned_at": date.today().isoformat(),
                "tune_step": args.tune_step,
            }
            args.save_thresholds.parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_thresholds, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
    else:
        if args.threshold is not None:
            global_threshold = args.threshold
        else:
            per_label = load_thresholds(args.thresholds)
            if per_label is None:
                global_threshold = DEFAULT_GLOBAL_THRESHOLD

    thresholds = build_thresholds(per_label, global_threshold)
    result = evaluate(rows, model, thresholds, allow_empty=args.allow_empty)

    print(f"Evaluated {len(rows)} samples")
    if per_label:
        print("Thresholds: per-label")
    else:
        print(f"Thresholds: global {global_threshold:.2f}")

    print("\nOverall:")
    print(
        f"  exact match accuracy: {result['exact_match']:.3f}\n"
        f"  micro precision/recall/f1: {result['micro']['precision']:.3f} "
        f"{result['micro']['recall']:.3f} {result['micro']['f1']:.3f}\n"
        f"  macro precision/recall/f1: {result['macro']['precision']:.3f} "
        f"{result['macro']['recall']:.3f} {result['macro']['f1']:.3f}"
    )

    print("\nPer-class metrics:")
    for cls in CLASSES:
        metrics = result["metrics"][cls]
        print(
            f"  {cls:18s} "
            f"p={metrics['precision']:.3f} "
            f"r={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} "
            f"fpr={metrics['fpr']:.3f} "
            f"fnr={metrics['fnr']:.3f} "
            f"support={int(metrics['support']):4d} "
            f"thr={thresholds[cls]:.2f}"
        )

    if args.tune and args.save_thresholds:
        print(f"\nWrote tuned thresholds to {args.save_thresholds}")


if __name__ == "__main__":
    main()
