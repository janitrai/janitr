#!/usr/bin/env python3
"""
Evaluate a fastText model and print metrics.
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "scam_detector.bin"
DEFAULT_VALID = REPO_ROOT / "data" / "valid.txt"

CLASSES = ["clean", "crypto", "scam"]


def get_probs(model, text: str) -> dict[str, float]:
    labels, probs = model.predict(text, k=len(CLASSES))
    scores: dict[str, float] = {cls: 0.0 for cls in CLASSES}
    for label, prob in zip(labels, probs):
        cls = label.replace("__label__", "", 1)
        if cls in scores:
            scores[cls] = float(prob)
    return scores


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


def build_confusion(rows: list[tuple[str, float]], threshold: float) -> dict[str, dict[str, int]]:
    confusion = {actual: {pred: 0 for pred in CLASSES} for actual in CLASSES}
    for actual, p_scam in rows:
        pred = "scam" if p_scam >= threshold else "clean"
        confusion[actual][pred] += 1
    return confusion


def summarize(confusion: dict[str, dict[str, int]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for cls in CLASSES:
        tp = confusion[cls][cls]
        fp = sum(confusion[a][cls] for a in CLASSES if a != cls)
        fn = sum(confusion[cls][p] for p in CLASSES if p != cls)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        metrics[f"{cls}_precision"] = precision
        metrics[f"{cls}_recall"] = recall
        metrics[f"{cls}_f1"] = f1
    total_clean = sum(confusion["clean"].values())
    total_scam = sum(confusion["scam"].values())
    fp = confusion["clean"]["scam"]
    fn = confusion["scam"]["clean"]
    metrics["fpr"] = safe_div(fp, total_clean)
    metrics["fnr"] = safe_div(fn, total_scam)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fastText model")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model file")
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID, help="Validation file")
    parser.add_argument("--fp-limit", type=int, default=200, help="Max false positives to print")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Predict scam when p(scam) >= threshold (defaults to argmax)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep thresholds and print metrics table",
    )
    parser.add_argument(
        "--sweep-step",
        type=float,
        default=0.02,
        help="Step size for threshold sweep",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=None,
        help="Find highest-recall threshold with FPR <= target",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=None,
        help="Find highest-recall threshold with precision >= target",
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

    model = fasttext.load_model(str(args.model))

    false_positives: list[str] = []
    total = 0
    rows: list[tuple[str, float]] = []
    row_texts: list[tuple[str, str, float]] = []

    with open(args.valid, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            actual, text = parsed
            if actual not in CLASSES:
                continue
            scores = get_probs(model, text)
            p_scam = scores["scam"]
            rows.append((actual, p_scam))
            row_texts.append((actual, text, p_scam))
            total += 1

    if total == 0:
        raise SystemExit("No valid rows found in validation file.")

    if args.sweep:
        step = args.sweep_step
        if step <= 0 or step >= 1:
            raise SystemExit("--sweep-step must be > 0 and < 1")
        print(f"Evaluated {total} samples")
        print("\nThreshold sweep (predict scam if p>=threshold)")
        header = f"{'thr':>6s} {'fpr':>7s} {'fnr':>7s} {'prec':>7s} {'rec':>7s} {'f1':>7s}"
        print(header)
        thr = 0.0
        sweep_rows: list[tuple[float, dict[str, float]]] = []
        while thr <= 1.000001:
            confusion = build_confusion(rows, thr)
            metrics = summarize(confusion)
            sweep_rows.append((thr, metrics))
            print(
                f"{thr:6.2f} {metrics['fpr']:7.3f} {metrics['fnr']:7.3f} "
                f"{metrics['scam_precision']:7.3f} {metrics['scam_recall']:7.3f} {metrics['scam_f1']:7.3f}"
            )
            thr += step

        def pick_by_target(kind: str, target: float) -> None:
            best: tuple[float, dict[str, float]] | None = None
            for thr_value, metrics in sweep_rows:
                if kind == "fpr" and metrics["fpr"] > target:
                    continue
                if kind == "precision" and metrics["scam_precision"] < target:
                    continue
                if best is None or metrics["scam_recall"] > best[1]["scam_recall"]:
                    best = (thr_value, metrics)
            if best is None:
                print(f"\nNo threshold met target {kind}={target:.3f}")
            else:
                thr_value, metrics = best
                print(
                    f"\nBest recall under target {kind}={target:.3f}: "
                    f"thr={thr_value:.2f} "
                    f"precision={metrics['scam_precision']:.3f} "
                    f"recall={metrics['scam_recall']:.3f} "
                    f"fpr={metrics['fpr']:.3f}"
                )

        if args.target_fpr is not None:
            pick_by_target("fpr", args.target_fpr)
        if args.target_precision is not None:
            pick_by_target("precision", args.target_precision)
        return

    threshold = args.threshold
    if threshold is None:
        threshold = 0.5
        threshold_note = "argmax (equivalent to threshold 0.50)"
    else:
        threshold_note = f"threshold={threshold:.2f}"

    confusion = build_confusion(rows, threshold)
    for actual, text, p_scam in row_texts:
        pred = "scam" if p_scam >= threshold else "clean"
        if actual == "clean" and pred == "scam":
            false_positives.append(text)

    print(f"Evaluated {total} samples")
    print(f"Decision rule: {threshold_note}")
    print("\nConfusion matrix (rows=actual, cols=predicted)")
    header = f"{'':12s} {'pred_clean':>10s} {'pred_scam':>10s}"
    print(header)
    for actual in CLASSES:
        row = confusion[actual]
        print(f"{('actual_' + actual):12s} {row['clean']:10d} {row['scam']:10d}")

    print("\nPer-class metrics:")
    metrics = summarize(confusion)
    for cls in CLASSES:
        print(
            f"  {cls:5s} | precision {metrics[cls + '_precision']:.3f} "
            f"| recall {metrics[cls + '_recall']:.3f} | f1 {metrics[cls + '_f1']:.3f}"
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
