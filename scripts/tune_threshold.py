#!/usr/bin/env python3
"""
Tune thresholds for the reduced quant-cutoff10k fastText model.
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "reduced" / "quant-cutoff10k.ftz"
DEFAULT_VALID = REPO_ROOT / "data" / "valid.txt"

CLASSES = ["clean", "topic_crypto", "scam"]
THRESHOLDS = [0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.985, 0.99, 0.995]
TARGET_FPR = 0.05


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


def build_confusion(
    rows: list[tuple[str, float]], threshold: float
) -> dict[str, dict[str, int]]:
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
    parser = argparse.ArgumentParser(
        description="Tune threshold for reduced fastText model"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model file")
    parser.add_argument(
        "--valid", type=Path, default=DEFAULT_VALID, help="Validation file"
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
            "fasttext is not installed. Install Python deps with: cd scripts && uv sync"
        ) from exc

    model = fasttext.load_model(str(args.model))

    rows: list[tuple[str, float]] = []
    total = 0
    with open(args.valid, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            actual, text = parsed
            if actual not in CLASSES:
                continue
            scores = get_probs(model, text)
            rows.append((actual, scores["scam"]))
            total += 1

    if total == 0:
        raise SystemExit("No valid rows found in validation file.")

    print(f"Evaluated {total} samples")
    print(f"Model: {args.model}")
    print("\nThreshold tuning (predict scam if p>=threshold)")
    header = f"{'thr':>6s} {'fpr':>7s} {'prec':>7s} {'rec':>7s} {'f1':>7s}"
    print(header)

    results: list[tuple[float, dict[str, float]]] = []
    for thr in THRESHOLDS:
        confusion = build_confusion(rows, thr)
        metrics = summarize(confusion)
        results.append((thr, metrics))
        print(
            f"{thr:6.2f} {metrics['fpr']:7.3f} {metrics['scam_precision']:7.3f} "
            f"{metrics['scam_recall']:7.3f} {metrics['scam_f1']:7.3f}"
        )

    best: tuple[float, dict[str, float]] | None = None
    for thr, metrics in results:
        if metrics["fpr"] <= TARGET_FPR:
            best = (thr, metrics)
            break

    if best is None:
        print(f"\nNo threshold met FPR <= {TARGET_FPR:.3f}")
        return

    thr, metrics = best
    print(
        f"\nLowest threshold with FPR <= {TARGET_FPR:.3f}: "
        f"thr={thr:.2f} "
        f"precision={metrics['scam_precision']:.3f} "
        f"recall={metrics['scam_recall']:.3f} "
        f"f1={metrics['scam_f1']:.3f} "
        f"fpr={metrics['fpr']:.3f}"
    )


if __name__ == "__main__":
    main()
