#!/usr/bin/env python3
"""
Mine false positives/negatives from labeled JSONL using a fastText model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_data import clean_text, map_label  # reuse training preprocessing

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "scam_detector.bin"
DEFAULT_INPUT = REPO_ROOT / "data" / "sample.jsonl"
DEFAULT_FP_OUT = REPO_ROOT / "data" / "false_positives.jsonl"
DEFAULT_FN_OUT = REPO_ROOT / "data" / "false_negatives.jsonl"


def get_p_scam(model, text: str) -> float:
    labels, probs = model.predict(text, k=2)
    for label, prob in zip(labels, probs):
        if label == "__label__scam":
            return float(prob)
    return 0.0


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine fastText errors from labeled JSONL")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model file")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Predict scam when p>=threshold")
    parser.add_argument("--fp-out", type=Path, default=DEFAULT_FP_OUT, help="False positives JSONL")
    parser.add_argument("--fn-out", type=Path, default=DEFAULT_FN_OUT, help="False negatives JSONL")
    parser.add_argument(
        "--hard-negatives-out",
        type=Path,
        default=None,
        help="Write false positives to fastText format for oversampling",
    )
    parser.add_argument(
        "--hard-negatives-max",
        type=int,
        default=0,
        help="Max hard negatives to write (0 = all)",
    )
    parser.add_argument("--strip-urls", action="store_true", help="Remove URLs from text")
    parser.add_argument("--no-normalize", action="store_true", help="Disable Unicode normalization")
    parser.add_argument("--no-lowercase", action="store_true", help="Disable lowercasing")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}")
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    if args.hard_negatives_max < 0:
        raise SystemExit("--hard-negatives-max must be >= 0")

    try:
        import fasttext  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "fasttext is not installed. Install with: pip install fasttext-wheel"
        ) from exc

    model = fasttext.load_model(str(args.model))

    false_positives: list[dict] = []
    false_negatives: list[dict] = []
    hard_negatives_written = 0

    hard_negatives_file = None
    if args.hard_negatives_out is not None:
        args.hard_negatives_out.parent.mkdir(parents=True, exist_ok=True)
        hard_negatives_file = open(args.hard_negatives_out, "w", encoding="utf-8")

    total = 0
    skipped_unknown = 0
    skipped_empty = 0

    for sample in iter_jsonl(args.input):
        label = map_label(sample.get("label", ""))
        if label is None:
            skipped_unknown += 1
            continue

        text = sample.get("text") or sample.get("raw_text") or ""
        cleaned = clean_text(
            text,
            normalize=not args.no_normalize,
            lowercase=not args.no_lowercase,
            strip_urls=args.strip_urls,
        )
        if not cleaned:
            skipped_empty += 1
            continue

        total += 1
        p_scam = get_p_scam(model, cleaned)
        pred = "scam" if p_scam >= args.threshold else "clean"

        if label == "clean" and pred == "scam":
            out = dict(sample)
            out["model_p_scam"] = p_scam
            out["model_pred"] = pred
            false_positives.append(out)
            if hard_negatives_file is not None:
                if args.hard_negatives_max == 0 or hard_negatives_written < args.hard_negatives_max:
                    hard_negatives_file.write(f"__label__clean {cleaned}\n")
                    hard_negatives_written += 1
        elif label == "scam" and pred == "clean":
            out = dict(sample)
            out["model_p_scam"] = p_scam
            out["model_pred"] = pred
            false_negatives.append(out)

    if hard_negatives_file is not None:
        hard_negatives_file.close()

    write_jsonl(args.fp_out, false_positives)
    write_jsonl(args.fn_out, false_negatives)

    print(f"Scored {total} labeled samples from {args.input}")
    print(f"Threshold: p(scam) >= {args.threshold:.2f}")
    print(f"False positives: {len(false_positives)} -> {args.fp_out}")
    print(f"False negatives: {len(false_negatives)} -> {args.fn_out}")
    if args.hard_negatives_out is not None:
        print(f"Hard negatives written: {hard_negatives_written} -> {args.hard_negatives_out}")
    if skipped_unknown or skipped_empty:
        print("Skipped rows:")
        print(f"  unknown label: {skipped_unknown}")
        print(f"  empty after cleaning: {skipped_empty}")


if __name__ == "__main__":
    main()
