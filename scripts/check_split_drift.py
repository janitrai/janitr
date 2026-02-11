#!/usr/bin/env python3
"""
Check label/pair distribution drift across split TXT files.

Input format: fastText TXT with one or more __label__* prefixes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

CLASSES = ["clean", "topic_crypto", "scam"]


def parse_line(line: str) -> tuple[list[str], str] | None:
    text = line.strip()
    if not text or not text.startswith("__label__"):
        return None
    parts = text.split()
    labels: list[str] = []
    idx = 0
    for token in parts:
        if token.startswith("__label__"):
            labels.append(token.replace("__label__", "", 1))
            idx += 1
        else:
            break
    labels = sorted({label for label in labels if label in CLASSES})
    if not labels:
        return None
    body = " ".join(parts[idx:]) if idx < len(parts) else ""
    return labels, body


def load_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_line(line)
            if parsed is None:
                continue
            labels, _ = parsed
            rows.append(labels)
    return rows


def rates(
    rows: list[list[str]],
) -> tuple[dict[str, float], dict[str, float], dict[str, int], dict[str, int]]:
    label_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()

    for labels in rows:
        unique = sorted(set(labels))
        for label in unique:
            label_counts[label] += 1
        for a, b in combinations(unique, 2):
            pair_counts[f"{a}|{b}"] += 1

    n = len(rows)
    label_rates = {label: (label_counts[label] / n if n else 0.0) for label in CLASSES}
    pair_rates = {
        pair: (count / n if n else 0.0) for pair, count in sorted(pair_counts.items())
    }
    return label_rates, pair_rates, dict(label_counts), dict(pair_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.txt"))
    parser.add_argument("--valid", type=Path, default=Path("data/valid.txt"))
    parser.add_argument("--calib", type=Path, default=Path("data/calib.txt"))
    parser.add_argument("--holdout", type=Path, default=Path("data/holdout.txt"))
    parser.add_argument(
        "--report-out", type=Path, default=Path("data/split_drift_report.json")
    )
    parser.add_argument("--max-label-delta", type=float, default=0.03)
    parser.add_argument("--max-pair-delta", type=float, default=0.05)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    split_paths = {
        "train": args.train,
        "valid": args.valid,
        "calib": args.calib,
        "holdout": args.holdout,
    }

    split_rows: dict[str, list[list[str]]] = {}
    for split, path in split_paths.items():
        if not path.exists():
            raise SystemExit(f"Split file not found: {path}")
        split_rows[split] = load_rows(path)

    all_rows = [labels for rows in split_rows.values() for labels in rows]
    global_label_rates, global_pair_rates, global_label_counts, global_pair_counts = (
        rates(all_rows)
    )

    splits: dict[str, dict[str, object]] = {}
    max_label_delta = 0.0
    max_pair_delta = 0.0

    all_pair_keys = sorted(global_pair_rates.keys())

    for split, rows in split_rows.items():
        label_rates, pair_rates, label_counts, pair_counts = rates(rows)
        label_delta = {}
        pair_delta = {}

        for label in CLASSES:
            delta = abs(label_rates[label] - global_label_rates[label])
            label_delta[label] = delta
            max_label_delta = max(max_label_delta, delta)

        for pair in all_pair_keys:
            rate = pair_rates.get(pair, 0.0)
            delta = abs(rate - global_pair_rates[pair])
            pair_delta[pair] = delta
            max_pair_delta = max(max_pair_delta, delta)

        splits[split] = {
            "rows": len(rows),
            "label_counts": label_counts,
            "pair_counts": pair_counts,
            "label_rates": label_rates,
            "pair_rates": pair_rates,
            "label_delta_abs": label_delta,
            "pair_delta_abs": pair_delta,
        }

    report = {
        "global": {
            "rows": len(all_rows),
            "label_counts": global_label_counts,
            "pair_counts": global_pair_counts,
            "label_rates": global_label_rates,
            "pair_rates": global_pair_rates,
        },
        "splits": splits,
        "max_label_delta_abs": max_label_delta,
        "max_pair_delta_abs": max_pair_delta,
        "thresholds": {
            "max_label_delta": args.max_label_delta,
            "max_pair_delta": args.max_pair_delta,
        },
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Rows per split:")
    for split in ["train", "valid", "calib", "holdout"]:
        print(f"  {split:7s} {splits[split]['rows']}")

    print("\nGlobal label rates:")
    for label in CLASSES:
        print(f"  {label:12s} {global_label_rates[label]:.4f}")

    print("\nMax drift:")
    print(f"  label delta abs max = {max_label_delta:.4f}")
    print(f"  pair  delta abs max = {max_pair_delta:.4f}")
    print(f"\nWrote report: {args.report_out}")

    failed = False
    if max_label_delta > args.max_label_delta:
        print(
            f"ERROR: label drift {max_label_delta:.4f} exceeds threshold {args.max_label_delta:.4f}"
        )
        failed = True
    if max_pair_delta > args.max_pair_delta:
        print(
            f"ERROR: pair drift {max_pair_delta:.4f} exceeds threshold {args.max_pair_delta:.4f}"
        )
        failed = True

    if failed and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
