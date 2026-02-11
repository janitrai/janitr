#!/usr/bin/env python3
"""
Create co-occurrence-aware train/valid/calib/holdout splits from JSONL.

Key goals:
- Preserve per-label prevalence across splits.
- Preserve pairwise co-occurrence prevalence (e.g., scam+topic_crypto).
- Produce fastText TXT files and split JSONL files for downstream pipeline steps.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable

from prepare_data import SCAM_RAW_LABELS, clean_text, extract_raw_labels  # type: ignore

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "sample.jsonl"

DEFAULT_TRAIN_JSONL = REPO_ROOT / "data" / "train.jsonl"
DEFAULT_VALID_JSONL = REPO_ROOT / "data" / "valid.jsonl"
DEFAULT_CALIB_JSONL = REPO_ROOT / "data" / "calib.jsonl"
DEFAULT_HOLDOUT_JSONL = REPO_ROOT / "data" / "holdout.jsonl"

DEFAULT_TRAIN_TXT = REPO_ROOT / "data" / "train.txt"
DEFAULT_VALID_TXT = REPO_ROOT / "data" / "valid.txt"
DEFAULT_CALIB_TXT = REPO_ROOT / "data" / "calib.txt"
DEFAULT_HOLDOUT_TXT = REPO_ROOT / "data" / "holdout.txt"

DEFAULT_META = REPO_ROOT / "data" / "stratified_split_meta.json"
DEFAULT_REPORT = REPO_ROOT / "data" / "stratified_split_report.json"

CLASSES = ["clean", "topic_crypto", "scam"]


@dataclass(frozen=True)
class SplitSpec:
    name: str
    ratio: float
    jsonl_path: Path
    txt_path: Path


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        text = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def normalize_labels(raw_labels: list[str]) -> list[str]:
    labels = []
    for label in raw_labels:
        item = str(label).strip().lower()
        if not item:
            continue
        if item == "crypto":
            item = "topic_crypto"
        labels.append(item)
    return labels


def map_training_labels(raw_labels: list[str]) -> list[str]:
    """
    Map raw taxonomy labels into Janitr's 3-class training label space.

    Unlike the old mutually-exclusive collapse, this mapping preserves
    scam/topic co-occurrence to avoid split mismatch across calibration/holdout.
    """

    raw_set = set(normalize_labels(raw_labels))

    out: set[str] = set()

    has_topic_crypto = "topic_crypto" in raw_set
    has_scam = bool(raw_set & SCAM_RAW_LABELS)

    if has_topic_crypto:
        out.add("topic_crypto")
    if has_scam:
        out.add("scam")

    if not out:
        out.add("clean")

    return sorted(out)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def write_fasttext(path: Path, rows: list[tuple[list[str], str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for labels, text in rows:
            if not labels:
                continue
            prefix = " ".join(f"__label__{label}" for label in labels)
            handle.write(f"{prefix} {text}\n")


def largest_remainder_counts(total: int, ratios: list[float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    if remainder > 0:
        fractions = sorted(
            ((raw[i] - counts[i], i) for i in range(len(ratios))), reverse=True
        )
        for _, idx in fractions[:remainder]:
            counts[idx] += 1
    return counts


def feature_tokens(labels: list[str]) -> list[str]:
    feats: list[str] = []
    unique = sorted(set(labels))
    for label in unique:
        feats.append(f"L:{label}")
    for a, b in combinations(unique, 2):
        feats.append(f"P:{a}|{b}")
    return feats


def format_rate(value: float) -> str:
    return f"{value:.4f}"


def build_distribution_report(
    split_rows: dict[str, list[tuple[list[str], str]]],
) -> dict[str, object]:
    totals = Counter()
    pair_totals = Counter()
    total_rows = 0

    for rows in split_rows.values():
        total_rows += len(rows)
        for labels, _ in rows:
            unique = sorted(set(labels))
            for label in unique:
                totals[label] += 1
            for a, b in combinations(unique, 2):
                pair_totals[f"{a}|{b}"] += 1

    global_label_rates = {
        label: (totals[label] / total_rows if total_rows else 0.0) for label in CLASSES
    }
    global_pair_rates = {
        pair: (count / total_rows if total_rows else 0.0)
        for pair, count in sorted(pair_totals.items())
    }

    split_reports: dict[str, dict[str, object]] = {}
    max_label_delta = 0.0
    max_pair_delta = 0.0

    for split, rows in split_rows.items():
        row_count = len(rows)
        labels_counter = Counter()
        pairs_counter = Counter()

        for labels, _ in rows:
            unique = sorted(set(labels))
            for label in unique:
                labels_counter[label] += 1
            for a, b in combinations(unique, 2):
                pairs_counter[f"{a}|{b}"] += 1

        label_rates: dict[str, float] = {}
        label_delta_abs: dict[str, float] = {}
        for label in CLASSES:
            rate = labels_counter[label] / row_count if row_count else 0.0
            label_rates[label] = rate
            delta = abs(rate - global_label_rates[label])
            label_delta_abs[label] = delta
            max_label_delta = max(max_label_delta, delta)

        pair_rates: dict[str, float] = {}
        pair_delta_abs: dict[str, float] = {}
        all_pairs = sorted(set(global_pair_rates) | set(pairs_counter.keys()))
        for pair in all_pairs:
            rate = pairs_counter[pair] / row_count if row_count else 0.0
            pair_rates[pair] = rate
            delta = abs(rate - global_pair_rates.get(pair, 0.0))
            pair_delta_abs[pair] = delta
            max_pair_delta = max(max_pair_delta, delta)

        split_reports[split] = {
            "rows": row_count,
            "label_counts": dict(labels_counter),
            "pair_counts": dict(pairs_counter),
            "label_rates": label_rates,
            "pair_rates": pair_rates,
            "label_delta_abs": label_delta_abs,
            "pair_delta_abs": pair_delta_abs,
        }

    return {
        "total_rows": total_rows,
        "global_label_counts": {label: totals[label] for label in CLASSES},
        "global_pair_counts": dict(sorted(pair_totals.items())),
        "global_label_rates": global_label_rates,
        "global_pair_rates": global_pair_rates,
        "splits": split_reports,
        "max_label_delta_abs": max_label_delta,
        "max_pair_delta_abs": max_pair_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)

    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--calib-ratio", type=float, default=0.05)
    parser.add_argument("--holdout-ratio", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--valid-jsonl", type=Path, default=DEFAULT_VALID_JSONL)
    parser.add_argument("--calib-jsonl", type=Path, default=DEFAULT_CALIB_JSONL)
    parser.add_argument("--holdout-jsonl", type=Path, default=DEFAULT_HOLDOUT_JSONL)

    parser.add_argument("--train-txt", type=Path, default=DEFAULT_TRAIN_TXT)
    parser.add_argument("--valid-txt", type=Path, default=DEFAULT_VALID_TXT)
    parser.add_argument("--calib-txt", type=Path, default=DEFAULT_CALIB_TXT)
    parser.add_argument("--holdout-txt", type=Path, default=DEFAULT_HOLDOUT_TXT)

    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)

    parser.add_argument("--strip-urls", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-lowercase", action="store_true")

    parser.add_argument("--max-label-delta", type=float, default=0.03)
    parser.add_argument("--max-pair-delta", type=float, default=0.05)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if split distribution drift exceeds thresholds.",
    )

    args = parser.parse_args()

    ratios = [args.train_ratio, args.valid_ratio, args.calib_ratio, args.holdout_ratio]
    if any(r <= 0 for r in ratios):
        raise SystemExit("All split ratios must be > 0")

    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 1e-9:
        raise SystemExit(f"Split ratios must sum to 1.0. Got {ratio_sum:.8f}")

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    split_specs = [
        SplitSpec("train", args.train_ratio, args.train_jsonl, args.train_txt),
        SplitSpec("valid", args.valid_ratio, args.valid_jsonl, args.valid_txt),
        SplitSpec("calib", args.calib_ratio, args.calib_jsonl, args.calib_txt),
        SplitSpec("holdout", args.holdout_ratio, args.holdout_jsonl, args.holdout_txt),
    ]

    records: list[dict[str, object]] = []
    skipped_empty = 0

    with args.input.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)

            raw_labels = extract_raw_labels(obj)
            training_labels = map_training_labels(raw_labels)

            raw_text = obj.get("text") or obj.get("raw_text") or ""
            cleaned = clean_text(
                raw_text,
                normalize=not args.no_normalize,
                lowercase=not args.no_lowercase,
                strip_urls=args.strip_urls,
            )
            if not cleaned:
                skipped_empty += 1
                continue

            record = {
                "line_no": line_no,
                "id": str(obj.get("id", "")),
                "obj": obj,
                "raw_labels": raw_labels,
                "training_labels": training_labels,
                "features": feature_tokens(training_labels),
                "text": cleaned,
                "ts": parse_time(obj.get("collected_at")),
            }
            records.append(record)

    if not records:
        raise SystemExit("No valid rows found after cleaning.")

    rng = random.Random(args.seed)
    rng.shuffle(records)

    split_names = [spec.name for spec in split_specs]
    split_ratios = {spec.name: spec.ratio for spec in split_specs}

    target_sizes = largest_remainder_counts(
        len(records), [spec.ratio for spec in split_specs]
    )
    target_size_by_split = {
        spec.name: target_sizes[idx] for idx, spec in enumerate(split_specs)
    }

    split_buckets: dict[str, list[dict[str, object]]] = {
        split: [] for split in split_names
    }

    # Co-occurrence-aware assignment by exact labelset preserves pair prevalence
    # much better than naive time/random splits.
    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for rec in records:
        key = tuple(sorted(rec["training_labels"]))  # type: ignore[arg-type]
        groups[key].append(rec)

    group_keys = sorted(groups.keys(), key=lambda key: len(groups[key]), reverse=True)
    for key in group_keys:
        bucket = groups[key]
        rng.shuffle(bucket)
        group_size = len(bucket)
        desired = {split: split_ratios[split] * group_size for split in split_names}
        assigned = {split: 0 for split in split_names}

        for rec in bucket:
            candidates = [
                split
                for split in split_names
                if len(split_buckets[split]) < target_size_by_split[split]
            ]
            if not candidates:
                candidates = split_names[:]

            # Fill the most under-assigned split for this labelset first.
            best_split = max(
                candidates,
                key=lambda split: (
                    desired[split] - assigned[split],
                    target_size_by_split[split] - len(split_buckets[split]),
                ),
            )
            split_buckets[best_split].append(rec)
            assigned[best_split] += 1

    split_json_rows: dict[str, list[dict]] = {}
    split_txt_rows: dict[str, list[tuple[list[str], str]]] = {}

    min_dt = datetime.min.replace(tzinfo=timezone.utc)

    for spec in split_specs:
        chunk = split_buckets[spec.name]
        chunk.sort(key=lambda row: (row["ts"] or min_dt, row["id"]))

        json_rows: list[dict] = []
        txt_rows: list[tuple[list[str], str]] = []

        for row in chunk:
            obj = dict(row["obj"])  # type: ignore[arg-type]
            training_labels = list(row["training_labels"])  # type: ignore[arg-type]

            if "labels" in obj:
                obj["raw_labels"] = obj["labels"]
            elif "label" in obj:
                obj["raw_labels"] = [obj["label"]]
            else:
                obj["raw_labels"] = []

            obj["labels"] = training_labels
            obj.pop("label", None)

            json_rows.append(obj)
            txt_rows.append((training_labels, str(row["text"])))

        split_json_rows[spec.name] = json_rows
        split_txt_rows[spec.name] = txt_rows

        write_jsonl(spec.jsonl_path, json_rows)
        write_fasttext(spec.txt_path, txt_rows)

    report = build_distribution_report(split_txt_rows)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    drift_label = float(report["max_label_delta_abs"])
    drift_pair = float(report["max_pair_delta_abs"])

    meta = {
        "input": str(args.input),
        "seed": args.seed,
        "ratios": {spec.name: spec.ratio for spec in split_specs},
        "target_sizes": target_size_by_split,
        "actual_sizes": {split: len(split_json_rows[split]) for split in split_names},
        "total_rows": len(records),
        "skipped_empty": skipped_empty,
        "strip_urls": bool(args.strip_urls),
        "normalize": not args.no_normalize,
        "lowercase": not args.no_lowercase,
        "report": str(args.report_out),
        "max_label_delta_abs": drift_label,
        "max_pair_delta_abs": drift_pair,
        "thresholds": {
            "max_label_delta": args.max_label_delta,
            "max_pair_delta": args.max_pair_delta,
        },
    }
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(
        f"Loaded {len(records)} rows from {args.input} (skipped empty={skipped_empty})"
    )
    for spec in split_specs:
        print(
            f"{spec.name:7s} rows={len(split_json_rows[spec.name]):4d} "
            f"ratio={len(split_json_rows[spec.name]) / len(records):.4f}"
        )

    print("\nGlobal label rates:")
    global_label_rates = report["global_label_rates"]  # type: ignore[assignment]
    for label in CLASSES:
        print(f"  {label:12s} {format_rate(global_label_rates[label])}")

    print("\nMax distribution deltas:")
    print(f"  label delta abs max = {drift_label:.4f}")
    print(f"  pair  delta abs max = {drift_pair:.4f}")

    print(f"\nWrote report: {args.report_out}")
    print(f"Wrote meta:   {args.meta_out}")

    failed = False
    if drift_label > args.max_label_delta:
        print(
            f"ERROR: label drift {drift_label:.4f} exceeds threshold {args.max_label_delta:.4f}"
        )
        failed = True
    if drift_pair > args.max_pair_delta:
        print(
            f"ERROR: pair drift {drift_pair:.4f} exceeds threshold {args.max_pair_delta:.4f}"
        )
        failed = True

    if failed and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
