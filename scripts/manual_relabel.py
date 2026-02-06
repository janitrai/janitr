#!/usr/bin/env python3
"""
Apply manual label changes to the dataset.

READ-ONLY by default. Use --apply to write changes.

Usage:
    # Preview changes from a changes file
    python scripts/manual_relabel.py data/sample.jsonl --changes changes.txt

    # Apply changes
    python scripts/manual_relabel.py data/sample.jsonl --changes changes.txt --apply

Changes file format (one per line):
    # Single label
    x_0001 topic_crypto

    # Multiple labels (space or comma separated)
    x_0394 topic_crypto scam
    x_0397 topic_crypto,scam
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


VALID_LABELS = {"clean", "topic_crypto", "scam", "promo", "ai_generated_reply"}


def normalize_labels(labels: list[str]) -> list[str]:
    normalized: list[str] = []
    for label in labels:
        if label == "crypto_scam":
            normalized.extend(["topic_crypto", "scam"])
            continue
        if label in VALID_LABELS:
            normalized.append(label)
    seen = set()
    deduped = []
    for label in normalized:
        if label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    if "clean" in deduped and len(deduped) > 1:
        deduped = [label for label in deduped if label != "clean"]
    return deduped


def parse_labels(tokens: list[str]) -> list[str]:
    labels: list[str] = []
    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if part:
                labels.append(part)
    return normalize_labels(labels)


def load_changes(changes_path: Path) -> dict[str, list[str]]:
    """Load id -> labels mapping from changes file."""
    changes: dict[str, list[str]] = {}
    for line in changes_path.open():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            id_, label_tokens = parts[0], parts[1:]
            labels = parse_labels(label_tokens)
            if not labels:
                print(f"⚠️  Skipping {id_}: no valid labels found", file=sys.stderr)
                continue
            changes[id_] = labels
    return changes


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", help="Path to JSONL file")
    ap.add_argument("--changes", "-c", required=True, help="Path to changes file")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes (default: preview only)",
    )
    args = ap.parse_args()

    path = Path(args.path)
    changes = load_changes(Path(args.changes))

    if not changes:
        print("No changes found in changes file.", file=sys.stderr)
        return

    print(f"Loaded {len(changes)} changes from {args.changes}", file=sys.stderr)

    # Process file
    applied = 0
    not_found = set(changes.keys())
    output_lines = []

    for line in path.open(encoding="utf-8"):
        obj = json.loads(line)
        id_ = obj.get("id")
        if id_ in changes:
            old_labels = []
            if isinstance(obj.get("labels"), list):
                old_labels = normalize_labels(obj.get("labels"))
            elif isinstance(obj.get("label"), str):
                old_labels = normalize_labels([obj.get("label")])
            new_labels = changes[id_]
            if (
                old_labels != new_labels
                or obj.get("labels") != new_labels
                or "label" in obj
            ):
                print(f"  {id_}: {old_labels} → {new_labels}")
                obj["labels"] = new_labels
                obj.pop("label", None)
                applied += 1
            not_found.discard(id_)
        output_lines.append(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))

    if not_found:
        print(f"\n⚠️  IDs not found in dataset: {sorted(not_found)}", file=sys.stderr)

    print(f"\nTotal changes to apply: {applied}", file=sys.stderr)

    if not args.apply:
        print(
            "\n⚠️  DRY RUN - no changes written. Use --apply to write.", file=sys.stderr
        )
        return

    # Write atomically
    with tempfile.NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", dir=path.parent
    ) as tmp:
        tmp.write("\n".join(output_lines) + "\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
    print(f"✅ Applied {applied} changes to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
