#!/usr/bin/env python3
"""
Dataset integrity checks for sample.jsonl.

Checks:
1. Valid JSON on every line
2. Required fields present (id, labels, text)
3. No duplicate IDs (excluding null)
4. Valid label values
5. No empty text
6. ID format consistency

Exit codes:
- 0: All checks passed
- 1: Errors found
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

from labelset import load_v2026_labels_from_labels_md

VALID_LABELS = set(load_v2026_labels_from_labels_md())
ID_PATTERN = re.compile(r"^x_\d+(_dup\d+)?$|^x_auto_\d+$")


def check_integrity(
    path: Path, fix_suggestions: bool = False
) -> tuple[list[str], list[str]]:
    """Run all integrity checks. Returns (errors, warnings)."""
    errors = []
    warnings = []

    id_counts = defaultdict(list)  # id -> [line_numbers]
    line_num = 0

    try:
        with path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    warnings.append(f"Line {line_num}: Empty line")
                    continue

                # Check 1: Valid JSON
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue

                # Check 2: Required fields
                id_ = obj.get("id")
                labels = obj.get("labels")
                text = obj.get("text")

                if labels is None:
                    errors.append(f"Line {line_num} (id={id_}): Missing 'labels' field")
                elif not isinstance(labels, list) or not labels:
                    errors.append(
                        f"Line {line_num} (id={id_}): 'labels' must be a non-empty list"
                    )

                if text is None:
                    errors.append(f"Line {line_num} (id={id_}): Missing 'text' field")

                # Check 3: Track IDs for duplicate detection
                if id_ is not None:
                    id_counts[id_].append(line_num)
                else:
                    warnings.append(f"Line {line_num}: Null ID")

                # Check 4: Valid labels
                if labels is not None and isinstance(labels, list):
                    if len(labels) != len(set(labels)):
                        errors.append(
                            f"Line {line_num} (id={id_}): Duplicate labels (labels={labels})"
                        )
                    for label in labels:
                        if label not in VALID_LABELS:
                            errors.append(
                                f"Line {line_num} (id={id_}): Invalid label '{label}' (valid: {VALID_LABELS})"
                            )
                    if "clean" in labels and len(labels) > 1:
                        errors.append(
                            f"Line {line_num} (id={id_}): 'clean' must be exclusive (labels={labels})"
                        )

                # Check 5: Empty text
                if text is not None and not text.strip():
                    warnings.append(f"Line {line_num} (id={id_}): Empty text")

                # Check 6: ID format (warning only)
                if id_ is not None and not ID_PATTERN.match(id_):
                    warnings.append(f"Line {line_num}: Non-standard ID format '{id_}'")

    except FileNotFoundError:
        errors.append(f"File not found: {path}")
        return errors, warnings

    # Check 3b: Report duplicates
    for id_, lines in id_counts.items():
        if len(lines) > 1:
            errors.append(f"Duplicate ID '{id_}' on lines: {lines}")

    return errors, warnings


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "path", nargs="?", default="data/sample.jsonl", help="Path to JSONL file"
    )
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    ap.add_argument("--quiet", action="store_true", help="Only output if errors found")
    args = ap.parse_args()

    path = Path(args.path)
    errors, warnings = check_integrity(path)

    exit_code = 0

    if errors:
        print(f"❌ ERRORS ({len(errors)}):")
        for e in errors[:50]:
            print(f"  {e}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more errors")
        exit_code = 1

    if warnings:
        if args.strict:
            print(f"❌ WARNINGS AS ERRORS ({len(warnings)}):")
            exit_code = 1
        elif not args.quiet:
            print(f"⚠️  WARNINGS ({len(warnings)}):")

        if not args.quiet or args.strict:
            for w in warnings[:30]:
                print(f"  {w}")
            if len(warnings) > 30:
                print(f"  ... and {len(warnings) - 30} more warnings")

    if exit_code == 0 and not args.quiet:
        # Count stats
        with path.open() as f:
            lines = [json.loads(l) for l in f if l.strip()]

        labels = defaultdict(int)
        for obj in lines:
            for label in obj.get("labels", []) or []:
                labels[label] += 1

        print(f"✅ All checks passed!")
        print(f"\nDataset stats:")
        print(f"  Total entries: {len(lines)}")
        for label, count in sorted(labels.items(), key=lambda x: -x[1]):
            print(f"  {label}: {count}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
