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
    x_0394 crypto_scam
    x_0397 crypto_scam
    x_0001 crypto
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def load_changes(changes_path: Path) -> dict[str, str]:
    """Load id -> new_label mapping from changes file."""
    changes = {}
    for line in changes_path.open():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            id_, label = parts[0], parts[1]
            changes[id_] = label
    return changes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Path to JSONL file")
    ap.add_argument("--changes", "-c", required=True, help="Path to changes file")
    ap.add_argument("--apply", action="store_true", help="Actually write changes (default: preview only)")
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
            old_label = obj.get("label")
            new_label = changes[id_]
            if old_label != new_label:
                print(f"  {id_}: {old_label} → {new_label}")
                obj["label"] = new_label
                applied += 1
            not_found.discard(id_)
        output_lines.append(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    
    if not_found:
        print(f"\n⚠️  IDs not found in dataset: {sorted(not_found)}", file=sys.stderr)
    
    print(f"\nTotal changes to apply: {applied}", file=sys.stderr)
    
    if not args.apply:
        print("\n⚠️  DRY RUN - no changes written. Use --apply to write.", file=sys.stderr)
        return
    
    # Write atomically
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=path.parent) as tmp:
        tmp.write("\n".join(output_lines) + "\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
    print(f"✅ Applied {applied} changes to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
