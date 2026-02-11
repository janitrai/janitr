#!/usr/bin/env python3
"""
Fix duplicate IDs in the dataset by collapsing duplicate rows per ID.

READ-ONLY by default. Use --apply to write changes.

Strategy:
- Keep one canonical row per duplicated ID
- Prefer the row with more populated metadata/labels/text
- Merge useful missing fields from duplicate rows into the canonical row
- Remove duplicate rows (instead of renaming IDs)
- Optionally assign IDs to null entries (`--fix-nulls`)
"""

import argparse
import copy
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path


LIST_FIELDS = {"labels", "urls", "addresses"}
LONGER_TEXT_FIELDS = {"text", "notes"}


def _is_populated(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def _unique_list(existing, incoming):
    out = []
    seen = set()

    def key_for(item):
        if isinstance(item, (dict, list)):
            return json.dumps(item, sort_keys=True, ensure_ascii=False)
        return item

    for item in [*(existing or []), *(incoming or [])]:
        key = key_for(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _score_entry(entry):
    populated_fields = sum(
        1
        for k in (
            "source_id",
            "source_url",
            "platform",
            "collected_at",
            "labels",
            "text",
            "notes",
            "urls",
            "addresses",
        )
        if _is_populated(entry.get(k))
    )
    return (
        populated_fields,
        len(entry.get("labels") or []),
        len(entry.get("urls") or []),
        len(entry.get("addresses") or []),
        len((entry.get("notes") or "").strip()),
        len((entry.get("text") or "").strip()),
    )


def _merge_entries(base, other):
    for key, other_value in other.items():
        if key in LIST_FIELDS:
            base[key] = _unique_list(base.get(key), other_value)
            continue

        if key in LONGER_TEXT_FIELDS:
            base_value = base.get(key)
            base_text = base_value if isinstance(base_value, str) else ""
            other_text = other_value if isinstance(other_value, str) else ""
            if len(other_text.strip()) > len(base_text.strip()):
                base[key] = other_value
            continue

        if key == "collected_at":
            base_value = base.get(key)
            if not _is_populated(base_value) and _is_populated(other_value):
                base[key] = other_value
            elif _is_populated(base_value) and _is_populated(other_value):
                # Keep the earliest observed collection time when both are present.
                base[key] = min(str(base_value), str(other_value))
            continue

        if not _is_populated(base.get(key)) and _is_populated(other_value):
            base[key] = other_value

    return base


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "path", nargs="?", default="data/sample.jsonl", help="Path to JSONL file"
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes (default: preview only)",
    )
    ap.add_argument(
        "--fix-nulls", action="store_true", help="Also assign IDs to null entries"
    )
    args = ap.parse_args()

    path = Path(args.path)

    # First pass: collect all entries and find duplicates
    entries = []
    id_to_indexes = defaultdict(list)

    for idx, line in enumerate(path.open(encoding="utf-8")):
        obj = json.loads(line)
        entries.append(obj)
        id_ = obj.get("id")
        if id_ is not None:
            id_to_indexes[id_].append(idx)

    # Find max existing ID number for auto-assignment
    max_id_num = 0
    for id_ in id_to_indexes:
        if id_ and id_.startswith("x_"):
            try:
                num = int(id_.split("_")[1])
                max_id_num = max(max_id_num, num)
            except (ValueError, IndexError):
                pass

    auto_id_counter = max_id_num + 1

    # Report duplicates
    duplicates = {k: v for k, v in id_to_indexes.items() if len(v) > 1}
    null_count = sum(1 for e in entries if e.get("id") is None)

    print(f"Total entries: {len(entries)}")
    print(f"Unique non-null IDs: {len(id_to_indexes)}")
    print(f"Duplicate IDs: {len(duplicates)}")
    print(f"Null IDs: {null_count}")
    print()

    if duplicates:
        print("Duplicated IDs (showing first 20):")
        for id_, indexes in list(duplicates.items())[:20]:
            print(f"  {id_}: {len(indexes)} occurrences")
        if len(duplicates) > 20:
            print(f"  ... and {len(duplicates) - 20} more")
        print()

    # Build canonical merged row for each duplicate group.
    merged_by_id = {}
    duplicate_summaries = []
    duplicate_rows_removed = 0

    for id_, indexes in duplicates.items():
        scored = sorted(
            ((i, _score_entry(entries[i])) for i in indexes),
            key=lambda item: (item[1], -item[0]),
            reverse=True,
        )
        canonical_idx = scored[0][0]
        merged = copy.deepcopy(entries[canonical_idx])

        for i in indexes:
            if i == canonical_idx:
                continue
            merged = _merge_entries(merged, entries[i])

        merged_by_id[id_] = merged
        duplicate_rows_removed += len(indexes) - 1

        varying_fields = []
        all_fields = set().union(*(entries[i].keys() for i in indexes))
        for field in sorted(all_fields):
            field_values = []
            for i in indexes:
                value = entries[i].get(field)
                if isinstance(value, list):
                    value = tuple(value)
                field_values.append(value)
            if len(set(field_values)) > 1:
                varying_fields.append(field)

        duplicate_summaries.append(
            {
                "id": id_,
                "count": len(indexes),
                "canonical_line": canonical_idx + 1,
                "first_line": indexes[0] + 1,
                "removed_lines": [i + 1 for i in indexes[1:]],
                "varying_fields": varying_fields,
            }
        )

    # Second pass: emit one row per non-null ID and optionally fix null IDs.
    first_index_for_id = {id_: indexes[0] for id_, indexes in id_to_indexes.items()}
    null_changes = []
    output_entries = []

    for idx, obj in enumerate(entries):
        old_id = obj.get("id")
        current = copy.deepcopy(obj)

        if old_id is None:
            if args.fix_nulls:
                new_id = f"x_auto_{auto_id_counter:04d}"
                auto_id_counter += 1
                current["id"] = new_id
                null_changes.append(
                    {"line": idx + 1, "old_id": old_id, "new_id": new_id}
                )
            output_entries.append(current)
            continue

        if idx != first_index_for_id[old_id]:
            # True duplicate row - drop it.
            continue

        if old_id in merged_by_id:
            current = merged_by_id[old_id]

        output_entries.append(current)

    total_changes = duplicate_rows_removed + len(null_changes)
    print(f"Changes to make: {total_changes}")
    print(f"  Duplicate rows to remove: {duplicate_rows_removed}")
    print(f"  Null IDs to assign: {len(null_changes)}")

    if duplicate_summaries:
        print("\nDuplicate resolutions (first 20):")
        for info in duplicate_summaries[:20]:
            fields = ", ".join(info["varying_fields"][:5]) or "none"
            if len(info["varying_fields"]) > 5:
                fields += ", ..."
            print(
                f"  {info['id']}: {info['count']} -> 1 "
                f"(canonical line {info['canonical_line']}, removed {info['removed_lines']}, varying: {fields})"
            )
        if len(duplicate_summaries) > 20:
            print(f"  ... and {len(duplicate_summaries) - 20} more")

    if null_changes:
        print("\nNull ID assignments (first 20):")
        for item in null_changes[:20]:
            print(f"  line {item['line']}: {item['old_id']} -> {item['new_id']}")
        if len(null_changes) > 20:
            print(f"  ... and {len(null_changes) - 20} more")

    if not args.apply:
        print("\n⚠️  DRY RUN - no changes written. Use --apply to write.")
        return

    # Write atomically
    with tempfile.NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", dir=path.parent
    ) as tmp:
        for obj in output_entries:
            tmp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
    print(f"\n✅ Fixed {total_changes} entries in {path}")
    print(f"✅ New total entries: {len(output_entries)}")


if __name__ == "__main__":
    main()
