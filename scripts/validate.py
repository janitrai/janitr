#!/usr/bin/env python3
"""
Validate data files against JSON schemas.

Usage:
    python scripts/validate.py data/snapshots/          # validate snapshot files
    python scripts/validate.py data/sample.jsonl --schema labeled  # validate labeled data
    python scripts/validate.py --check-schemas          # validate schema files themselves
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from labelset import load_v2026_labels_from_labels_md

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except ImportError:
    print(
        "Error: jsonschema not installed. Install Python deps with: cd scripts && uv sync",
        file=sys.stderr,
    )
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent
SCHEMAS_DIR = REPO_ROOT / "docs" / "schemas"


def build_labeled_sample_schema() -> dict[str, Any]:
    """JSON Schema for labeled samples, using the canonical v2026 label set."""

    try:
        labels = load_v2026_labels_from_labels_md()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Label guide not found. Expected docs/LABELS.md to exist in the repo."
        ) from e

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Labeled Sample",
        "type": "object",
        # Labeled samples are often produced from partial UI scrapes; require only
        # the minimum needed for training/inference workflows.
        "required": ["id", "text", "labels"],
        "additionalProperties": True,
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "platform": {
                "type": "string",
                "enum": ["x", "discord", "web", "dm", "other"],
            },
            "source_id": {"type": "string"},
            "source_url": {"type": "string"},
            "collected_at": {"type": "string", "format": "date-time"},
            # Common alternate fields in some pipelines.
            "source": {"type": "string"},
            "collected": {"type": "string"},
            "text": {"type": "string"},
            "urls": {"type": "array", "items": {"type": "string"}},
            "addresses": {"type": "array", "items": {"type": "string"}},
            "labels": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "enum": labels,
                },
            },
            "notes": {"type": "string"},
            # Deprecated single-label field from older tooling. If present, it must
            # still be a valid v2026 label.
            "label": {"type": "string", "enum": labels},
        },
    }


def load_schema(schema_name: str) -> dict[str, Any]:
    """Load a schema by name."""
    if schema_name == "labeled":
        return build_labeled_sample_schema()

    schema_path = SCHEMAS_DIR / f"{schema_name}.schema.json"
    if not schema_path.exists():
        # Try without .schema suffix
        schema_path = SCHEMAS_DIR / f"{schema_name}.json"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_name}")

    with open(schema_path) as f:
        return json.load(f)


def validate_jsonl(
    file_path: Path, schema: dict[str, Any]
) -> tuple[int, int, list[str]]:
    """Validate a JSONL file. Returns (valid_count, error_count, errors)."""
    validator = Draft202012Validator(schema)
    valid = 0
    errors_list = []

    with open(file_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors_list.append(f"  Line {line_num}: Invalid JSON - {e}")
                continue

            errs = list(validator.iter_errors(record))
            if errs:
                for err in errs:
                    path = ".".join(str(p) for p in err.absolute_path) or "(root)"
                    errors_list.append(f"  Line {line_num}, {path}: {err.message}")
            else:
                valid += 1

    return valid, len(errors_list), errors_list


def validate_json(file_path: Path, schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a single JSON file. Returns (is_valid, errors)."""
    validator = Draft202012Validator(schema)

    with open(file_path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {e}"]

    errs = list(validator.iter_errors(data))
    if errs:
        errors_list = []
        for err in errs:
            path = ".".join(str(p) for p in err.absolute_path) or "(root)"
            errors_list.append(f"  {path}: {err.message}")
        return False, errors_list

    return True, []


def check_schemas() -> int:
    """Validate that all schema files are valid JSON Schema."""
    print("Checking schema files...")
    errors = 0

    for schema_file in SCHEMAS_DIR.glob("*.json"):
        try:
            with open(schema_file) as f:
                schema = json.load(f)
            # Check it's a valid schema
            Draft202012Validator.check_schema(schema)
            print(f"  ✓ {schema_file.name}")
        except json.JSONDecodeError as e:
            print(f"  ✗ {schema_file.name}: Invalid JSON - {e}")
            errors += 1
        except jsonschema.SchemaError as e:
            print(f"  ✗ {schema_file.name}: Invalid schema - {e.message}")
            errors += 1

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate data files against schemas")
    parser.add_argument("path", nargs="?", help="File or directory to validate")
    parser.add_argument(
        "--schema",
        default="x-post-snapshot",
        help="Schema to use: x-post-snapshot, labeled, or path to schema file",
    )
    parser.add_argument(
        "--check-schemas", action="store_true", help="Validate schema files"
    )
    args = parser.parse_args()

    if args.check_schemas:
        errors = check_schemas()
        sys.exit(1 if errors else 0)

    if not args.path:
        parser.print_help()
        sys.exit(1)

    path = Path(args.path)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    # Load schema
    try:
        schema = load_schema(args.schema)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Using schema: {args.schema}")
    print()

    total_valid = 0
    total_errors = 0
    all_errors = []

    # Collect files to validate
    if path.is_file():
        files = [path]
    else:
        files = list(path.glob("**/*.jsonl")) + list(path.glob("**/*.json"))
        files = [f for f in files if not f.name.endswith(".schema.json")]

    if not files:
        print(f"No .json or .jsonl files found in {path}")
        sys.exit(1)

    for file_path in sorted(files):
        rel_path = (
            file_path.relative_to(REPO_ROOT)
            if file_path.is_relative_to(REPO_ROOT)
            else file_path
        )

        if file_path.suffix == ".jsonl":
            valid, err_count, errors = validate_jsonl(file_path, schema)
            total_valid += valid
            total_errors += err_count
            status = "✓" if err_count == 0 else "✗"
            print(f"{status} {rel_path}: {valid} valid, {err_count} errors")
            if errors:
                all_errors.extend([f"{rel_path}:"] + errors)
        else:
            is_valid, errors = validate_json(file_path, schema)
            if is_valid:
                total_valid += 1
                print(f"✓ {rel_path}")
            else:
                total_errors += 1
                print(f"✗ {rel_path}")
                all_errors.extend([f"{rel_path}:"] + errors)

    print()
    print(f"Total: {total_valid} valid, {total_errors} errors")

    if all_errors:
        print()
        print("Errors:")
        for err in all_errors:
            print(err)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
