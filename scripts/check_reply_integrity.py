#!/usr/bin/env python3
"""Dataset integrity checks for replies.jsonl.

Schema:
- docs/REPLIES_DATA_MODEL.md

Checks:
1. Valid JSON on every line
2. Required top-level fields present
3. No duplicate record IDs across the file
4. Valid label values
5. Tweet object schema checks
6. Tweet role enum checks
7. parent_status_id references another tweet in-sample (or null)
8. No duplicate tweet status_id values within a sample
9. Basic type checks for optional tweet metadata

Exit codes:
- 0: All checks passed
- 1: Errors found
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from labelset import load_v2026_labels_from_labels_md

VALID_LABELS = set(load_v2026_labels_from_labels_md())

TWEET_ROLES = {"original_post", "ai_reply", "evidence", "other_reply", "context"}
METRIC_FIELDS = {
    "like_count",
    "reply_count",
    "repost_count",
    "quote_count",
    "view_count",
}


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and value >= 0


def _looks_like_x_status_id(value: object) -> bool:
    # X status IDs are numeric strings, typically 16-20 digits.
    return isinstance(value, str) and value.isdigit() and 16 <= len(value) <= 20


def _is_iso_datetime(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = value.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return True


def _validate_metrics(
    metrics: object,
    field_name: str,
    line_num: int,
    record_id: object,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(metrics, dict):
        errors.append(
            f"Line {line_num} (id={record_id}): '{field_name}' must be an object"
        )
        return

    for key, value in metrics.items():
        if key not in METRIC_FIELDS:
            warnings.append(
                f"Line {line_num} (id={record_id}): '{field_name}' contains unknown metric '{key}'"
            )
            continue
        if not _is_non_negative_int(value):
            errors.append(
                f"Line {line_num} (id={record_id}): '{field_name}.{key}' must be a non-negative integer"
            )


def _validate_tweet(
    tweet: object,
    tweet_idx: int,
    line_num: int,
    record_id: object,
    errors: list[str],
    warnings: list[str],
) -> dict[str, object] | None:
    prefix = f"Line {line_num} (id={record_id}) tweets[{tweet_idx}]"

    if not isinstance(tweet, dict):
        errors.append(f"{prefix}: tweet must be an object")
        return None

    status_id = tweet.get("status_id")
    if not _is_non_empty_string(status_id):
        errors.append(f"{prefix}: 'status_id' must be a non-empty string")
    elif not _looks_like_x_status_id(status_id):
        warnings.append(f"{prefix}: non-standard status_id format '{status_id}'")

    handle = tweet.get("handle")
    if not _is_non_empty_string(handle):
        errors.append(f"{prefix}: 'handle' must be a non-empty string")
    elif isinstance(handle, str) and handle.startswith("@"):
        errors.append(f"{prefix}: 'handle' should not include @")

    text = tweet.get("text")
    if text is None:
        errors.append(f"{prefix}: missing 'text' field")
    elif not isinstance(text, str):
        errors.append(f"{prefix}: 'text' must be a string")
    elif not text.strip():
        warnings.append(f"{prefix}: empty text")

    role = tweet.get("role")
    if role not in TWEET_ROLES:
        errors.append(f"{prefix}: 'role' must be one of {sorted(TWEET_ROLES)}")

    parent_status_id = tweet.get("parent_status_id")
    if parent_status_id is not None and not _is_non_empty_string(parent_status_id):
        errors.append(
            f"{prefix}: 'parent_status_id' must be a non-empty string or null"
        )
    elif (
        isinstance(status_id, str)
        and isinstance(parent_status_id, str)
        and status_id == parent_status_id
    ):
        errors.append(f"{prefix}: 'parent_status_id' cannot equal 'status_id'")

    created_at = tweet.get("created_at")
    if not _is_iso_datetime(created_at):
        errors.append(f"{prefix}: 'created_at' must be an ISO 8601 timestamp")

    # Optional tweet fields
    if "source_url" in tweet and not isinstance(tweet["source_url"], str):
        errors.append(f"{prefix}: 'source_url' must be a string")

    if "display_name" in tweet and not isinstance(tweet["display_name"], str):
        errors.append(f"{prefix}: 'display_name' must be a string")

    if "user_id" in tweet and not isinstance(tweet["user_id"], str):
        errors.append(f"{prefix}: 'user_id' must be a string")

    if "verified" in tweet and not isinstance(tweet["verified"], bool):
        errors.append(f"{prefix}: 'verified' must be a boolean")

    for int_field in ("follower_count", "following_count", "tweet_count"):
        if int_field in tweet and not _is_non_negative_int(tweet[int_field]):
            errors.append(f"{prefix}: '{int_field}' must be a non-negative integer")

    if "bio" in tweet and not isinstance(tweet["bio"], str):
        errors.append(f"{prefix}: 'bio' must be a string")

    if "metrics" in tweet:
        _validate_metrics(
            tweet["metrics"],
            "metrics",
            line_num,
            record_id,
            errors,
            warnings,
        )

    return {
        "tweet_idx": tweet_idx,
        "status_id": status_id,
        "parent_status_id": parent_status_id,
        "role": role,
    }


def check_integrity(path: Path) -> tuple[list[str], list[str]]:
    """Run all integrity checks. Returns (errors, warnings)."""

    errors: list[str] = []
    warnings: list[str] = []

    id_counts: defaultdict[str, list[int]] = defaultdict(list)

    try:
        with path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    warnings.append(f"Line {line_num}: Empty line")
                    continue

                # Check 1: valid JSON
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue

                if not isinstance(obj, dict):
                    errors.append(
                        f"Line {line_num}: Record must be a JSON object (got {type(obj).__name__})"
                    )
                    continue

                # Required fields
                id_ = obj.get("id")
                platform = obj.get("platform")
                collected_at = obj.get("collected_at")
                labels = obj.get("labels")
                tweets = obj.get("tweets")

                # Check 2: required fields and basic types
                if not _is_non_empty_string(id_):
                    errors.append(
                        f"Line {line_num} (id={id_}): 'id' must be a non-empty string"
                    )
                else:
                    id_counts[id_.strip()].append(line_num)

                if platform != "x":
                    errors.append(
                        f"Line {line_num} (id={id_}): 'platform' must be 'x' for replies dataset"
                    )

                if not _is_iso_datetime(collected_at):
                    errors.append(
                        f"Line {line_num} (id={id_}): Missing or invalid 'collected_at' (ISO 8601 required)"
                    )

                if labels is None:
                    errors.append(f"Line {line_num} (id={id_}): Missing 'labels' field")
                elif not isinstance(labels, list) or not labels:
                    errors.append(
                        f"Line {line_num} (id={id_}): 'labels' must be a non-empty list"
                    )

                if tweets is None:
                    errors.append(f"Line {line_num} (id={id_}): Missing 'tweets' field")
                elif not isinstance(tweets, list) or not tweets:
                    errors.append(
                        f"Line {line_num} (id={id_}): 'tweets' must be a non-empty array"
                    )

                # Optional top-level fields
                if "notes" in obj and not isinstance(obj["notes"], str):
                    errors.append(
                        f"Line {line_num} (id={id_}): 'notes' must be a string"
                    )

                # Check 4: valid labels
                if labels is not None and isinstance(labels, list):
                    if len(labels) != len(set(labels)):
                        errors.append(
                            f"Line {line_num} (id={id_}): Duplicate labels (labels={labels})"
                        )

                    for label in labels:
                        if not _is_non_empty_string(label):
                            errors.append(
                                f"Line {line_num} (id={id_}): labels must contain non-empty strings (got {label!r})"
                            )
                            continue
                        if label not in VALID_LABELS:
                            errors.append(
                                f"Line {line_num} (id={id_}): Invalid label '{label}'"
                            )

                # Check tweets (schema + relationships)
                if tweets is not None and isinstance(tweets, list) and tweets:
                    seen_status_ids: set[str] = set()
                    tweet_nodes: list[dict[str, object]] = []
                    role_counts: defaultdict[str, int] = defaultdict(int)

                    for idx, tweet in enumerate(tweets, 1):
                        node = _validate_tweet(
                            tweet,
                            idx,
                            line_num,
                            id_,
                            errors,
                            warnings,
                        )
                        if node is None:
                            continue

                        status_id = node.get("status_id")
                        if isinstance(status_id, str):
                            status_id = status_id.strip()
                            if status_id in seen_status_ids:
                                errors.append(
                                    f"Line {line_num} (id={id_}): duplicate status_id '{status_id}' in tweets[]"
                                )
                            else:
                                seen_status_ids.add(status_id)

                        role = node.get("role")
                        if isinstance(role, str):
                            role_counts[role] += 1

                        tweet_nodes.append(node)

                    if role_counts.get("ai_reply", 0) < 1:
                        errors.append(
                            f"Line {line_num} (id={id_}): sample must include at least one tweet with role='ai_reply'"
                        )

                    for node in tweet_nodes:
                        tweet_idx = node.get("tweet_idx")
                        parent_status_id = node.get("parent_status_id")
                        if parent_status_id is None:
                            continue

                        if isinstance(parent_status_id, str):
                            parent_status_id = parent_status_id.strip()
                            if parent_status_id not in seen_status_ids:
                                errors.append(
                                    f"Line {line_num} (id={id_}): tweets[{tweet_idx}].parent_status_id='{parent_status_id}' does not match any tweets[].status_id"
                                )

    except FileNotFoundError:
        errors.append(f"File not found: {path}")
        return errors, warnings

    # Check 3b: report duplicates
    for id_, lines in id_counts.items():
        if len(lines) > 1:
            errors.append(f"Duplicate ID '{id_}' on lines: {lines}")

    return errors, warnings


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "path", nargs="?", default="data/replies.jsonl", help="Path to JSONL file"
    )
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    ap.add_argument("--quiet", action="store_true", help="Only output if errors found")
    args = ap.parse_args()

    path = Path(args.path)
    errors, warnings = check_integrity(path)

    exit_code = 0

    if errors:
        print(f"ERRORS ({len(errors)}):")
        for error in errors[:50]:
            print(f"  {error}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more errors")
        exit_code = 1

    if warnings:
        if args.strict:
            print(f"WARNINGS AS ERRORS ({len(warnings)}):")
            exit_code = 1
        elif not args.quiet:
            print(f"WARNINGS ({len(warnings)}):")

        if not args.quiet or args.strict:
            for warning in warnings[:30]:
                print(f"  {warning}")
            if len(warnings) > 30:
                print(f"  ... and {len(warnings) - 30} more warnings")

    if exit_code == 0 and not args.quiet:
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        label_counts: defaultdict[str, int] = defaultdict(int)
        role_counts: defaultdict[str, int] = defaultdict(int)
        total_tweets = 0

        for row in rows:
            for label in row.get("labels", []) or []:
                if isinstance(label, str):
                    label_counts[label] += 1

            tweets = row.get("tweets")
            if isinstance(tweets, list):
                total_tweets += len(tweets)
                for tweet in tweets:
                    if isinstance(tweet, dict):
                        role = tweet.get("role")
                        if isinstance(role, str):
                            role_counts[role] += 1

        print("All checks passed!")
        print("\nDataset stats:")
        print(f"  Total entries: {len(rows)}")
        print(f"  Total tweets: {total_tweets}")
        for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
            print(f"  tweets.role={role}: {count}")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  {label}: {count}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
