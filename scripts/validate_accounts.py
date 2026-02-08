#!/usr/bin/env python3
"""Validate data/accounts.jsonl against the account schema.

Usage:
    python scripts/validate_accounts.py [path]

Defaults to data/accounts.jsonl if no path given.
Exit code 0 = valid, 1 = errors found.
"""

import json
import sys
from pathlib import Path

VALID_CATEGORIES = {
    "clean",
    "crypto",
    "scam_hub",
    "scam_reply",
    "bot",
    "researcher",
    "media",
    "influencer",
}

VALID_PRIORITIES = {"high", "medium", "low"}

# Load valid labels from LABELS.md canonical list (hardcoded subset for speed)
# Keep in sync with docs/LABELS.md
VALID_LABELS = {
    # base
    "clean",
    # security & fraud
    "scam",
    "phishing",
    "malware",
    "impersonation",
    "fake_support",
    "recovery_scam",
    "job_scam",
    "romance_scam",
    "account_compromise",
    # spam & manipulation
    "spam",
    "reply_spam",
    "dm_spam",
    "promo",
    "affiliate",
    "lead_gen",
    "engagement_bait",
    "follow_train",
    "giveaway",
    "platform_manipulation",
    "astroturf",
    "bot",
    "ai_generated",
    "ai_generated_reply",
    "ai_slop",
    "content_farm",
    "copypasta",
    "stolen_content",
    "clickbait",
    "low_effort",
    "vaguepost",
    "ragebait",
    # information integrity
    "misinformation",
    "civic_misinfo",
    "manipulated_media",
    "conspiracy",
    "pseudoscience",
    # safety & sensitive
    "hate",
    "harassment",
    "threat_violence",
    "violent_extremism",
    "graphic_violence",
    "self_harm",
    "adult_nudity",
    "nonconsensual_nudity",
    "child_exploitation",
    "illegal_goods",
    "privacy_doxxing",
    "profanity",
    # topics
    "topic_news",
    "topic_world_news",
    "topic_local_news",
    "topic_war_conflict",
    "topic_crime_truecrime",
    "topic_disasters_tragedy",
    "topic_law_courts",
    "topic_environment_climate",
    "topic_social_issues",
    "topic_politics",
    "topic_elections",
    "topic_finance",
    "topic_investing",
    "topic_personal_finance",
    "topic_crypto",
    "topic_real_estate",
    "topic_shopping_deals",
    "topic_marketing_advertising",
    "topic_gambling",
    "topic_technology",
    "topic_ai",
    "topic_cybersecurity",
    "topic_programming_dev",
    "topic_startups_vc",
    "topic_consumer_electronics",
    "topic_entertainment",
    "topic_tv_movies",
    "topic_music",
    "topic_books",
    "topic_anime_manga",
    "topic_gaming",
    "topic_esports",
    "topic_celebrity",
    "topic_celebrity_gossip",
    "topic_comedy_memes",
    "topic_health",
    "topic_nutrition_diet",
    "topic_fitness",
    "topic_mental_health",
    "topic_beauty_fashion",
    "topic_food_drink",
    "topic_travel",
    "topic_home_garden",
    "topic_family_parenting",
    "topic_relationships_dating",
    "topic_sports",
    "topic_religion",
    "topic_adult_services",
    "topic_language_other",
    # special
    "spoiler",
}


def validate_record(record: dict, line_num: int) -> list[str]:
    """Validate a single account record. Returns list of error strings."""
    errors = []

    # Required fields
    if "handle" not in record:
        errors.append(f"line {line_num}: missing required field 'handle'")
    elif not isinstance(record["handle"], str) or not record["handle"].strip():
        errors.append(f"line {line_num}: 'handle' must be a non-empty string")
    elif record["handle"].startswith("@"):
        errors.append(
            f"line {line_num}: 'handle' should not include @ prefix "
            f"(got '{record['handle']}')"
        )

    if "category" not in record:
        errors.append(f"line {line_num}: missing required field 'category'")
    elif record["category"] not in VALID_CATEGORIES:
        errors.append(
            f"line {line_num}: invalid category '{record['category']}' "
            f"(valid: {sorted(VALID_CATEGORIES)})"
        )

    # Optional field types
    handle = record.get("handle", f"<line {line_num}>")

    if "user_id" in record and not isinstance(record["user_id"], str):
        errors.append(f"line {line_num} ({handle}): 'user_id' must be a string")

    if "display_name" in record and not isinstance(record["display_name"], str):
        errors.append(f"line {line_num} ({handle}): 'display_name' must be a string")

    if "bio" in record and not isinstance(record["bio"], str):
        errors.append(f"line {line_num} ({handle}): 'bio' must be a string")

    for count_field in ("follower_count", "following_count"):
        if count_field in record:
            if not isinstance(record[count_field], int) or record[count_field] < 0:
                errors.append(
                    f"line {line_num} ({handle}): '{count_field}' must be "
                    f"a non-negative integer"
                )

    if "verified" in record and not isinstance(record["verified"], bool):
        errors.append(f"line {line_num} ({handle}): 'verified' must be a boolean")

    if "suspended" in record and not isinstance(record["suspended"], bool):
        errors.append(f"line {line_num} ({handle}): 'suspended' must be a boolean")

    if "scrape_priority" in record:
        if record["scrape_priority"] not in VALID_PRIORITIES:
            errors.append(
                f"line {line_num} ({handle}): invalid scrape_priority "
                f"'{record['scrape_priority']}' (valid: {sorted(VALID_PRIORITIES)})"
            )

    if "labels" in record:
        if not isinstance(record["labels"], list):
            errors.append(f"line {line_num} ({handle}): 'labels' must be an array")
        else:
            for label in record["labels"]:
                if label not in VALID_LABELS:
                    errors.append(
                        f"line {line_num} ({handle}): unknown label '{label}'"
                    )

    if "notes" in record and not isinstance(record["notes"], str):
        errors.append(f"line {line_num} ({handle}): 'notes' must be a string")

    if "url" in record and not isinstance(record["url"], str):
        errors.append(f"line {line_num} ({handle}): 'url' must be a string")

    # ISO 8601 date fields (basic check)
    for date_field in ("created_at", "collected_at"):
        if date_field in record:
            val = record[date_field]
            if not isinstance(val, str) or len(val) < 10:
                errors.append(
                    f"line {line_num} ({handle}): '{date_field}' must be "
                    f"an ISO 8601 string"
                )

    return errors


def validate_file(path: Path) -> tuple[int, int, list[str]]:
    """Validate a JSONL file. Returns (total, valid, errors)."""
    all_errors = []
    total = 0
    valid = 0
    handles_seen = set()

    if not path.exists():
        return 0, 0, [f"File not found: {path}"]

    if path.stat().st_size == 0:
        print(f"  {path}: empty file (no records)")
        return 0, 0, []

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"line {line_num}: invalid JSON — {e}")
                continue

            if not isinstance(record, dict):
                all_errors.append(f"line {line_num}: record must be a JSON object")
                continue

            errors = validate_record(record, line_num)

            # Check for duplicate handles
            handle = record.get("handle", "").lower()
            if handle:
                if handle in handles_seen:
                    errors.append(
                        f"line {line_num} ({record.get('handle')}): duplicate handle"
                    )
                handles_seen.add(handle)

            if errors:
                all_errors.extend(errors)
            else:
                valid += 1

    return total, valid, all_errors


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/accounts.jsonl")

    print(f"Validating {path}...")
    total, valid, errors = validate_file(path)

    if errors:
        print(f"\n{len(errors)} error(s) found:\n")
        for error in errors:
            print(f"  ✗ {error}")
        print(f"\n{valid}/{total} records valid")
        sys.exit(1)
    else:
        print(f"  ✓ {total} records, all valid")
        sys.exit(0)


if __name__ == "__main__":
    main()
