#!/usr/bin/env python3
"""Unify other-tagger fetched data into replies.jsonl format.

Reads:
  - data/replies.jsonl (existing, levelsio)
  - data/other-taggers-deep-fetched.jsonl (537 with AI reply text)
  - data/other-taggers-fetched.jsonl (19 with AI reply)

Writes:
  - data/replies.jsonl (unified, deduplicated by AI reply tweet ID)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPLIES = REPO / "data" / "replies.jsonl"
DEEP = REPO / "data" / "other-taggers-deep-fetched.jsonl"
SMALL = REPO / "data" / "other-taggers-fetched.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_reply_id(row: dict) -> str:
    """Extract the AI reply tweet ID from a fetched row."""
    ar = row.get("ai_reply") or {}
    return ar.get("tweet_id") or ar.get("status_id") or ""


def get_callout_id(row: dict) -> str:
    """Extract the callout tweet ID from a fetched row."""
    co = row.get("callout") or {}
    return co.get("tweet_id") or co.get("status_id") or ""


def convert_fetched_to_reply(row: dict) -> dict | None:
    """Convert a fetched row to replies.jsonl format."""
    ai_reply = row.get("ai_reply") or {}
    callout = row.get("callout") or {}
    original = row.get("original_post")

    # Need ai_reply with text
    reply_text = ai_reply.get("text", "")
    if not reply_text:
        return None

    reply_id = ai_reply.get("tweet_id") or ai_reply.get("status_id") or ""
    if not reply_id:
        return None

    tagger = row.get("tagger", "unknown")
    flagged = row.get("flagged_handle", "unknown")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    tweets = []

    # Original post (if available)
    if original and original.get("text"):
        orig_id = original.get("tweet_id") or original.get("status_id") or ""
        tweets.append(
            {
                "status_id": str(orig_id),
                "handle": original.get("handle", ""),
                "text": original.get("text", ""),
                "role": "original_post",
                "parent_status_id": None,
                "created_at": original.get("created_at", ""),
                "verified": original.get("verified", False),
            }
        )
        if original.get("display_name"):
            tweets[-1]["display_name"] = original["display_name"]

    # AI reply
    parent_id = ai_reply.get("in_reply_to_status_id") or None
    # If we have original post, use its ID as parent
    if tweets and not parent_id:
        parent_id = tweets[0]["status_id"]
    tweets.append(
        {
            "status_id": str(reply_id),
            "handle": ai_reply.get("handle", flagged),
            "text": reply_text,
            "role": "ai_reply",
            "parent_status_id": str(parent_id) if parent_id else None,
            "created_at": ai_reply.get("created_at", ""),
            "verified": ai_reply.get("verified", False),
        }
    )
    if ai_reply.get("display_name"):
        tweets[-1]["display_name"] = ai_reply["display_name"]

    # Callout/evidence
    callout_id = callout.get("tweet_id") or callout.get("status_id") or ""
    tweets.append(
        {
            "status_id": str(callout_id),
            "handle": callout.get("handle", tagger),
            "text": callout.get("text", ""),
            "role": "evidence",
            "parent_status_id": str(reply_id),
            "created_at": callout.get("created_at", ""),
            "verified": callout.get("verified", False),
        }
    )
    if callout.get("display_name"):
        tweets[-1]["display_name"] = callout["display_name"]

    return {
        "id": f"sample-{reply_id}",
        "platform": "x",
        "collected_at": now,
        "labels": ["ai_generated_reply"],
        "notes": f"Flagged by @{tagger} as AI reply",
        "tweets": tweets,
    }


def main():
    # Load existing replies
    existing = load_jsonl(REPLIES)
    print(f"Existing replies.jsonl: {len(existing)} samples")

    # Collect existing AI reply IDs to deduplicate
    seen_reply_ids = set()
    for row in existing:
        for tw in row.get("tweets", []):
            if tw.get("role") == "ai_reply":
                seen_reply_ids.add(tw["status_id"])

    print(f"Existing unique AI reply IDs: {len(seen_reply_ids)}")

    # Load and convert other-tagger data
    new_samples = []
    dupes = 0

    for path, label in [(DEEP, "deep"), (SMALL, "small")]:
        if not path.exists():
            print(f"  {label}: file not found, skipping")
            continue
        rows = load_jsonl(path)
        converted = 0
        skipped_no_text = 0
        skipped_dupe = 0
        for row in rows:
            reply_id = get_reply_id(row)
            if reply_id in seen_reply_ids:
                skipped_dupe += 1
                dupes += 1
                continue
            sample = convert_fetched_to_reply(row)
            if sample is None:
                skipped_no_text += 1
                continue
            seen_reply_ids.add(reply_id)
            new_samples.append(sample)
            converted += 1
        print(
            f"  {label} ({path.name}): {len(rows)} rows → {converted} converted, {skipped_no_text} no text, {skipped_dupe} dupes"
        )

    print(f"\nNew samples to add: {len(new_samples)}")
    print(f"Duplicates skipped: {dupes}")
    print(f"Final total: {len(existing) + len(new_samples)}")

    # Write unified file
    all_samples = existing + new_samples
    with open(REPLIES, "w") as f:
        for row in all_samples:
            f.write(json.dumps(row) + "\n")

    print(f"\nWritten {len(all_samples)} samples to {REPLIES}")


if __name__ == "__main__":
    main()
