#!/usr/bin/env python3
"""Fetch callout threads from other taggers (non-levelsio) via syndication API.

For each callout tweet, gets the parent tweet (the actual AI reply) and
the grandparent (the original post). Saves to data/other-taggers-fetched.jsonl
and appends new entries to data/replies.jsonl.
"""

import json
import time
import urllib.request
import sys
from pathlib import Path
from datetime import datetime, timezone

SYNDICATION_URL = "https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&token=0"


def fetch_tweet(tweet_id: str) -> dict | None:
    url = SYNDICATION_URL.format(tweet_id=tweet_id)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR fetching {tweet_id}: {e}", file=sys.stderr)
        return None


def extract(tweet: dict) -> dict:
    user = tweet.get("user", {})
    return {
        "status_id": tweet.get("id_str", ""),
        "handle": user.get("screen_name", ""),
        "display_name": user.get("name", ""),
        "text": tweet.get("text", ""),
        "verified": user.get("is_blue_verified", False),
        "created_at": tweet.get("created_at", ""),
        "in_reply_to_status_id": tweet.get("in_reply_to_status_id_str"),
        "in_reply_to_screen_name": tweet.get("in_reply_to_screen_name"),
    }


def make_reply_entry(
    callout_data, ai_reply_data, original_data, flagged_handle, tagger
):
    """Build a replies.jsonl entry."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tweets = []

    if original_data:
        tweets.append(
            {
                "status_id": original_data["status_id"],
                "handle": original_data["handle"],
                "text": original_data["text"],
                "role": "original_post",
                "parent_status_id": original_data.get("in_reply_to_status_id"),
                "created_at": original_data["created_at"],
                "verified": original_data["verified"],
            }
        )

    if ai_reply_data:
        tweets.append(
            {
                "status_id": ai_reply_data["status_id"],
                "handle": ai_reply_data["handle"],
                "text": ai_reply_data["text"],
                "role": "ai_reply",
                "parent_status_id": ai_reply_data.get("in_reply_to_status_id"),
                "created_at": ai_reply_data["created_at"],
                "verified": ai_reply_data["verified"],
                "display_name": ai_reply_data.get("display_name", ""),
            }
        )

    tweets.append(
        {
            "status_id": callout_data["status_id"],
            "handle": callout_data["handle"],
            "text": callout_data["text"],
            "role": "evidence",
            "parent_status_id": callout_data.get("in_reply_to_status_id"),
            "created_at": callout_data["created_at"],
            "verified": callout_data["verified"],
        }
    )

    reply_id = (
        ai_reply_data["status_id"] if ai_reply_data else callout_data["status_id"]
    )
    return {
        "id": f"sample-{reply_id}",
        "platform": "x",
        "collected_at": now,
        "labels": ["ai_generated_reply"],
        "notes": f"Flagged by @{tagger} as 'Blocked for AI reply'",
        "tweets": tweets,
    }


def main():
    data_dir = Path(__file__).parent.parent / "data"
    callouts_file = data_dir / "other-taggers-callouts-raw.jsonl"
    output_file = data_dir / "other-taggers-fetched.jsonl"
    replies_file = data_dir / "replies.jsonl"

    # Load existing replies to avoid duplicates
    existing_ids = set()
    with open(replies_file) as f:
        for line in f:
            d = json.loads(line.strip())
            existing_ids.add(d["id"])

    callouts = []
    with open(callouts_file) as f:
        for line in f:
            if line.strip():
                callouts.append(json.loads(line))

    print(f"Loaded {len(callouts)} callouts, {len(existing_ids)} existing replies")

    results = []
    new_entries = []

    for i, c in enumerate(callouts):
        tid = c["callout_tweet_id"]
        handle = c["flagged_handle"]
        tagger = c["tagger"]
        print(f"[{i + 1}/{len(callouts)}] @{tagger} → @{handle} (tweet {tid})...")

        # Fetch callout tweet
        callout_tweet = fetch_tweet(tid)
        if not callout_tweet:
            print(f"  SKIP: could not fetch callout")
            continue

        callout_extracted = extract(callout_tweet)

        # Get parent (AI reply)
        parent = callout_tweet.get("parent")
        ai_reply_extracted = None
        original_extracted = None

        if parent:
            ai_reply_extracted = extract(parent)
            print(
                f"  AI reply: @{ai_reply_extracted['handle']}: {ai_reply_extracted['text'][:80]}..."
            )

            # Get grandparent (original post)
            grandparent = parent.get("parent")
            if grandparent:
                original_extracted = extract(grandparent)
                print(
                    f"  Original: @{original_extracted['handle']}: {original_extracted['text'][:60]}..."
                )
            else:
                # Try fetching the AI reply to get its parent
                ai_reply_full = fetch_tweet(ai_reply_extracted["status_id"])
                time.sleep(0.3)
                if ai_reply_full:
                    gp = ai_reply_full.get("parent")
                    if gp:
                        original_extracted = extract(gp)
                        print(
                            f"  Original (2nd fetch): @{original_extracted['handle']}: {original_extracted['text'][:60]}..."
                        )
        else:
            print(f"  No parent (deleted?)")

        result = {
            "callout": callout_extracted,
            "ai_reply": ai_reply_extracted,
            "original_post": original_extracted,
            "flagged_handle": handle,
            "tagger": tagger,
        }
        results.append(result)

        # Build replies.jsonl entry if we have the AI reply
        if ai_reply_extracted:
            entry = make_reply_entry(
                callout_extracted,
                ai_reply_extracted,
                original_extracted,
                handle,
                tagger,
            )
            if entry["id"] not in existing_ids:
                new_entries.append(entry)
                existing_ids.add(entry["id"])
                print(f"  → NEW entry: {entry['id']}")
            else:
                print(f"  → Already exists: {entry['id']}")

        time.sleep(0.5)

    # Write raw results
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} raw results to {output_file}")

    # Append new entries to replies.jsonl
    if new_entries:
        with open(replies_file, "a") as f:
            for e in new_entries:
                f.write(json.dumps(e) + "\n")
        print(f"Appended {len(new_entries)} new entries to {replies_file}")
    else:
        print("No new entries to append")

    # Summary
    with_parent = sum(1 for r in results if r["ai_reply"])
    print(f"\nSummary:")
    print(f"  Total callouts: {len(results)}")
    print(f"  With AI reply: {with_parent}")
    print(f"  New entries added: {len(new_entries)}")


if __name__ == "__main__":
    main()
