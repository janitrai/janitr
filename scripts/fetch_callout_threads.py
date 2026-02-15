#!/usr/bin/env python3
"""Fetch levelsio 'Blocked for AI reply' callout tweets via syndication API.

For each callout tweet, gets the parent tweet (the actual AI reply) and saves
structured data to data/levelsio-callouts-fetched.jsonl.
"""

import json
import time
import urllib.request
import sys
from pathlib import Path

SYNDICATION_URL = "https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&token=0"


def fetch_tweet(tweet_id: str) -> dict | None:
    """Fetch a single tweet via syndication API."""
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


def extract_tweet_data(tweet: dict) -> dict:
    """Extract relevant fields from a syndication tweet response."""
    user = tweet.get("user", {})
    return {
        "tweet_id": tweet.get("id_str", ""),
        "text": tweet.get("text", ""),
        "handle": user.get("screen_name", ""),
        "display_name": user.get("name", ""),
        "verified": user.get("is_blue_verified", False),
        "created_at": tweet.get("created_at", ""),
        "likes": tweet.get("favorite_count", 0),
        "replies": tweet.get("conversation_count", 0),
        "lang": tweet.get("lang", ""),
        "in_reply_to_status_id": tweet.get("in_reply_to_status_id_str", ""),
        "in_reply_to_screen_name": tweet.get("in_reply_to_screen_name", ""),
    }


def main():
    data_dir = Path(__file__).parent.parent / "data"
    callouts_file = data_dir / "levelsio-callouts-raw.jsonl"
    output_file = data_dir / "levelsio-callouts-fetched.jsonl"

    # Load callout tweet IDs
    callouts = []
    with open(callouts_file) as f:
        for line in f:
            line = line.strip()
            if line:
                callouts.append(json.loads(line))

    print(f"Loaded {len(callouts)} callout tweets")

    results = []
    for i, callout in enumerate(callouts):
        tweet_id = callout["callout_tweet_id"]
        handle = callout["flagged_handle"]
        print(
            f"[{i + 1}/{len(callouts)}] Fetching callout tweet {tweet_id} (flagged: @{handle})..."
        )

        tweet_data = fetch_tweet(tweet_id)
        if not tweet_data:
            print(f"  SKIP: could not fetch")
            continue

        callout_extracted = extract_tweet_data(tweet_data)

        # Extract parent tweet (the actual AI reply)
        parent = tweet_data.get("parent")
        parent_extracted = None
        if parent:
            parent_extracted = extract_tweet_data(parent)
            print(
                f"  Parent: @{parent_extracted['handle']}: {parent_extracted['text'][:80]}..."
            )
        else:
            print(f"  No parent tweet in response")

        # Also check if there's a grandparent (the original post the AI replied to)
        grandparent = parent.get("parent") if parent else None
        grandparent_extracted = None
        if grandparent:
            grandparent_extracted = extract_tweet_data(grandparent)
            print(
                f"  Grandparent: @{grandparent_extracted['handle']}: {grandparent_extracted['text'][:80]}..."
            )

        result = {
            "callout": callout_extracted,
            "ai_reply": parent_extracted,
            "original_post": grandparent_extracted,
            "flagged_handle": handle,
            "tagger": "levelsio",
            "source": "x_search",
        }
        results.append(result)

        # Be polite: 0.5s between requests
        time.sleep(0.5)

    # Write results
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone! Wrote {len(results)} results to {output_file}")

    # Summary
    with_parent = sum(1 for r in results if r["ai_reply"])
    with_grandparent = sum(1 for r in results if r["original_post"])
    print(f"  With AI reply (parent): {with_parent}/{len(results)}")
    print(f"  With original post (grandparent): {with_grandparent}/{len(results)}")


if __name__ == "__main__":
    main()
