#!/usr/bin/env python3
"""Fetch remaining levelsio callouts and convert to replies.jsonl format.

Reads /tmp/levelsio-remaining.jsonl (132 callouts not yet in replies.jsonl),
fetches via syndication API, and appends to data/replies.jsonl in the same
format as existing entries.
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
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"  HTTP {e.code}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return None

def tweet_to_entry(tweet_data: dict) -> dict:
    """Convert syndication tweet to replies.jsonl tweet entry."""
    user = tweet_data.get("user", {})
    return {
        "status_id": tweet_data.get("id_str", ""),
        "handle": user.get("screen_name", ""),
        "text": tweet_data.get("text", ""),
        "created_at": tweet_data.get("created_at", ""),
        "likes": tweet_data.get("favorite_count", 0),
        "lang": tweet_data.get("lang", ""),
    }

def main():
    data_dir = Path(__file__).parent.parent / "data"
    remaining_file = Path("/tmp/levelsio-remaining.jsonl")
    output_file = data_dir / "replies.jsonl"
    
    callouts = []
    with open(remaining_file) as f:
        for line in f:
            if line.strip():
                callouts.append(json.loads(line))
    
    print(f"Fetching {len(callouts)} remaining levelsio callouts")
    
    now = datetime.now(timezone.utc).isoformat()
    added = 0
    skipped = 0
    deleted = 0
    
    with open(output_file, "a") as out:
        for i, callout in enumerate(callouts):
            tweet_id = str(callout["callout_tweet_id"])
            handle = callout["flagged_handle"]
            
            print(f"[{i+1}/{len(callouts)}] @{handle} (callout {tweet_id})...", end=" ")
            
            tweet_data = fetch_tweet(tweet_id)
            if not tweet_data:
                print("callout deleted")
                deleted += 1
                time.sleep(0.5)
                continue
            
            callout_entry = tweet_to_entry(tweet_data)
            
            # Get parent (the AI reply)
            parent = tweet_data.get("parent")
            if not parent:
                print("no parent (AI reply deleted)")
                deleted += 1
                time.sleep(0.5)
                continue
            
            parent_entry = tweet_to_entry(parent)
            
            # Get grandparent (original post) if available
            grandparent = parent.get("parent")
            
            # Build the tweet thread
            tweets = []
            if grandparent:
                tweets.append(tweet_to_entry(grandparent))
            tweets.append(parent_entry)  # AI reply
            tweets.append(callout_entry)  # levelsio's callout
            
            sample = {
                "id": f"sample-{tweet_id}",
                "platform": "x",
                "collected_at": now,
                "labels": ["ai_generated_reply"],
                "notes": f"Flagged by @levelsio as 'Blocked for AI reply'",
                "tweets": tweets,
            }
            
            out.write(json.dumps(sample) + "\n")
            out.flush()
            added += 1
            
            print(f"@{parent_entry['handle']}: {parent_entry['text'][:60]}...")
            time.sleep(0.5)
    
    print(f"\nDone!")
    print(f"  Added to replies.jsonl: {added}")
    print(f"  Deleted/missing: {deleted}")
    print(f"  Total replies.jsonl: {sum(1 for _ in open(output_file))}")

if __name__ == "__main__":
    main()
