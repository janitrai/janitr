#!/usr/bin/env python3
"""Fetch other-tagger callout thread data via syndication API.

Reads data/other-taggers-deep-callouts.jsonl (666 callouts) and fetches
parent tweets (the actual AI replies) to build ground truth data.

Saves to data/other-taggers-deep-fetched.jsonl
"""

import json
import time
import urllib.request
import sys
from pathlib import Path

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
        print(f"  HTTP {e.code} fetching {tweet_id}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ERROR fetching {tweet_id}: {e}", file=sys.stderr)
        return None

def extract_tweet_data(tweet: dict) -> dict:
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
    callouts_file = data_dir / "other-taggers-deep-callouts.jsonl"
    output_file = data_dir / "other-taggers-deep-fetched.jsonl"
    
    # Load callouts
    callouts = []
    with open(callouts_file) as f:
        for line in f:
            line = line.strip()
            if line:
                callouts.append(json.loads(line))
    
    print(f"Loaded {len(callouts)} callout tweets to fetch")
    
    # Check for existing progress (resume support)
    fetched_ids = set()
    existing_results = []
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    callout_id = obj.get("callout", {}).get("tweet_id", "")
                    if callout_id:
                        fetched_ids.add(callout_id)
                        existing_results.append(obj)
        print(f"Resuming: {len(fetched_ids)} already fetched")
    
    remaining = [c for c in callouts if str(c["callout_tweet_id"]) not in fetched_ids]
    print(f"Remaining to fetch: {len(remaining)}")
    
    if not remaining:
        print("Nothing to do!")
        return
    
    # Open output in append mode
    with open(output_file, "a") as out:
        for i, callout in enumerate(remaining):
            tweet_id = callout["callout_tweet_id"]
            handle = callout["flagged_handle"]
            tagger = callout["tagger"]
            
            print(f"[{i+1}/{len(remaining)}] @{tagger} → @{handle} (tweet {tweet_id})...", end=" ")
            
            tweet_data = fetch_tweet(str(tweet_id))
            if not tweet_data:
                print("No parent (deleted)")
                result = {
                    "callout": {"tweet_id": str(tweet_id)},
                    "ai_reply": None,
                    "original_post": None,
                    "flagged_handle": handle,
                    "tagger": tagger,
                    "source": "x_search_other_taggers",
                }
            else:
                callout_extracted = extract_tweet_data(tweet_data)
                
                parent = tweet_data.get("parent")
                parent_extracted = extract_tweet_data(parent) if parent else None
                
                grandparent = parent.get("parent") if parent else None
                grandparent_extracted = extract_tweet_data(grandparent) if grandparent else None
                
                if parent_extracted:
                    print(f"@{parent_extracted['handle']}: {parent_extracted['text'][:60]}...")
                else:
                    print("No parent (deleted)")
                
                result = {
                    "callout": callout_extracted,
                    "ai_reply": parent_extracted,
                    "original_post": grandparent_extracted,
                    "flagged_handle": handle,
                    "tagger": tagger,
                    "source": "x_search_other_taggers",
                }
            
            out.write(json.dumps(result) + "\n")
            out.flush()
            
            # Rate limit: 0.5s between requests
            time.sleep(0.5)
    
    # Final summary
    all_results = existing_results.copy()
    with open(output_file) as f:
        all_results = [json.loads(line) for line in f if line.strip()]
    
    with_parent = sum(1 for r in all_results if r.get("ai_reply"))
    total = len(all_results)
    print(f"\nDone! Total: {total} results")
    print(f"  With AI reply (parent): {with_parent}/{total} ({100*with_parent//total}%)")
    print(f"  Deleted/missing parent: {total - with_parent}/{total}")

if __name__ == "__main__":
    main()
