#!/usr/bin/env python3
"""Deep scrape levelsio 'blocked for AI reply' callouts using browser + syndication API.

Uses the openclaw browser to search X with date ranges, extracts callout tweet IDs,
then batch-fetches via syndication API.

Run phases:
1. Browser scrape: collect callout tweet IDs from X search (this script outputs to stdout)
2. Syndication fetch: batch-fetch parent tweets
3. Build replies.jsonl entries
"""

import json
import sys
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

SYNDICATION_URL = "https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&token=0"
DATA_DIR = Path(__file__).parent.parent / "data"

def fetch_tweet(tweet_id: str) -> dict | None:
    url = SYNDICATION_URL.format(tweet_id=tweet_id)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR {tweet_id}: {e}", file=sys.stderr)
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

def batch_fetch_and_append(callouts_file: str):
    """Read callouts file, fetch via syndication, append new entries to replies.jsonl."""
    replies_file = DATA_DIR / "replies.jsonl"
    
    # Load existing
    existing_ids = set()
    existing_handles = set()
    with open(replies_file) as f:
        for line in f:
            d = json.loads(line.strip())
            existing_ids.add(d["id"])
            for t in d.get("tweets", []):
                if t.get("role") == "ai_reply":
                    existing_handles.add(t["handle"].lower())
    
    callouts = []
    with open(callouts_file) as f:
        for line in f:
            if line.strip():
                callouts.append(json.loads(line))
    
    # Filter out already-processed
    new_callouts = [c for c in callouts if c["flagged_handle"].lower() not in existing_handles]
    print(f"Total callouts: {len(callouts)}, already have: {len(callouts)-len(new_callouts)}, to fetch: {len(new_callouts)}")
    
    new_entries = []
    new_bots = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    for i, c in enumerate(new_callouts):
        tid = c["callout_tweet_id"]
        handle = c["flagged_handle"]
        tagger = c.get("tagger", "levelsio")
        print(f"[{i+1}/{len(new_callouts)}] @{handle} (tweet {tid})...")
        
        data = fetch_tweet(tid)
        if not data:
            continue
        
        callout_ext = extract(data)
        parent = data.get("parent")
        ai_reply_ext = None
        original_ext = None
        
        if parent:
            ai_reply_ext = extract(parent)
            print(f"  Reply: @{ai_reply_ext['handle']}: {ai_reply_ext['text'][:70]}...")
            
            # Try to get grandparent
            gp = parent.get("parent")
            if gp:
                original_ext = extract(gp)
            else:
                gp_data = fetch_tweet(ai_reply_ext["status_id"])
                time.sleep(0.3)
                if gp_data and gp_data.get("parent"):
                    original_ext = extract(gp_data["parent"])
        else:
            print(f"  No parent (deleted)")
        
        if ai_reply_ext:
            tweets = []
            if original_ext:
                tweets.append({**original_ext, "role": "original_post"})
            tweets.append({**ai_reply_ext, "role": "ai_reply"})
            tweets.append({**callout_ext, "role": "evidence"})
            
            entry = {
                "id": f"sample-{ai_reply_ext['status_id']}",
                "platform": "x",
                "collected_at": now,
                "labels": ["ai_generated_reply"],
                "notes": f"Flagged by @{tagger} as 'Blocked for AI reply'",
                "tweets": tweets,
            }
            if entry["id"] not in existing_ids:
                new_entries.append(entry)
                existing_ids.add(entry["id"])
        
        new_bots.append(handle.lower())
        time.sleep(0.4)
    
    # Append to replies.jsonl
    if new_entries:
        with open(replies_file, "a") as f:
            for e in new_entries:
                f.write(json.dumps(e) + "\n")
        print(f"\nAppended {len(new_entries)} new entries to replies.jsonl")
    
    # Append to bot handles
    bots_file = DATA_DIR / "levelsio-flagged-ai-reply-bots.jsonl"
    existing_bots = set()
    with open(bots_file) as f:
        for line in f:
            existing_bots.add(json.loads(line)["handle"].lower())
    
    really_new = [b for b in new_bots if b not in existing_bots]
    if really_new:
        with open(bots_file, "a") as f:
            for h in really_new:
                f.write(json.dumps({"handle": h, "source": "deep_scrape_2026-02-13", "tagger": "levelsio"}) + "\n")
        print(f"Added {len(really_new)} new bot handles")
    
    total = sum(1 for _ in open(replies_file))
    total_bots = sum(1 for _ in open(bots_file))
    print(f"\nTotals: {total} replies, {total_bots} bot handles")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        batch_fetch_and_append(sys.argv[1])
    else:
        print("Usage: python scrape_levelsio_deep.py <callouts-file.jsonl>")
        print("Generate callouts file from browser scraping first.")
