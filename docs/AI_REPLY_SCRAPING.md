# AI Reply Scraping Strategy

## Goal

Build a labeled dataset of AI-generated replies on X for training janitr's `ai_generated_reply` classifier.

## Core Principle

The ground truth dataset is built from **explicit human tags** — real people on X who publicly call out specific replies as AI-generated. This is not limited to any single account. Anyone who explicitly tags a reply as AI (by replying "AI reply", "bot", "ChatGPT reply", "blocked for AI reply", etc.) is a valid labeler.

The key distinction:

- **Ground truth** = a human explicitly tagged _that specific reply_ as AI
- **Inferred** = we suspect it's AI based on account signals, heuristics, or patterns, but nobody explicitly tagged it

## Data Sources & File Layout

### 1. Ground truth: `data/replies.jsonl` — Explicitly tagged AI replies

These are replies where a **human on X explicitly tagged the specific reply as AI-generated** by responding to it publicly. This is our highest-confidence labeled data.

**How it works:** People on X frequently call out AI-generated replies by responding directly to them with phrases like:

- "Blocked for AI reply"
- "AI reply"
- "Bot reply"
- "ChatGPT reply"
- "This is AI"
- "AI slop"
- Other clear callouts

Each callout gives us:

- The **specific reply** that was tagged (the positive example)
- The **parent post** it was replying to (context)
- A **human judgment** on that exact reply being AI-generated
- The **tagger's identity** (provenance)

**How to collect:**

1. Search X for known taggers by handle: `from:<handle> "AI reply"`, `from:<handle> "blocked for AI reply"`, etc.
2. Search X for common callout phrases without a `from:` filter to discover new taggers
3. For each result, the tweet the tagger is replying to is the AI-generated reply
4. Collect that reply + its parent post as a `replies.jsonl` entry
5. Set `labels: ["ai_generated_reply"]`
6. Set `notes` to reference the tagger and their exact callout

**Search method:** Always use X **global search** with the `from:` operator (e.g. `from:levelsio "AI reply"`). The search box on a user's profile page does NOT scope results to that user — it just redirects to a generic global search. So profile-page search is useless for this; always use the `from:` operator in global search.

**Key rule:** Only include replies in this file where we have an **explicit tag from a human on that specific reply**. No inference, no guessing.

#### Known taggers

These are accounts known to regularly and publicly flag AI replies. This list should grow over time — anyone who consistently calls out AI replies is a valuable source.

**@levelsio** (Pieter Levels, 800K+ followers)

- Has been systematically tagging AI reply bots since December 2022
- Uses the phrase "Blocked for AI reply" as his standard callout
- Key posts:
  - Dec 19, 2022: First major thread (1.7M views) — `/levelsio/status/1604841600416624642`
  - Jun 6, 2023: "not even trying anymore" — `/levelsio/status/1665954990769446914`
  - Sep 28, 2024: "Monthly update, GPT-4 level" — `/levelsio/status/1839992260341182647`
  - Jun 10, 2025: Submitted dataset to @TheGregYang (X engineer) — `/levelsio/status/1932471006507212989`
  - Nov 20, 2025: "best AI reply bot ever" — `/levelsio/status/1991458844405559550`
- Search: `from:levelsio "blocked for AI reply"` or `from:levelsio "AI reply"`

_Add more known taggers here as they're discovered. Look for accounts that:_

- _Have large followings (attract more AI reply bots)_
- _Regularly and publicly call out AI replies_
- _Use consistent phrasing that's searchable_

#### Discovering new taggers

Strategies to find more people who tag AI replies:

- Search X for common callout phrases without a `from:` filter
- Look at quote tweets of known AI reply bot accounts
- Check replies to viral posts from large accounts — callouts often happen there
- Monitor AI/tech community discussions about reply bots
- Check accounts that engage with levelsio's AI reply threads

### 2. Inferred/suspected: `data/replies_inferred.jsonl` — Our guesses

These are replies we **believe** are AI-generated based on heuristics, patterns, or account-level signals — but where no human explicitly tagged that specific reply.

Examples of what goes here:

- Other replies from accounts that have been flagged (but not the specific reply that was tagged)
- Replies detected by our own classifier or heuristics
- Replies from accounts on known bot lists
- Replies that look AI-generated based on style/patterns

**Important:** This file is useful for training and analysis but has lower label confidence than ground truth. Always keep it separate.

### 3. Account list: `data/flagged-ai-reply-bots.jsonl` — Flagged accounts

A list of X accounts that have been flagged as AI reply bots by any known tagger. One account per line.

Each entry should include:

- `handle`: the account's handle
- `flagged_date`: when they were flagged
- `source`: who flagged them and how (e.g. `"levelsio_blocked_for_ai_reply"`, `"community_report"`)
- `label`: `"ai_reply_bot"`
- `confidence`: `"high"` for explicit tags, `"medium"` for inferred

This is derived from ground truth collection but at the account level. Useful for:

- Scraping additional replies from these accounts (→ `replies_inferred.jsonl`)
- Account-level features (age, follower ratio, bio patterns)
- Cross-referencing with other bot lists

## Data Extraction Methods

See `docs/SYNDICATION_API.md` for full details on the syndication API.

### Syndication API (preferred for bulk collection)

X's public embed API at `cdn.syndication.twimg.com/tweet-result?id={status_id}&token=0` returns structured JSON for any public tweet — no auth, no API key, no login. When fetching a reply, the response includes a nested `parent` object with the full parent tweet. This means fetching the evidence tweet gives you 2 of 3 tweets in a thread chain for free.

**Pipeline:** Google search (discover IDs) → syndication API (extract data) → assemble threads

### Browser scraping (fallback)

For cases where the syndication API doesn't return enough data (e.g. need to browse reply trees, discover threads visually), use the OpenClaw browser tool against X directly.

## Scraping Workflow

### Step 1: Collect ground truth (explicitly tagged replies)

```
For each known tagger and search phrase:
  1. Search X for their callout posts (Google site:x.com or X global search)
  2. Fetch the evidence tweet via syndication API
  3. Extract the AI reply from the `parent` field
  4. Fetch the original post via syndication using in_reply_to_status_id_str
  5. Assemble into thread format and write to data/replies.jsonl
```

### Step 2: Build account list from ground truth

```
From replies.jsonl:
  1. Extract unique reply author handles
  2. Write to data/flagged-ai-reply-bots.jsonl with source attribution
```

### Step 3: Collect additional replies from flagged accounts

```
For each account in data/flagged-ai-reply-bots.jsonl:
  1. Scrape their recent replies
  2. Include the parent post for each reply
  3. Write to data/replies_inferred.jsonl
  4. Label as ["ai_generated_reply"] with notes indicating inference source
```

### Step 4: Collect negative examples (human replies)

```
From attractor accounts (data/accounts.jsonl):
  1. Scrape reply threads
  2. Filter out known bot accounts
  3. Write human replies to data/replies.jsonl with labels: ["clean"]
```

## Schema

All reply files use the schema defined in `docs/REPLIES_DATA_MODEL.md`.

Key fields for provenance:

- `labels`: What we're labeling this as
- `notes`: **How we know** — must reference the explicit tag or state the inference method
- `source_url`: Link to the specific reply

## Quality Rules

1. **Ground truth file (`replies.jsonl`) must only contain explicitly tagged replies** — no exceptions
2. **Every entry needs provenance** in `notes` — who tagged it, how, when
3. **Inferred file (`replies_inferred.jsonl`) must state the inference method** in `notes`
4. **Never mix ground truth and inferred data** — keep them in separate files
5. **Collect context** — always include parent post text and author metadata
6. **Dedup by reply ID** — a reply appears in only one file (ground truth takes priority)
