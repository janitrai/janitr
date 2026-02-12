# Replies Data Model (Thread-Based)

This document defines the JSONL schema for the reply dataset used for AI-generated reply detection on X.

- Dataset file: `data/replies.jsonl`
- Separate from: `data/sample.jsonl`
- One JSON object per line

The unit of labeling is the **AI reply**, but each sample stores a **thread of tweets** (a small conversation graph) so reviewers and models can reason about context and evidence.

## Design Principles

1. Preserve the full granular label taxonomy from `docs/LABELS.md` at the data layer (no label collapsing in JSONL).
2. Store the thread structure explicitly (parent-child links), not flattened `parent_text`/`thread_context` fields.
3. Keep the schema small and on-device friendly: tweet text is required; metadata is optional.

## Top-Level Schema (One JSONL Record)

Required fields:

| Field          | Type              | Notes                                              |
| -------------- | ----------------- | -------------------------------------------------- |
| `id`           | string            | Unique sample ID (dataset-level).                  |
| `platform`     | string            | Must be `x`.                                       |
| `collected_at` | string (ISO 8601) | Time this sample was collected.                    |
| `labels`       | string[]          | Non-empty; values must come from `docs/LABELS.md`. |
| `tweets`       | object[]          | Non-empty array of tweet objects (schema below).   |

Optional fields:

| Field   | Type   | Notes                                 |
| ------- | ------ | ------------------------------------- |
| `notes` | string | Labeler rationale or collection notes |

## Tweet Object Schema

Each item in `tweets` is a tweet-like object with a role and a parent pointer.

Required fields:

| Field              | Type              | Notes                                                                      |
| ------------------ | ----------------- | -------------------------------------------------------------------------- |
| `status_id`        | string            | Tweet/status ID.                                                           |
| `handle`           | string            | Username without `@`.                                                      |
| `text`             | string            | Tweet text (raw, untruncated).                                             |
| `role`             | enum              | One of: `original_post`, `ai_reply`, `evidence`, `other_reply`, `context`. |
| `parent_status_id` | string \| null    | The `status_id` this tweet is replying to. `null` indicates a root node.   |
| `created_at`       | string (ISO 8601) | Tweet creation timestamp.                                                  |

Optional fields:

| Field             | Type    | Notes                                |
| ----------------- | ------- | ------------------------------------ |
| `source_url`      | string  | Canonical URL of the tweet.          |
| `display_name`    | string  | Display name.                        |
| `user_id`         | string  | Numeric user ID when available.      |
| `verified`        | boolean | Verification badge at scrape time.   |
| `follower_count`  | integer | Non-negative.                        |
| `following_count` | integer | Non-negative.                        |
| `bio`             | string  | Profile bio text.                    |
| `tweet_count`     | integer | Non-negative.                        |
| `metrics`         | object  | Optional metrics object (see below). |

If present, `metrics` is an object with optional integer fields:

- `like_count`
- `reply_count`
- `repost_count`
- `quote_count`
- `view_count`

## Roles

Roles describe why a tweet is included in the sample:

- `original_post`: The tweet that attracted the AI reply (the post being replied to).
- `ai_reply`: The AI-generated reply being labeled (the primary target for labeling).
- `evidence`: A human tagger explicitly calling out the AI reply (e.g. "Blocked for AI reply").
- `other_reply`: Any other reply in the local thread (human replies, other bots, etc.).
- `context`: Ancestor tweets providing additional context for the conversation.

## Thread Structure and Relationships

- The `tweets` array can represent a non-linear conversation. Do not assume a linear chain.
- Parent-child links are defined by `parent_status_id`.
- `parent_status_id` must either be `null` (root) or reference another tweet's `status_id` in the same sample.
- A "minimum ground truth" sample is typically 3 tweets:
  - one `original_post`
  - one `ai_reply`
  - one `evidence`

## Example

```json
{
  "id": "sample-2021695412122661028",
  "platform": "x",
  "collected_at": "2026-02-12T10:00:00Z",
  "labels": ["ai_generated_reply"],
  "notes": "Flagged by @levelsio",
  "tweets": [
    {
      "status_id": "2021693766793318833",
      "handle": "levelsio",
      "text": "New Brazilian buffet tour...",
      "role": "original_post",
      "parent_status_id": null,
      "created_at": "2026-02-11T21:00:00Z",
      "verified": true,
      "follower_count": 821200
    },
    {
      "status_id": "2021695412122661028",
      "handle": "BlockShaolin",
      "text": "The church buffet isnt about religion - its a social credit testing ground...",
      "role": "ai_reply",
      "parent_status_id": "2021693766793318833",
      "created_at": "2026-02-11T21:30:00Z",
      "verified": true,
      "follower_count": 50
    },
    {
      "status_id": "2021699240393609717",
      "handle": "levelsio",
      "text": "Blocked for AI reply",
      "role": "evidence",
      "parent_status_id": "2021695412122661028",
      "created_at": "2026-02-11T21:45:00Z",
      "verified": true,
      "follower_count": 821200
    }
  ]
}
```

## File Layout

| File                               | Contents                                                                                                                          | Label confidence    |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `data/replies.jsonl`               | Replies **explicitly tagged** as AI by humans on X (e.g. someone replying "AI reply" / "blocked for AI reply" to a specific post) | High — ground truth |
| `data/replies_inferred.jsonl`      | Replies we **suspect** are AI based on heuristics, account signals, or classifier output                                          | Lower — inferred    |
| `data/flagged-ai-reply-bots.jsonl` | Account-level list of flagged AI reply bot handles (from any tagger)                                                              | Account-level only  |

**Rule:** Never mix ground truth and inferred data. See `docs/AI_REPLY_SCRAPING.md` for the full scraping strategy.

## Validation

Use the dedicated integrity checker:

```bash
python scripts/check_reply_integrity.py data/replies.jsonl
```

Strict mode (warnings fail validation):

```bash
python scripts/check_reply_integrity.py data/replies.jsonl --strict
```
