# Data Model

This document describes the schemas and storage patterns used in this project.

## Overview

We use a **two-layer storage approach**:

1. **Layer 1 (lossless)**: Raw API responses stored exactly as received
2. **Layer 2 (labeled)**: Lightweight labeled samples for ML training

## Data Collection Methods

### Browser/UI Scraping (primary method)

Most data is collected via OpenClaw's browser automation, scraping the X web UI.

**Limitations:**
- We only get fields visible in the DOM (no full API response)
- Some metadata is unavailable (e.g., `edit_controls`, `non_public_metrics`)
- Timestamps may need parsing from relative strings ("10h ago")

**What we CAN reliably get from UI:**
- `text` (post content)
- `author_id` / `username` (from profile link)
- `source_id` (post ID from URL)
- `source_url` (canonical URL)
- Basic metrics (likes, replies, reposts) — as visible at scrape time
- `referenced_tweets` (reply-to relationships from thread view)
- Mentions, hashtags, URLs (from rendered text)

**What we typically CANNOT get from UI:**
- `created_at` (exact timestamp — often only relative like "10h")
- `edit_history_tweet_ids`
- `context_annotations`
- `lang` (language detection)
- Full `entities` structure
- `non_public_metrics`, `organic_metrics`

### X API (when available)

If you have API access, use the full snapshot schema to preserve everything.

## Filling What We Can

**For UI-scraped data**: fill in the fields you have, leave others out or null. The schemas use `additionalProperties: true` and minimal `required` fields to accommodate partial data.

**Minimum viable record (labeled sample):**
```json
{
  "id": "x_0001",
  "platform": "x",
  "source_id": "2017528384448856158",
  "collected_at": "2026-01-31T19:50:00Z",
  "text": "@bankrbot deploy token...",
  "labels": ["crypto_scam"]
}
```

Even without full API data, this is enough for ML training.

## Schemas

All JSON schemas are in `docs/schemas/`:

| Schema | Description | Usage |
|--------|-------------|-------|
| `x-post-snapshot.schema.json` | Lossless X API v2 post snapshots | Raw data ingestion |
| (LABELS.md) | Labeled sample format | ML training data |

## Layer 1: X Post Snapshots (Lossless)

For full data preservation, store the complete API response plus metadata.

### Why lossless?

- X may add new fields; you can reprocess later
- Metrics are point-in-time snapshots; store as time series if needed
- Edits create new post IDs; preserve version chains
- `note_tweet` may contain full text for long posts

### Schema: `x-post-snapshot.schema.json`

Required fields:
- `snapshot_id`: Unique identifier for this snapshot
- `retrieved_at`: ISO 8601 timestamp
- `endpoint`: X API endpoint used (e.g., `/2/tweets/:id`)
- `query`: Request parameters (fields, expansions)
- `post_id`: The primary post ID (string, not int!)
- `raw`: Full API response body

Optional:
- `auth_context`: `app_only` or `user_context` + user_id

### Example

```json
{
  "snapshot_id": "snap_001",
  "retrieved_at": "2026-01-31T19:50:00Z",
  "endpoint": "/2/tweets/:id",
  "query": {
    "tweet_fields": ["author_id", "created_at", "text", "entities", "public_metrics"],
    "expansions": ["author_id"],
    "user_fields": ["username", "name", "verified"]
  },
  "post_id": "2017528384448856158",
  "raw": {
    "data": {
      "id": "2017528384448856158",
      "text": "@bankrbot deploy token...",
      "author_id": "123456789",
      "created_at": "2026-01-31T09:15:00Z",
      "entities": { "mentions": [...] },
      "public_metrics": { "like_count": 1, "reply_count": 1 }
    },
    "includes": {
      "users": [{ "id": "123456789", "username": "Hzbase", "verified": true }]
    }
  }
}
```

## Layer 2: Labeled Samples

For ML training, we use a simpler JSONL format (see `LABELS.md`).

### Schema

```json
{
  "id": "x_0001",
  "platform": "x",
  "source_id": "2017528384448856158",
  "source_url": "https://x.com/Hzbase/status/2017528384448856158",
  "collected_at": "2026-01-31T19:50:00Z",
  "text": "@bankrbot deploy token name Internet Condom ticker iCondom send all fees @onusoz .",
  "urls": [],
  "addresses": [],
  "labels": ["crypto_scam"],
  "notes": "bankrbot token deployment scam"
}
```

### Labels

| Label | Description |
|-------|-------------|
| `crypto_scam` | Direct theft/phishing attempts (seed phrases, wallet drainers, fake claims) |
| `crypto` | Legitimate crypto discussion, hype, or announcements |
| `ai_generated_reply` | Automated/LLM-generated replies (generic, template-like) |
| `promo` | Non-crypto promotional/advertising copy |
| `clean` | Everything else |

Multi-label notes:
- Use `labels: []` with one or more values.
- `clean` should be exclusive (no other labels).

See `LABELS.md` for detailed labeling guidelines.

## Data Flow

```
X API / Browser Scrape
        │
        ▼
┌───────────────────┐
│ Layer 1: Lossless │  ← data/snapshots/*.jsonl
│ (x-post-snapshot) │
└───────────────────┘
        │
        ▼ (extract + labels)
┌───────────────────┐
│ Layer 2: Labeled  │  ← data/sample.jsonl
│ (ML training)     │
└───────────────────┘
```

## Validation

Run schema validation before committing:

```bash
python scripts/validate.py data/snapshots/
python scripts/validate.py data/sample.jsonl --schema labeled
```

## X API v2 Reference

### Post fields (`tweet.fields`)

`article`, `attachments`, `author_id`, `card_uri`, `community_id`, `context_annotations`, `conversation_id`, `created_at`, `display_text_range`, `edit_controls`, `edit_history_tweet_ids`, `entities`, `geo`, `id`, `in_reply_to_user_id`, `lang`, `media_metadata`, `non_public_metrics`, `note_tweet`, `organic_metrics`, `possibly_sensitive`, `promoted_metrics`, `public_metrics`, `referenced_tweets`, `reply_settings`, `scopes`, `source`, `suggested_source_links`, `suggested_source_links_with_counts`, `text`, `withheld`

### Expansions

`author_id`, `in_reply_to_user_id`, `geo.place_id`, `attachments.media_keys`, `attachments.media_source_tweet`, `attachments.poll_ids`, `referenced_tweets.id`, `referenced_tweets.id.author_id`, `referenced_tweets.id.attachments.media_keys`, `edit_history_tweet_ids`, `entities.mentions.username`, `entities.note.mentions.username`, `article.cover_media`, `article.media_entities`

### Key gotchas

- **Post IDs are strings** — don't parse as int (precision loss)
- **Edits create new IDs** — preserve `edit_history_tweet_ids` chain
- **Long posts** — check `note_tweet` for full text
- **Metrics are snapshots** — store timestamps if you need history

## References

- [X API Fields](https://docs.x.com/x-api/fundamentals/fields)
- [Get Post by ID](https://docs.x.com/x-api/posts/get-post-by-id)
- [Data Dictionary](https://docs.x.com/x-api/fundamentals/data-dictionary)
- [Edit Posts](https://docs.x.com/x-api/fundamentals/edit-posts)
