# X Syndication API — Free Tweet Data Extraction

## Overview

X (Twitter) exposes a public, unauthenticated JSON API at `cdn.syndication.twimg.com` that powers embedded tweets on third-party websites. This API returns structured tweet data without requiring any API key, OAuth token, or login session.

## Endpoint

```
GET https://cdn.syndication.twimg.com/tweet-result?id={status_id}&token=0
```

- `id`: the numeric tweet status ID
- `token`: set to `0` (required parameter but value doesn't matter)
- Returns: `application/json`

## What You Get

A single request returns:

| Field                       | Description                                    |
| --------------------------- | ---------------------------------------------- |
| `text`                      | Full tweet text                                |
| `id_str`                    | Status ID                                      |
| `user.screen_name`          | Author handle                                  |
| `user.name`                 | Display name                                   |
| `user.is_blue_verified`     | Verified status                                |
| `created_at`                | ISO timestamp                                  |
| `favorite_count`            | Like count                                     |
| `conversation_count`        | Reply count                                    |
| `lang`                      | Detected language                              |
| `entities`                  | Hashtags, URLs, user_mentions, symbols         |
| `in_reply_to_screen_name`   | Who this tweet replies to (if reply)           |
| `in_reply_to_status_id_str` | Parent tweet ID (if reply)                     |
| `parent`                    | **Full nested parent tweet object** (if reply) |
| `quoted_tweet`              | **Full nested quoted tweet object** (if QT)    |
| `edit_control`              | Edit history info                              |
| `mediaDetails`              | Media attachments (images, video) when present |

### Key Insight: Parent Nesting

When you fetch a **reply tweet**, the response includes a `parent` field containing the full parent tweet object. This means:

- Fetching the **evidence tweet** (e.g. levelsio's "Blocked for AI reply") gives you the evidence tweet + the AI reply it's responding to
- One more fetch of the AI reply's `in_reply_to_status_id_str` gives you the original post
- **2 fetches = complete 3-tweet thread**

## Example

```bash
curl -s "https://cdn.syndication.twimg.com/tweet-result?id=2021699240393609717&token=0" | jq .
```

Response (abbreviated):

```json
{
  "__typename": "Tweet",
  "id_str": "2021699240393609717",
  "text": "@BlockShaolin Blocked for AI reply",
  "user": {
    "screen_name": "levelsio",
    "name": "@levelsio",
    "is_blue_verified": true
  },
  "in_reply_to_screen_name": "BlockShaolin",
  "in_reply_to_status_id_str": "2021695412122661028",
  "created_at": "2026-02-11T21:33:54.000Z",
  "parent": {
    "id_str": "2021695412122661028",
    "text": "@levelsio The church buffet isn't about religion - it's a social credit testing ground...",
    "user": {
      "screen_name": "BlockShaolin",
      "name": "katie"
    },
    "in_reply_to_status_id_str": "2021693766793318833"
  }
}
```

## Rate Limits

- No authentication required
- No known hard rate limit, but be reasonable (1-2 req/sec)
- No API key costs
- Works from any IP

## Limitations

- **Single tweet per request** — no batch endpoint
- **No search** — you need to know the status ID already (use Google `site:x.com` search to discover IDs)
- **Deleted tweets** return an error/empty response
- **Protected accounts** won't return data
- **Thread traversal** is manual — follow `in_reply_to_status_id_str` chain yourself
- **No retweet/reply list** — can't get "all replies to tweet X"
- **May break without notice** — this is an undocumented internal API, not an official product

## Use in janitr

### Thread Collection Pipeline

For collecting "Blocked for AI reply" ground truth threads:

1. **Discover evidence tweets** via Google: `site:x.com "blocked for AI reply"`
2. **Extract status IDs** from Google result URLs
3. **Fetch evidence tweet** via syndication → get evidence + AI reply (from `parent`)
4. **Fetch original post** via syndication using AI reply's `in_reply_to_status_id_str`
5. **Assemble thread** into `replies.jsonl` schema

This replaces the manual browser-based collection workflow for bulk extraction.

### Combined with Google Search

Google search (`site:x.com "blocked for AI reply"`) + syndication API = a complete scraping pipeline with zero API keys:

| Step                    | Tool                                | Auth needed |
| ----------------------- | ----------------------------------- | ----------- |
| Discover callout tweets | Google Search                       | None        |
| Extract tweet data      | Syndication API                     | None        |
| Build thread chains     | Syndication API (follow parent IDs) | None        |

## Comparison with Other Methods

| Method                 | Auth            | Rate limits   | Data quality           | Effort |
| ---------------------- | --------------- | ------------- | ---------------------- | ------ |
| **Syndication API**    | None            | Generous      | Good (structured JSON) | Low    |
| X API v2 (pay-per-use) | OAuth + $$      | Strict tiers  | Best                   | Medium |
| Browser scraping       | X login session | Slow, fragile | Good                   | High   |
| Nitter instances       | None            | Varies        | Moderate               | Medium |

The syndication API hits a sweet spot: free, structured, and reliable enough for dataset collection at our scale (hundreds of threads, not millions).
