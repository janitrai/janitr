# Data Sources (AI-first)

## Goal
Source text data for `scam`, `crypto`, `ai_generated_reply`, `promo`, and `clean`, with provenance
stored for every record and without losing important information.

## Primary sources
- X (Twitter): posts and replies.
- Discord: DMs and server messages.
- Web: site content or scraped text you control.

## Ingestion
- Use your existing OpenClaw instance to collect posts via its provided browser
  and write JSONL records in the schema below.
- OpenClaw is the only ingestion path (no scripts in this repo).
- Ensure the OpenClaw instructions preserve full text and metadata.
- X home-feed scraping flow: see `docs/SCRAPING.md`.

## OpenClaw JSONL instruction block (copy/paste)
```
Collect posts from X using your queries. For each post, output one JSONL line
with the required schema fields:
id, platform="x", source_id, source_url, collected_at (ISO), text, labels[],
urls[], addresses[], notes (optional).

Preserve the original text exactly (including emojis, casing, punctuation).
Do not truncate text. Keep original URLs (no shortening or normalization).

Write to: data/raw/x_openclaw_YYYYMMDD.jsonl
```

## AI-first labeling workflow
1) Collect raw text + metadata.
2) Use AI models to label `scam` at scale.
3) Source `ai_generated_reply` candidates by searching X for “AI reply”.
4) Collect non-crypto promo/ads and label as `promo`.
5) Keep everything else as `clean` unless the AI labels it otherwise.

## Data fidelity (do not lose information)
- Preserve original text exactly (including emojis, casing, punctuation).
- Keep original URLs; do not shorten or normalize them.
- Store the full text even if it is long; do not truncate.
- Avoid lossy transformations (strip HTML only if you also store raw).

## Provenance fields (required)
- `platform`: x | discord | web | dm | other
- `source_id`: platform-native id (tweet id, message id, etc.)
- `source_url`: canonical URL when available
- `collected_at`: ISO timestamp

## JSONL schema (recap)
Each record:
```
{
  "id": "x_0001",
  "platform": "x",
  "source_id": "1234567890",
  "source_url": "https://x.com/...",
  "collected_at": "2026-01-31T00:00:00Z",
  "text": "raw text",
  "labels": ["crypto", "scam"],
  "urls": ["https://example.com"],
  "addresses": ["..."],
  "notes": "optional"
}
```

Multi-label example:
```
{
  "id": "x_0002",
  "platform": "x",
  "source_id": "1234567891",
  "source_url": "https://x.com/...",
  "collected_at": "2026-01-31T00:05:00Z",
  "text": "USDC + product launch promo...",
  "labels": ["crypto", "promo"]
}
```

## Sampling strategy
- Ensure class balance across `scam`, `crypto`, `ai_generated_reply`, `promo`, and `clean`.
- Use multiple labels when the attributes are orthogonal (e.g. `crypto` + `promo`).
- Deduplicate near-identical text.
- Keep a separate holdout split for evaluation.

## Output location
- Store curated datasets in `data/` as JSONL.
- Keep raw dumps in `data/raw/` (optional).
