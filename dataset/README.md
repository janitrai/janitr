---
license: mit
task_categories:
  - text-classification
language:
  - en
tags:
  - crypto
  - scam-detection
  - spam
  - moderation
  - content-filtering
  - ai-detection
pretty_name: Janitr
size_categories:
  - n<1K
---

# Janitr Dataset

A dataset for detecting scams, crypto content, promos, and AI-generated replies in social media content.

## Dataset Description

This dataset contains labeled examples of:
- **scam**: Posts attempting to steal funds (phishing, wallet drainers)
- **crypto**: Legitimate crypto discussion and hype
- **promo**: Promotional/advertising copy
- **ai_generated_reply**: Automated/LLM-generated replies (bot spam, template responses)
- **clean**: Normal content

### Intended Use

Training local, lightweight classifiers to filter undesirable content from social feeds, DMs, and web content. Designed to run on CPU without network calls.

### Source

Collected from X (Twitter) via UI scraping. All samples include provenance (source URL, collection timestamp).

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier |
| `text` | string | The post/message content |
| `labels` | array | One or more: `scam`, `crypto`, `promo`, `ai_generated_reply`, `clean` |
| `label_ids` | array | Numeric label IDs (0=clean, 1=crypto, 2=scam, 3=ai_generated_reply, 4=promo) |
| `platform` | string | Source platform (x, discord, web, dm, other) |
| `source_id` | string | Platform-native ID (e.g., tweet ID) |
| `source_url` | string | Canonical URL (when available) |
| `collected_at` | string | ISO 8601 timestamp of collection |

### Data Splits

| Split | Samples | Description |
|-------|---------|-------------|
| train | TBD | Training set (~80%) |
| test | TBD | Test set (~20%) |

### Label Distribution

| Label | Count | Description |
|-------|-------|-------------|
| `clean` (0) | TBD | Normal content |
| `crypto` (1) | TBD | Crypto-related content |
| `scam` (2) | TBD | Scam/phishing attempts |
| `ai_generated_reply` (3) | TBD | Automated responses |
| `promo` (4) | TBD | Promotional content |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("janitrai/janitr")

# Access splits
train = dataset["train"]
test = dataset["test"]

# Example
print(train[0])
# {'id': 'x_0011', 'text': '@bankrbot deploy...', 'labels': ['crypto', 'scam'], 'label_ids': [1, 2], ...}
```

## Labeling Guidelines

See [LABELS.md](https://github.com/janitrai/janitr/blob/main/LABELS.md) for detailed labeling rules.

**scam** — Only when there's clear theft/phishing intent:
- Seed phrase / private key requests
- "Claim airdrop" + wallet connect prompts
- Fake exchanges, wallet drainers

**ai_generated_reply** — Strong stylistic cues of automation:
- Generic, template-like responses
- Repetitive phrasing across accounts
- Low specificity to original post

**crypto** — Legitimate crypto discussion:
- Market commentary, announcements, product updates

**promo** — Promotional/advertising content:
- Product launches, affiliate pitches, marketing hooks

**clean** — Everything else:
- Normal crypto discussion ("bags", "hodl")
- Human replies, even short ones

## Limitations

- Collected via UI scraping, not API (some metadata unavailable)
- English-only for v0
- Biased toward crypto Twitter; may not generalize to other platforms
- Small dataset size initially

## Citation

```bibtex
@dataset{internet_condom_2026,
  title={Janitr: Crypto Scam and AI Reply Detection Dataset},
  author={OpenClaw Contributors},
  year={2026},
  url={https://huggingface.co/datasets/janitrai/janitr}
}
```

## License

MIT
