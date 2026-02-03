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
pretty_name: Internet Condom
size_categories:
  - n<1K
---

# Internet Condom Dataset

A dataset for detecting crypto scams and AI-generated replies in social media content.

## Dataset Description

This dataset contains labeled examples of:
- **crypto_scam**: Posts attempting to steal funds (token deployments, wallet drainers, phishing)
- **ai_generated_reply**: Automated/LLM-generated replies (bot spam, template responses)
- **clean**: Normal content (including crypto discussion that isn't a scam)

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
| `label` | string | One of: `crypto_scam`, `ai_generated_reply`, `clean` |
| `label_id` | int | Numeric label (0=clean, 1=crypto_scam, 2=ai_generated_reply) |
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
| `crypto_scam` (1) | TBD | Scam/phishing attempts |
| `ai_generated_reply` (2) | TBD | Automated responses |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("your-username/internet-condom")

# Access splits
train = dataset["train"]
test = dataset["test"]

# Example
print(train[0])
# {'id': 'x_0011', 'text': '@bankrbot deploy...', 'label': 'crypto_scam', 'label_id': 1, ...}
```

## Labeling Guidelines

See [LABELS.md](https://github.com/your-username/internetcondom/blob/main/LABELS.md) for detailed labeling rules.

**crypto_scam** — Only when there's clear theft/phishing intent:
- Seed phrase / private key requests
- "Claim airdrop" + wallet connect prompts
- Fake exchanges, wallet drainers

**ai_generated_reply** — Strong stylistic cues of automation:
- Generic, template-like responses
- Repetitive phrasing across accounts
- Low specificity to original post

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
  title={Internet Condom: Crypto Scam and AI Reply Detection Dataset},
  author={OpenClaw Contributors},
  year={2026},
  url={https://huggingface.co/datasets/your-username/internet-condom}
}
```

## License

MIT
