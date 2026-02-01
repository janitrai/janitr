# Internet Condom

Local runnable models for filtering out crypto scam content, AI replies, and
other undesirable content.

## Scope (v0)
- Input: text content (DMs, posts, web text) with URL parsing.
- Output: label + confidence + optional reasons.
- Classes: `crypto_scam`, `ai_reply`, `clean`.
- Goal: low false positives, with user-adjustable thresholds.

## Pipeline (minimal)
1) Parse text and extract URLs (and optionally wallet addresses).
2) Rule features: scam domains, seed phrase / wallet-drainer phrases, etc.
3) ML baseline: TF-IDF + Logistic Regression on CPU.
4) Thresholding: per-class confidence cutoffs to keep false positives low.

## Data
See `docs/LABELS.md` for labeling rules and `data/sample.jsonl` for the data shape.
See `docs/DATA_MODEL.md` for schemas and storage patterns.

Dataset creation is AI-first: use AI models to label crypto scams at scale and
source `ai_reply` candidates by searching X for "AI reply". Ingestion is done
entirely via OpenClaw (no scripts in this repo), using its provided browser to
collect everything. Store provenance for every sample (platform, source id/url,
timestamp). Preserve the original text and URLs without lossy transformations.

**Note:** We collect data via UI scraping (browser automation), not the X API.
This means we only capture fields visible in the DOM. See `docs/DATA_MODEL.md`
for details on what fields are available and what's missing. Fill in what you
can; partial records are fine for ML training.

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/PLAN.md](docs/PLAN.md) | Project roadmap and phases |
| [docs/TRAINING_INFERENCE_STACK.md](docs/TRAINING_INFERENCE_STACK.md) | SOTA training & inference architecture |
| [docs/DATA_MODEL.md](docs/DATA_MODEL.md) | Schemas and storage patterns |
| [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) | Where to find data |
| [docs/LABELS.md](docs/LABELS.md) | Labeling rules and definitions |
| [docs/SOURCES.md](docs/SOURCES.md) | Source tracking |
| [docs/ACCOUNTS.md](docs/ACCOUNTS.md) | Scam account registry |

## Local-first
No network calls required for classification. Models should run on CPU.
