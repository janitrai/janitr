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
See `LABELS.md` for labeling rules and `data/sample.jsonl` for the data shape.
Dataset creation is AI-first: use AI models to label crypto scams at scale and
source `ai_reply` candidates by searching X for “AI reply”. Store provenance
for every sample (platform, source id/url, timestamp). Preserve the original
text and URLs without lossy transformations.

## Local-first
No network calls required for classification. Models should run on CPU.
