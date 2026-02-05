# Janitr Plan (v0)

## Goal

Build a local-first classifier that flags undesired content with low false
positives and user-adjustable thresholds.

## Scope (initial)

- Inputs: text content (DMs, posts, web text) with URL parsing.
- Outputs: labels[] + confidence + optional reasons.
- Classes: `scam`, `crypto`, `ai_generated_reply`, `promo`, `clean`.

## Non-goals (v0)

- No account-level bot scoring.
- No real-time social graph analysis.
- No remote inference dependencies.

## Principles

- Local-first: run on CPU without network calls for inference.
- Conservative thresholds: prioritize low false positives.
- Provenance: store where each text came from.
- Fidelity: store data without losing important information.

## Roadmap

### Phase 0 — Definitions

- Finalize labels and labeling rules.
- Define minimum data schema (JSONL) with provenance fields.
- Define data-fidelity rules (preserve original text, links, and metadata).
- Decide user-adjustable thresholds and defaults.

### Phase 1 — Data Sourcing (AI-first)

- Ingest with OpenClaw only (no scripts in this repo).
- Use the provided browser to collect all source content.
- Use AI models to label scam content at scale.
- For `ai_generated_reply`, source candidates by searching X for “AI reply”.
- Collect non-crypto promo/ads and label as `promo`.
- Store provenance for every record (platform, source id/url, timestamp).

### Phase 2 — Baseline ML (CPU)

- TF-IDF + Logistic Regression (or Linear SVM).
- Train primarily on AI-labeled data.
- Calibrate class thresholds to control false positives.

### Phase 3 — Model Upgrade (optional)

- Small transformer embeddings (MiniLM/Distil) + light classifier.
- Quantization for speed.
- Evaluate improvements vs baseline.

### Phase 4 — Packaging

- CLI interface returning JSON.
- Simple config for thresholds and rule toggles.
- Optional local daemon for integration.

## Data Strategy

- Use AI models to label the bulk of the dataset.
- Add `ai_generated_reply` samples by searching X for “AI reply”.
- Add `promo` samples from non-crypto marketing content.
- Keep provenance fields for source tracking.

## Evaluation

- Report precision/recall/F1 per class.
- Emphasize `scam` precision.
- Maintain a fixed holdout set from day one.

## Thresholding

- Per-class confidence cutoffs.
- Default to conservative (low false positives).
- Allow user overrides in config.

## Risks

- Label noise from weak labeling sources.
- Domain shift across platforms (X vs Discord vs web).
- Over-filtering if thresholds are too aggressive.

## Open Questions

- Which AI models to use for labeling at scale?
- How to version and update label data?
- Do we need multilingual support in v0?
