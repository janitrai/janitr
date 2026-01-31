# Internet Condom Plan (v0)

## Goal
Build a local-first classifier that flags undesired content with low false
positives and user-adjustable thresholds.

## Scope (initial)
- Inputs: text content (DMs, posts, web text) with URL parsing.
- Outputs: label + confidence + optional reasons.
- Classes: `crypto_scam`, `ai_reply`, `clean`.

## Non-goals (v0)
- No account-level bot scoring.
- No real-time social graph analysis.
- No remote inference dependencies.

## Principles
- Local-first: run on CPU without network calls.
- Conservative: prioritize low false positives.
- Transparent: surface reasons (rule hits, URL matches).

## Roadmap
### Phase 0 — Definitions
- Finalize labels and labeling rules.
- Define minimum data schema (JSONL).
- Decide user-adjustable thresholds and defaults.

### Phase 1 — Rules + Heuristics
- URL parsing + domain checks (scam lists).
- Keyword/phrase heuristics (seed phrase, wallet connect, airdrop).
- Basic address detection (optional).
- Output rule hits as features + explanations.

### Phase 2 — Baseline ML (CPU)
- TF-IDF + Logistic Regression (or Linear SVM).
- Train on small hand-labeled set + weak labels from rules.
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
- Start with 200–500 hand-labeled samples per class.
- Expand with weak labels (scam domains, phishing patterns).
- Active learning: review highest-uncertainty samples.

## Evaluation
- Report precision/recall/F1 per class.
- Emphasize `crypto_scam` precision.
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
- Which scam lists to bundle locally?
- How to version and update label data?
- Do we need multilingual support in v0?
