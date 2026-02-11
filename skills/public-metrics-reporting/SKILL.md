---
name: public-metrics-reporting
description: Convert Janitr experiment metrics into public-facing updates with concrete numbers and plain language. Use when asked for tweets, announcements, changelog notes, PR summaries, or TLDRs about accuracy, precision, recall, false positive rate, model size, and dataset growth.
---

# Public Metrics Reporting

## Workflow

1. Define the comparison window.

- Set a clear `before` baseline (usually shipped model) and `after` candidate.
- If multiple candidates exist, use the one selected for shipping.
- Refuse "improved" claims without both baseline and candidate numbers.

2. Extract the required metrics.

- Always capture:
  - Dataset size (`total samples` or split sizes).
  - Scam precision, scam recall, scam false-positive rate.
  - Topic recall and false-positive rate (`topic_crypto` when present).
  - One overall metric (`exact`, `micro_f1`, or `accuracy`).
  - Model size (`KB` or `MB`).
- Prefer JSON artifacts under `docs/reports/experiments/**`.
- Use text eval files only when JSON is missing.

3. Compute and state deltas.

- For each headline metric, show `before -> after (delta)`.
- Treat direction correctly:
  - Higher is better: precision, recall, F1, accuracy.
  - Lower is better: false-positive rate.
- Mention every meaningful regression, even when net result is positive.

4. Translate to plain language.

- Map terms consistently:
  - `false positive rate (FPR)` -> `false alarm rate`.
  - `precision` -> `when we flag something, how often we are right`.
  - `recall` -> `how many real cases we catch`.
- Avoid unexplained jargon in public copy (`holdout`, `calib`, `macro`).
- If technical terms must appear, add a one-line explanation.

5. Produce output for channel.

- Return these by default unless user asks otherwise:
  - `TLDR` one-liner.
  - `Tweet` (<=280 chars, numbers first, plain wording).
  - `Release note` bullets (3-6 bullets, includes tradeoffs).
- Keep percentages to one decimal place unless precision is critical.

## Output Rules

- Always include absolute context with rates when available (example: `18 false alarms out of 1,000 posts`).
- Never hide tradeoffs. If recall drops while false alarms improve, state both.
- Do not report only relative percentages without raw before/after values.
- When sources conflict, call out the mismatch and ask which artifact is canonical.

## Quick Commands

Use these as starting points and adjust the date/path:

```bash
# Compare shipped baseline vs tuned single-stage summary
jq '.baseline_shipped_holdout, .single_stage_class_specific_best_holdout' \
  docs/reports/experiments/2026-02-11/SUMMARY_AGGREGATED.json

# Pull winner metrics from bakeoff summary
jq '.overall_winner_by_gate' \
  docs/reports/experiments/2026-02-11-postsplit-bakeoff/2026-02-11-bakeoff_summary.json

# Parse class metrics from text eval when JSON is unavailable
rg -n "Per-class metrics|exact match accuracy|micro precision|scam|topic_crypto|clean" \
  docs/reports/experiments/**/*.txt
```

## Templates

Use `references/public-copy-templates.md` to generate:

- Plain-language TLDR
- Public tweet
- Release-note bullet list
