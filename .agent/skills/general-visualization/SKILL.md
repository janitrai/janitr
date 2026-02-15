---
name: general-visualization
description: Format comparisons as aligned plain text with alternatives as columns, metrics as rows, decimal-point alignment, and inline winner-impact stars.
---

# General Visualization

## Purpose

Render quantitative results in a compact format that is easy to scan in terminal-style clients.
Prefer aligned plain text blocks over Markdown tables unless the user explicitly asks for Markdown tables.

## Core Rules

- Use aligned plain-text rows for comparisons; do not use Markdown tables.
- For comparisons, keep alternatives on the horizontal axis (columns) and metrics on the vertical axis (rows).
- Use the real alternative names as column headers (for example, `fastText-ftz`, `transformer-int8`).
- Do not use generic `Left`/`Right` headers or separate mapping notes.
- Keep decimal points vertically aligned within each numeric column.
- Keep one precision policy per output.
- Default precision policy: 2 significant figures.
- For decimals with absolute value under 1, omit the leading zero (`.84`, `-.07`).
- Use a `change` column as absolute delta: `second alternative column - first alternative column`.
- Format `change` as signed integer percentages (rounded) with percent signs aligned (`+3%`, `-6%`, `0%`).
- Preserve units and directionality (`higher is better` vs `lower is better`).
- Do not include a `better/worse` verdict column.
- Do not use HTML tags for emphasis.
- Attach the impact marker directly to the better value as a suffix.
- Keep a fixed-width star slot so markers line up vertically (`*`, `**`, `***`, `****`).
- For ties or near-ties (`<1%`), do not add any marker.
- Impact markers:
  - no marker for roughly equal (`0%` to `<1%`)
  - `*` for `1%` to `<5%`
  - `**` for `5%` to `<10%`
  - `***` for `10%` to `50%`
  - `****` for `>50%`
- If direction is ambiguous, state the assumption explicitly.
- Keep row labels short and stable (`Scam precision`, `Macro F1`, `Latency p95`).
- Do not hide regressions.

## Metric Direction Defaults

- Higher is better: `precision`, `recall`, `f1`, `accuracy`, `auc`, `throughput`.
- Lower is better: `fpr`, `fnr`, `latency`, `error`, `loss`, `size`, `memory`.
- For unknown metric names, require an explicit assumption before choosing the better value.

## Improvement Calculation

- Compute signed change as `change% = (second alternative - first alternative) * 100`.
- Determine which side is better using metric direction.
- Use `abs(change%)` to choose marker strength.
- If `abs(change%) < 1`, use no marker.

## Workflow

1. Collect rows: metric, baseline, candidate.
2. Set direction for each metric.
3. Determine the better value for each metric.
4. Compute signed `change = second-column - first-column` in percentage points.
5. Map `abs(change)` to impact marker (`*`, `**`, `***`, `****`) with no marker under `1%`.
6. Append the marker to the winning value only and pad marker width for alignment.
7. Render aligned rows with alternatives as columns and metrics as rows.

## Output Templates

### Side-by-Side Comparison (default, terminal-safe)

Metric fastText-ftz transformer-int8 Change
Scam precision .92 .95 \* +3%
Scam recall .70 .78 _\*\* +8%
Scam FPR .018 .012 -1%
Macro F1 .83 .85 _ +2%

### Comparison with Delta

Metric baseline-v1 candidate-v2 Change
Model size MB 3.4 3.1 ** -9%
Scam recall .78 \* .76 -2%
Latency p95 ms 58 ** 61 +5%

## Style Controls

- Keep output terminal-friendly plain text; avoid HTML tags entirely.
- If user requests no table, still use aligned rows (not Markdown tables).
- If output is short (1-3 metrics), keep to single compact block without extra sections.
- Do not add a dedicated better/worse column unless explicitly requested.

This skill intentionally uses no bundled scripts/resources.
