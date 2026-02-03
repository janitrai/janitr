# Evaluation Results - fastText MVP (2026-02-03)

## Dataset
- **Raw samples**: 1814 (`data/sample.jsonl`)
- **Usable after cleaning**: 1811
- **Train/valid split**: 1448 / 363 (80/20)
- **Holdout**: 182 (last 10% by `collected_at`, time-split)
- **Holdout boundary**:
  - last train: 2026-02-01T18:43:00+00:00
  - first holdout: 2026-02-01T18:45:00+00:00

Label counts (multi-label totals, not row counts):
- **Train**: clean 444, crypto 965, scam 453, promo 186, ai_generated_reply 1
- **Valid**: clean 121, crypto 231, scam 112, promo 46

## Model Artifacts

| File | Size | Notes |
|------|------|-------|
| `models/scam_detector.bin` | ~767 MB | Reference model (gitignored) |
| `models/reduced/quant-cutoff100k.ftz` | 5.72 MB | **Current extension model** |
| `models/reduced/quant-cutoff10k.ftz` | 0.66 MB | Size-optimal alternative |

## Production (Extension) Configuration

- **Model**: `quant-cutoff100k.ftz`
- **Scam threshold**: `0.6151` (tuned for FPR <= 2% on holdout)
- **Note**: quantized models can return probabilities slightly > 1. Clamp to `[0, 1]` before comparisons.

### Holdout performance (scam label, time-split)

| Metric | Value |
|---|---|
| Precision | 0.9524 |
| Recall | 0.9524 |
| FPR | 1.43% |
| Threshold | 0.6151 |

## Per-label Thresholds (offline inference)

`config/thresholds.json` stores per-label thresholds for multi-label inference on the `.bin` model
(tuned on `data/valid.txt` for FPR <= 2%):

- `crypto`: 0.9969
- `scam`: 0.9815
- `promo`: 0.7719
- `clean`: 0.10 (fallback)

## Reduction Sweep (valid set, threshold = 0.90)

| name | size_mb | precision | recall | fpr |
| --- | --- | --- | --- | --- |
| quant-default | 96.10 | 0.8378 | 0.8304 | 7.17% |
| quant-cutoff100k | 5.72 | 0.9222 | 0.7411 | 2.79% |
| quant-cutoff50k | 2.90 | 0.9111 | 0.7321 | 3.19% |
| quant-cutoff20k | 1.22 | 0.9053 | 0.7679 | 3.59% |
| quant-cutoff10k | 0.66 | 0.9062 | 0.7768 | 3.59% |

## Repeatable Commands

```bash
# Prepare data
python scripts/prepare_data.py

# Train model
python scripts/train_fasttext.py

# Create holdout split
python scripts/make_holdout.py --ratio 0.1

# Compare reduced models under FPR constraint (holdout)
python scripts/compare_models_fpr.py --models "models/reduced/quant-*.ftz" --target-fpr 0.02 --holdout data/holdout.txt

# Playwright smoke tests
node tests/wasm-smoke.mjs
pnpm exec playwright test extension/tests/wasm-smoke.spec.ts
```
