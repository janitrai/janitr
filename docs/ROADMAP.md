# Model Size Reduction Roadmap

## Goals
- Target <10MB ideal; 20-30MB acceptable
- Keep the full training pipeline intact (reference .bin remains source of truth)
- Reduction is scripted and repeatable

## Baseline (2026-02-03)
- `models/scam_detector.bin` ~767MB (reference)
- `models/reduced/quant-cutoff100k.ftz` 5.72MB (**current extension model**)
- `models/reduced/quant-cutoff10k.ftz` 0.66MB (size-optimal alternative)

## Repeatable pipeline
1. Train reference model (existing pipeline) -> `models/scam_detector.bin`
2. Run reduction suite -> `models/reduced/*.ftz` + `models/reduced/reduction_results.csv`
3. Compare models under an FPR constraint on holdout
4. Promote chosen model to extension and update docs

## Reduction options (post-training)
- Quantize with qout/qnorm/dsub for aggressive compression
- Prune vocabulary + ngrams with cutoff (keep top N)
- Optional PCA reduction before quantization (experimental; may harm accuracy)
- Retrain-after-quantization is excluded for now (must start from .bin only)

## How to run
```bash
python scripts/reduce_fasttext.py --profile compact
python scripts/reduce_fasttext.py --profile grid --cutoffs 0,200000,100000,50000,20000 --dsubs 2,4,8
python scripts/reduce_fasttext.py --profile compact --pca-dims 50,25

# Compare reduced models under FPR constraint (holdout)
python scripts/compare_models_fpr.py --models "models/reduced/quant-*.ftz" --target-fpr 0.02 --holdout data/holdout.txt
```

## Results (2026-02-03, threshold 0.90 on valid)

| name | size_mb | precision | recall | fpr |
| --- | --- | --- | --- | --- |
| quant-default | 96.10 | 0.8378 | 0.8304 | 7.17% |
| quant-cutoff100k | 5.72 | 0.9222 | 0.7411 | 2.79% |
| quant-cutoff50k | 2.90 | 0.9111 | 0.7321 | 3.19% |
| quant-cutoff20k | 1.22 | 0.9053 | 0.7679 | 3.59% |
| quant-cutoff10k | 0.66 | 0.9062 | 0.7768 | 3.59% |

**Note**: `qout=True` variants fail on this model (matrix too small for quantization).

### Key findings
- For **low false alarms**, `quant-cutoff100k` yields the best recall under FPR <= 2% on holdout.
- `quant-cutoff10k` remains the best size/recall tradeoff when <1MB is required.
- Larger models reduce false positives but cost more disk and load time.

## FPR-target tuning (holdout, scam label)

| Model | Threshold | FPR | Recall | Precision |
| --- | --- | --- | --- | --- |
| quant-cutoff100k | 0.6151 | 1.43% | 0.9524 | 0.9524 |

**Production config (extension):**
- Model: `quant-cutoff100k.ftz`
- Threshold: `0.6151`
- FPR target: <= 2% on holdout

## Next Steps
1. Keep expanding clean data with scammy keywords to reduce false positives
2. Maintain time-based holdout and re-check FPR on every retrain
3. Evaluate smaller models if bundle size becomes a hard constraint
