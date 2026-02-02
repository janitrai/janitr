# Model Size Reduction Roadmap

## Goals
- Target <10MB ideal; 20-30MB acceptable
- Keep the full training pipeline intact (reference .bin remains source of truth)
- Reduction is scripted and repeatable

## Baseline (2026-02-02)
- models/scam_detector.bin ~767MB (reference)
- models/scam_detector.ftz ~97MB (current quantized)

## Repeatable pipeline
1. Train reference model (existing pipeline) -> models/scam_detector.bin
2. Run reduction suite -> models/reduced/*.ftz + models/reduced/reduction_results.csv
3. Select smallest model that keeps FPR <= 5% at threshold 0.90 with acceptable recall
4. Promote chosen model to models/scam_detector.ftz and update docs/MODEL.md + docs/EVAL_RESULTS.md

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
```
Results in models/reduced/reduction_results.csv

## Results (2026-02-02, threshold 0.90)

| name | size_mb | precision | recall | fpr | notes |
| --- | --- | --- | --- | --- | --- |
| quant-default | 96.06 | 0.9348 | 0.7247 | 3.5% | baseline quantized |
| quant-cutoff100k | 5.72 | 0.8924 | 0.7921 | 8.2% | |
| quant-cutoff50k | 2.95 | 0.8987 | 0.7978 | 8.2% | |
| quant-cutoff20k | 1.27 | 0.8931 | 0.7978 | 8.2% | |
| **quant-cutoff10k** | **0.69** | 0.8938 | 0.8034 | **7.1%** | ⭐ best size/FPR tradeoff |
| quant-cutoff5k | 0.40 | 0.8788 | 0.8146 | 10.6% | |
| quant-cutoff1k | 0.16 | 0.8720 | 0.8034 | 11.8% | |

**Note**: `qout=True` variants fail on this model (matrix too small for quantization).

### Key findings
- Aggressive vocab pruning (cutoff) dramatically reduces size with modest accuracy loss
- **quant-cutoff10k (690KB)** is the sweet spot: smallest FPR among reduced models
- Interestingly, 10k cutoff has *lower* FPR than larger cutoffs (extra vocab may cause false positives)
- May need threshold >0.90 to hit FPR <5% target

## Parallel work items
- Add clean crypto posts with scammy keywords to reduce false positives
- Validate WASM compatibility for top candidates
- Decide acceptance criteria for production promotion
