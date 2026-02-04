# Scam Detection Model

## Quick Start

```bash
# Check a single text
python scripts/inference.py "🚀 FREE AIRDROP! Connect wallet now!"

# Output: 🚨 SCAM (p=0.987, high confidence)

# Check from stdin
echo "just vibing with crypto friends" | python scripts/inference.py --stdin

# Output: ✅ CLEAN (p=0.123, medium confidence)
```

## Model Details

| Property | Value |
|----------|-------|
| Type | fastText supervised |
| Training samples | 1448 |
| Validation samples | 363 |
| Classes | clean, crypto, scam, promo |
| Model size | ~767MB (unquantized) |
| Inference time | <1ms per sample |

## Production Thresholds

Per-label thresholds live in `config/thresholds.json` and are applied during inference.
Use a single global threshold only for debugging or when the thresholds file is absent.
The browser extension currently uses a scam-only threshold tuned for FPR <= 2% on holdout.

**Note:** `ai_generated_reply` is currently excluded from training and inference until we have more labeled data.

## Training Pipeline

```bash
# 1. Prepare data (splits into train/valid)
python scripts/prepare_data.py

# 2. Train model
python scripts/train_fasttext.py

# 3. Evaluate
python scripts/evaluate.py --threshold 0.90

# Or tune per-label thresholds
python scripts/evaluate.py --tune --save-thresholds config/thresholds.json

# Create a time-based holdout split
python scripts/make_holdout.py --ratio 0.1
```

## Files

```
data/
  sample.jsonl      # Raw labeled data (source of truth)
  train.txt         # Generated training file
  valid.txt         # Generated validation file

models/
  scam_detector.bin # Trained model (gitignored)

scripts/
  prepare_data.py   # Data preprocessing
  train_fasttext.py # Model training
  evaluate.py       # Evaluation with metrics
  inference.py      # Production inference
```

## Label Schema

| Label | Description | Example |
|-------|-------------|---------|
| `scam` | Theft/phishing attempts (crypto or not) | "Send 1 ETH get 2 back!" |
| `crypto` | Legitimate crypto discussion | "BTC looking bullish today" |
| `promo` | Promotional/advertising copy | "Free trial — get access in 2 minutes" |
| `clean` | Non-crypto content | "Great weather today" |

Multi-label:
- Store labels as an array (e.g. `["crypto", "promo"]`).
- `clean` should be exclusive.
- fastText supports multi-label by putting multiple `__label__...` tokens on the same line.

**Recommendation:** keep `scam` and `crypto` as separate labels in the dataset. If you want a single “crypto_scam” decision, derive it during training or inference by combining `scam` + `crypto`.

**Training note:** you can cluster labels during training (e.g., merge into coarse super-classes or train with grouped targets) to improve performance, while keeping the dataset labels fine-grained for future remapping.

## Browser Extension Integration

For WASM deployment, the model needs to be:
1. Quantized (reduces ~767MB → ~5-6MB)
2. Compiled to WASM via fastText WASM port

Current extension model: `models/reduced/quant-cutoff100k.ftz` (5.72MB).

See `docs/QUANTIZATION.md` for size targets, quality gates, and the quantization pipeline.
See `docs/ARCHITECTURE.md` for the full production pipeline.
