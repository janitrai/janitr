# Evaluation Results - fastText MVP (2026-02-02)

## Dataset
- **Total samples**: 1683
- **Training**: 1346 samples (80%)
- **Validation**: 337 samples (20%)
- **Label distribution**: 916 scam, 419 clean, 347 crypto, 1 ai_generated_reply

## Model Artifacts

| File | Size | Notes |
|------|------|-------|
| `models/scam_detector.bin` | 767 MB | Original fastText model |
| `models/scam_detector.ftz` | 97 MB | Quantized (87% reduction) |

**Target**: <10 MB for browser extension deployment

## Production Configuration

```python
PRODUCTION_THRESHOLD = 0.90
```

At threshold 0.90:
- **False Positive Rate**: 4.7% ✅ (4/85 clean posts flagged as scam)
- **Recall**: 75.8% (catches ~3/4 of scams)
- **Target FPR**: ≤5% — **ACHIEVED**

## Threshold Sweep Results

| Threshold | Precision | Recall | F1 | FPR |
|-----------|-----------|--------|-----|-----|
| 0.50 | 0.88 | 0.82 | 0.85 | 12.94% |
| 0.60 | 0.89 | 0.81 | 0.85 | 11.76% |
| 0.70 | 0.90 | 0.80 | 0.85 | 10.59% |
| 0.80 | 0.92 | 0.78 | 0.85 | 8.24% |
| **0.90** | **0.94** | **0.76** | **0.84** | **4.71%** |
| 0.95 | 0.96 | 0.70 | 0.81 | 2.35% |

## Tradeoffs

- **0.50 threshold**: Catches more scams (82% recall) but 13% false positives = too noisy
- **0.90 threshold**: Fewer false alarms (4.7% FPR) but misses ~24% of scams
- **Decision**: Prefer low FPR for browser extension UX — users disable noisy tools

## False Positives at 0.90 (4 samples)

1. Legitimate crypto discussions with scam-adjacent keywords
2. Sarcastic/joking posts about scams
3. Official project announcements with "claim"/"airdrop" language

## Next Steps

1. ~~Threshold tuning~~ ✅ Done (0.90 is production default)
2. **Model size reduction** — 97 MB → target <10 MB
   - Reduce embedding dimensions
   - Prune vocabulary
   - Use unigrams only (drop bigrams)
3. **More clean data** — add crypto posts with scammy keywords
4. **Browser extension integration** — WASM export

## Training Commands

```bash
# Prepare data
python scripts/prepare_data.py

# Train model
python scripts/train_fasttext.py

# Evaluate (uses 0.90 threshold by default)
python scripts/evaluate.py

# Threshold sweep
python scripts/evaluate.py --sweep
```
