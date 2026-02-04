# Model Quantization Targets

This document specifies quantization targets by deployment context, independent of model version.

## Deployment Contexts

### 1. Browser Extension (WASM) — Primary

| Property | Target |
|----------|--------|
| **Max size** | 6 MB |
| **Ideal size** | 3-5 MB |
| **Format** | `.ftz` (fastText quantized) |
| **Latency** | <10ms per inference |

**Rationale:** Extension must load quickly and not bloat browser memory. Users on slow connections should see <3s model fetch. 6MB is the hard ceiling; aim for 3-5MB.

**Accuracy tradeoff:** Accept moderate recall loss to stay within size. FPR < 2% is non-negotiable.

**Quantization approach:**
- Use `-cutoff` to prune vocabulary (start at 50k-100k, tune down if needed)
- `dsub=2` (product quantization)
- Test with `-qnorm` and `-qout` for additional compression

---

### 2. Server-Side / API (Future)

| Property | Target |
|----------|--------|
| **Max size** | 100 MB |
| **Ideal size** | <50 MB |
| **Format** | `.bin` (unquantized) or `.ftz` |
| **Latency** | <5ms per inference |

**Rationale:** Server has more resources. Use larger model for higher accuracy when enrichment stages need it (Stage 1 in architecture).

**Accuracy tradeoff:** Prioritize recall; FPR still matters but can be slightly higher if downstream stages correct.

**Quantization approach:**
- Default quantization (no cutoff) acceptable
- Or skip quantization entirely if memory allows

---

### 3. Mobile App (Future)

| Property | Target |
|----------|--------|
| **Max size** | 10 MB |
| **Ideal size** | 3-5 MB |
| **Format** | `.ftz` or ONNX quantized |
| **Latency** | <20ms per inference |

**Rationale:** Similar constraints to browser. Slightly more headroom but still tight.

**Quantization approach:** Same as browser extension.

---

## Quantization Pipeline

Script: `scripts/reduce_fasttext.py`

**Standard size variants to generate:**

| Name | Cutoff | Expected Size | Use Case |
|------|--------|---------------|----------|
| `quant-default` | 0 | ~100 MB | Server-side |
| `quant-cutoff100k` | 100,000 | ~6 MB | Browser (max) |
| `quant-cutoff50k` | 50,000 | ~3 MB | Browser (ideal) |
| `quant-cutoff20k` | 20,000 | ~1.5 MB | Aggressive |
| `quant-cutoff10k` | 10,000 | ~700 KB | Minimal |

**Always generate all variants** when retraining/recalibrating. Store in `models/reduced/` with accuracy metrics in `reduction_results.csv`.

---

## Grid Search Results (2026-02-04)

Comprehensive grid search across cutoffs (1k-200k) × dsub (1,2,4,8) × qnorm × qout.

### Key Finding: Smaller is Better

Counter-intuitively, **cutoff=1000** produced the best crypto recall. Aggressive vocabulary pruning forces the model to focus on the most discriminative features.

### Best Model (Browser Extension)

```
File: models/experiments/quant_grid_10mb/grid_w1_c25_lr0.2_cut1000_dsub8_qout0_qnorm0.ftz
Size: 0.12 MB (120 KB)
```

| Label | Recall | Precision | FPR | Threshold |
|-------|--------|-----------|-----|-----------|
| Crypto | **89.2%** | 99.3% | 1.1% | 0.7432 |
| Scam | 33.7% | 91.9% | 1.9% | 0.9305 |

### Comparison: Cutoff vs Crypto Recall (dsub=8, qout=0, qnorm=0)

| Cutoff | Size | Crypto Recall | Notes |
|--------|------|---------------|-------|
| 1,000 | 0.12 MB | 89.2% | **Best** |
| 5,000 | 0.20 MB | 86.7% | |
| 10,000 | 0.30 MB | 83.7% | |
| 20,000 | 0.51 MB | 80.7% | |
| 50,000 | 1.16 MB | 74.7% | |
| 100,000 | >10 MB | - | Exceeds limit |

### Why Small Cutoff Wins

1. **Focus**: Fewer vocabulary items = model learns most predictive tokens
2. **Generalization**: Less overfitting to rare words in training data
3. **Quantization-friendly**: Smaller matrices quantize with less information loss

### Full Results

See `docs/logs/2026-02-04.md` for complete grid search table with all 40+ configurations.

---

## Quality Gates

Before deploying any quantized model:

1. **FPR < 2%** on holdout set (hard requirement)
2. **Crypto recall > 60%** (soft target, document if lower)
3. **Scam recall > 25%** (soft target, improving with data)
4. **Size within target** for deployment context

Run `scripts/reduce_fasttext.py` and check `reduction_results.csv` against these gates.

---

## Versioning

When generating quantized variants:
- Tag with source model hash or training date
- Keep `reduction_results.csv` in sync
- Update `MODEL.md` "Current extension model" when deploying new version

---

## References

- [fastText quantization docs](https://fasttext.cc/docs/en/faqs.html#how-can-i-reduce-the-size-of-my-fasttext-models)
- `docs/ARCHITECTURE.md` — deployment contexts
- `docs/MODEL.md` — current production model
