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
