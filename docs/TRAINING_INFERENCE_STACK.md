# Training & Inference Stack

SOTA training and inference architecture for extremely efficient local ML filtering.
Runs on browser CPU, phones, and grandma's 2012 Windows tower.

## Architecture Overview

Use a **2-stage stack**:

1. **Rules + ultra-light ML (primary)** — runs everywhere (browser CPU, phones, 2012 PCs)
2. **Tiny Transformer (optional secondary)** — only for borderline cases, accelerated by WebGPU when available

This is the highest accuracy achievable under extreme CPU/RAM constraints because most spam/scam/AI-slop signals are lexical/URL-pattern driven and don't justify paying Transformer cost on every post.

---

## Train from Scratch or Fine-tune?

### Primary model: Train from scratch

Train a bag-of-ngrams classifier (fastText or hashed linear model):
- Extremely fast on CPU
- Tiny RAM footprint
- Robust to obfuscation with character n-grams
- Easy to distill into (teacher → student)

fastText is explicitly designed for efficient supervised text classification (spam-like problems) and has a WebAssembly module for in-browser execution.

### Secondary model (optional): Fine-tune

Fine-tune a small pretrained Transformer encoder (student) and run it only when needed. Deploy via ONNX Runtime Web and/or Transformers.js with WebGPU acceleration.

Training a Transformer from scratch is almost never worth it here (data + compute heavy, and we don't need generative capability—just classification).

---

## Stage Architecture

### Stage 0: Deterministic Rules (always-on)

Zero/near-zero cost checks before ML:

- **URL extraction + normalization** (punycode handling, strip trackers, domain/TLD tokens)
- **Wallet/address patterns** (ETH `0x…`, SOL base58 patterns, seed phrase triggers)
- **Known scam phrase patterns** ("airdrop", "connect wallet", "verify", "claim", "limited time", etc.)
- **Cheap "AI reply" heuristics** (e.g., "as an AI…", templated disclaimers)

**Output:**
- Hard block if rules are high-confidence
- Else pass features forward as special tokens (e.g., `HAS_WALLET`, `HAS_URL`, `__TLD_XYZ__`)

This gives you "reasons" essentially for free.

### Stage 1 (primary): fastText Supervised Classifier

Word + char n-grams. Why this is the best default:

- fastText is built for efficient text classification
- WebAssembly support means the exact same model runs in-browser
- Char n-grams handle obfuscation (clаim with Cyrillic "a", spaced-out words, leetspeak, etc.)

**Expected footprint:** A few MB to tens of MB depending on bucket sizes and embeddings; usually far smaller + faster than Transformers for equivalent throughput on CPU.

### Stage 2 (optional): Tiny Transformer Encoder

Only run when Stage 1 confidence is near threshold (or when user enables "high accuracy mode").

**Constraints:**
- 4–6 layers
- Hidden size 256–384
- Max sequence length 96–128 tokens (fits X posts)
- Export to ONNX, quantize aggressively

Run in browser via ONNX Runtime Web (WASM CPU fallback, WebGPU fast path).

---

## Data Pipeline

### Fields to Store Per Sample

```
id                    # unique identifier
source                # platform (x, telegram, discord, etc.)
source_id             # platform-specific post ID
source_url            # direct link to content
timestamp             # collection time
raw_text              # original unmodified text
normalized_text       # NFKC, lowercase, collapse whitespace, strip zero-width
urls: [               # list of extracted URLs
  {
    raw               # original URL string
    domain            # extracted domain
    registrable_domain # if PSL offline implemented
    tld               # top-level domain
    path_tokens       # split on /, -, _
  }
]
rule_hits             # list of matched rule IDs (for reasons + feature tokens)
label                 # crypto_scam | ai_reply | clean
label_confidence      # model confidence score
label_source          # human | teacher | heuristic
lang                  # optional; cheap fastText language ID later if needed
```

---

## Splits to Prevent Overfitting

**Do not random-split only.** Use:

1. **Time-based split** — train on older, test on newer to simulate deployment drift
2. **Domain-based split for scams** — hold out entire domains so the model must learn patterns beyond memorizing URLs

---

## Training Pipeline

### Step 1: Establish True Baseline (v0)

Implement:
- TF-IDF + logistic regression on CPU
- Plus rules

This gives you an error profile and a labeling feedback loop quickly.

### Step 2: Replace TF-IDF with Deployable Ultra-light Model

Pick one (fastText is usually simplest end-to-end):

**Option A (recommended): fastText**
- Train supervised classifier with:
  - Word n-grams (1–2)
  - Subword/char n-grams enabled (key for obfuscation)
  - Hashing buckets sized to your memory budget
- Export `.bin` and `.vec` as needed (usually just need `.bin`)
- fastText supports supervised training for text classification
- Can be used via WebAssembly in-browser

**Option B: Hashed Linear Model**
- HashingVectorizer-like pipeline (word + char n-grams) + linear classifier
- Pros: easiest "top features" explanations
- Cons: must implement hashing/tokenization carefully in JS to match training

### Step 3: Teacher → Student Distillation (SOTA quality)

To maximize quality without paying runtime cost:

1. **Train a high-accuracy teacher** (not for deployment):
   - A larger Transformer encoder sequence classifier
   - Can be anything strong you can run during training

2. **Use teacher to generate:**
   - Hard pseudo-labels on weakly-labeled pool
   - Or soft probabilities for distillation

3. **Train the student (fastText) on:**
   - Your human-labeled set
   - Plus teacher-labeled expansions
   - With hard-negative mining (see below)

This is how you get near-Transformer accuracy while keeping fastText CPU performance.

### Step 4: Hard-Negative Mining Loop (main win for low FP)

Goal: low false positives with adjustable thresholds.

**Process:**
- Collect:
  - False positives from real browsing sessions (user "unhide" action)
  - False negatives (user "report scam/slop" action)
- Re-train weekly or per N samples
- Overweight hard negatives during training (or oversample them)

---

## Quantization + Optimization (for optional Transformer)

If you deploy Stage 2:

1. **Export to ONNX**
2. **Quantize:**
   - Start with dynamic int8 (easy, usually good accuracy)
   - If needed, move to static/PTQ with calibration or QAT
3. **Use Olive** (from Microsoft) to automate optimization/quantization workflows across hardware targets

**Result:** Smaller downloads, lower RAM bandwidth, faster inference.

---

## Inference Stack

### Browser (extension/userscript) Runtime

**Goal:** Never block UI; keep RAM stable; handle a stream of posts.

#### Execution Environment

**Stage 1 (fastText):**
- Run in a WebWorker using fastText WebAssembly

**Stage 2 (optional Transformer):**
- ONNX Runtime Web:
  - WebGPU fast path
  - WASM CPU fallback
- Transformers.js v3 is a practical wrapper for this in JS

#### Performance Controls

- **Limit ORT WASM threads** to avoid pegging CPUs on old hardware
  - ONNX Runtime Web exposes `env.wasm.numThreads`
- **Batch posts when possible** (e.g., process 8–32 at a time) to reduce overhead
- **Cache by content hash:**
  - Don't re-run on the same post when it reappears
  - Store decisions in an LRU map (memory-bounded)

#### Gating Logic

```
Rules first:
  if rule says "definite scam" → hide immediately

Stage 1:
  if p(scam) > threshold_scam → hide
  else if p(ai_reply) > threshold_ai → collapse
  else if near threshold band → Stage 2 (if enabled)

Stage 2:
  final decision + calibrated probability
  Always allow user override (and log for training)
```

### Local Desktop / Server-side Browsing Instances

If running inside automation/browsers on machines (OpenClaw instances):
- Reuse the same model artifacts
- Run Stage 1 in:
  - Node.js with WASM, or
  - Native fastText if allowed
- Run Stage 2 in:
  - ONNX Runtime native CPU (same ONNX model)

**Train once, deploy everywhere.**

---

## Probability Calibration + Thresholds

**Do not treat raw scores as probabilities.**

### Offline:
- Fit temperature scaling / isotonic regression on a validation set
- Select thresholds to hit target FP rates (e.g., precision ≥ 0.995 on clean)

### Runtime:
- Expose "strictness" slider that maps to thresholds:
  - **Conservative mode:** only hide at very high confidence
  - **Aggressive mode:** hide at moderate confidence

This directly implements the goal of user-adjustable cutoffs.

---

## Implementation Milestones

### Milestone A — MVP that actually ships

1. Rules engine + URL extraction
2. Train Stage 1 fastText on dataset
3. Browser worker inference + thresholding + UI hide/collapse
4. Feedback capture (undo/report) → append to jsonl

### Milestone B — Make it robust to evasion

1. Add char n-grams + unicode normalization
2. Domain-holdout + time-based evaluation
3. Hard-negative mining loop

### Milestone C — Optional Transformer "precision booster"

1. Train teacher Transformer for labeling/distillation
2. Train tiny student Transformer (if still needed)
3. Export to ONNX + int8 quantization
4. Browser WebGPU path (ORT WebGPU EP)
5. Gate to Transformer only on borderline cases

### Milestone D — Release engineering

- Version model files + rule lists
- Deterministic builds
- Benchmark harness:
  - Latency per post
  - CPU %
  - Peak RSS / browser memory
- Regression test set of "never hide" clean posts (to guard against FP spikes)

---

## Summary: Minimal SOTA Stack

### Training

| Component | Technology |
|-----------|------------|
| Stage 1 | fastText supervised (word + char n-grams) |
| Teacher (offline only) | Large Transformer encoder |
| Distillation | Teacher → fastText + hard negatives |
| Optional Stage 2 | Tiny Transformer fine-tune + ONNX export + quantization |
| Optimization tooling | Olive for ONNX workflows |

### Inference

| Component | Technology |
|-----------|------------|
| Browser Stage 1 | fastText WASM in WebWorker |
| Browser Stage 2 (optional) | ONNX Runtime Web (WebGPU if available; WASM fallback) |
| Optional wrapper | Transformers.js v3 |
| CPU throttling | `env.wasm.numThreads` in ORT Web |

This gives you an extremely cheap always-on filter that still has a path to "near-teacher" quality via distillation, and a selective Transformer fallback for the hardest cases.
