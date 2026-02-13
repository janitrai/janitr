---
date: 2026-02-13
author: Bob <bob@dutifulbob.com>
title: Tiny Transformer Implementation Plan
tags: [janitr, scam_classifier, transformer, onnx, extension]
updated: 2026-02-13T08:08:20Z
---

2026-02-13T08:08:20Z

# Tiny Transformer Option Plan (Janitr)

## Goal

Add an optional tiny-transformer inference path for Janitr's existing classification task (`scam`, `topic_crypto`, `clean`) in the browser extension, while keeping the current fastText path as default.

Target outcomes:

- Keep false positives low (`FPR <= 2%` on holdout).
- Keep local inference practical in extension runtime.
- Keep model artifacts small enough for extension delivery.

## Current Repo Baseline (Grounding)

- Existing production path is fastText in browser offscreen flow:
  - `extension/background.js`
  - `extension/offscreen.js`
  - `extension/fasttext/scam-detector.js`
- Current training scripts are fastText-focused:
  - `scripts/prepare_data.py`
  - `scripts/train_fasttext.py`
  - `scripts/evaluate.py`
- Current JSONL dataset snapshot for transformer work:
  - `data/train.jsonl`: `3421` rows
  - `data/valid.jsonl`: `428` rows
  - `data/holdout.jsonl`: `214` rows
  - `data/sample.jsonl`: `4281` rows total
- Current training labels are multi-label at source and consolidated for production into:
  - `scam` (`783` in train)
  - `clean` (`1621` in train)
  - `topic_crypto` (`1636` in train)
- Expected data growth: `~4k` now to `100k-1M` over time.

## Target Architecture

### Student model (deployed)

Use a compact BERT-style encoder fine-tuned for Janitr's existing task.

- Layers: `4`
- Hidden size: `192`
- Attention heads: `4`
- FFN/intermediate size: `768`
- Max sequence length: `96`
- Dropout: `0.1`
- Tokenizer: WordPiece vocab `8192`
- Inference heads:
  - Head A: 2-class softmax (`scam` vs `clean`)
  - Head B: sigmoid topic head (`topic_crypto`, future topic labels)
- Final class decision for current production surface:
  - If `scam` exceeds threshold -> `scam`
  - Else if `topic_crypto` exceeds threshold -> `topic_crypto`
  - Else -> `clean`

Rationale:

- Fits short X post text and local browser constraints.
- Keeps parameter count in ~3-4M range, suitable for int8 ONNX deployment.
- Preserves compatibility with current Janitr collapse strategy while allowing topic-head expansion later.

### Teacher model (train-time only)

Use a larger tweet-native teacher for distillation and pseudo-labeling.

- Primary teacher: `cardiffnlp/twitter-roberta-large-2022-154m` (TimeLM 2022 Large).
  - Tweet-native pretraining on 2018-2022 data.
  - Fits Jetson-class training hardware with fp16 + gradient checkpointing.
- Alternative teacher: `vinai/bertweet-large`.
  - Strong TweetEval history and large tweet corpus.
  - Slightly older temporal coverage (through 2019).
- Multilingual option: `jhu-clsp/bernice` (skip unless multilingual support becomes a requirement).
- Teacher is train-time only and never shipped in extension.

Teacher scaling strategy:

- At current `~4k`: train 3 seeds of the same teacher and average logits for soft targets.
- At `100k-1M`: use a single teacher; ensemble no longer required for stability/cost.

## Tokenizer Choice

Use a custom WordPiece tokenizer trained on repo data.

- Vocabulary size: `8192`
- Normalization: NFKC + lowercase + zero-width cleanup (match behavior in `scripts/prepare_data.py`)
- Special tokens: `[PAD] [UNK] [CLS] [SEP] [MASK]`
- Reason: 8k vocab keeps embedding table small while still covering X slang/handles/URLs.

Tokenizer artifacts to ship with model:

- `tokenizer.json`
- `special_tokens_map.json`
- `tokenizer_config.json`
- `vocab.txt` (if needed by runtime tooling)

## Data Preparation Plan (Janitr labels and splits)

### 1) Build transformer prep script aligned with current datasets

Add `scripts/prepare_transformer_data.py` to produce model-ready JSONL from:

- `data/train.jsonl`
- `data/valid.jsonl`
- `data/holdout.jsonl`

Preparation rules:

- Preserve raw multi-label payload in output metadata.
- Generate collapsed teacher/student targets:
  - `y_scam_clean` for Head A (`scam` vs `clean`)
  - `y_topics` multi-hot vector for Head B (`topic_crypto`, optional future labels)
- Reuse normalization behavior from `scripts/prepare_data.py` (NFKC, zero-width cleanup, lowercase defaults).
- Keep holdout untouched (no pseudo-labeling, no oversampling).

### 2) Optional unlabeled corpus for DAPT

Create `scripts/build_unlabeled_corpus.py` to build MLM text for domain-adaptive pretraining (DAPT):

- Source from unlabeled X snapshots over the same pipeline as Janitr collection.
- Deduplicate and clean noisy boilerplate.
- Target initial corpus size: `50k-500k` texts; expand with dataset growth.

### 3) Split strategy

- Keep provided `train/valid/holdout` files as primary split source.
- For regenerated splits at scale, use time-aware partitioning and reduce handle leakage.
- Keep soft-label generation confined to train/valid; holdout remains hard-label only.

### 4) Data scale gates

- At `~4k`, allow transformer experiments but treat results as pre-scale.
- Promote to default-candidate training track once `>=100k` labeled rows are available.
- At `>=100k`, enable logit caching pipeline by default.

## Training Pipeline

### Phase A: DAPT (optional but recommended)

Add `scripts/train_teacher_dapt.py`:

1. Continue pretraining the chosen teacher with MLM on unlabeled tweet corpus.
2. Use short runs first (for example 10k-50k steps) and evaluate transfer gains.
3. Save adapted checkpoint under `models/teacher_dapt/`.

Output:

- `models/teacher_dapt/`

### Phase B: Teacher supervised training + calibration

Add `scripts/train_transformer_teacher.py`:

1. Fine-tune primary teacher (`cardiffnlp/twitter-roberta-large-2022-154m`) on Janitr train/valid.
2. Train Head A + Head B jointly with class-weighted objective.
3. At `~4k`, run 3 seeds and retain best checkpoints for ensemble logits.
4. Save predictions on valid/holdout for analysis.

Add `scripts/calibrate_teacher.py`:

1. Perform temperature scaling per label/head on validation split.
2. Save calibrated temperatures for soft-label generation.
3. Verify calibration with ECE/Brier diagnostics per label.

Output:

- `models/teacher/`
- `models/teacher_calibration.json`
- `models/teacher_valid_preds.jsonl`

### Phase C: Distillation to tiny student

Add `scripts/train_transformer_student_distill.py`:

- Initialize student with target architecture (4L/192H/4 heads).
- Train with mixed objective:
  - Hard-label loss + soft-label KL distillation.
  - Distillation temperature `T=2-4`.
  - `alpha=0.5` at `~4k` (hard/soft balanced), moving toward `alpha=0.8` at `100k+`.
- Add intermediate-layer distillation:
  - Project teacher hidden states to 192 via trainable projection.
  - Apply L2 matching loss on aligned intermediate layers.
- Use calibrated teacher outputs only (never raw logits) for soft labels.

At scale (`100k-1M`):

- Precompute and cache teacher logits to float16 with `scripts/cache_teacher_logits.py`.
- Train student from cached logits to reduce teacher forward-pass cost.

Output:

- `models/student/`
- `models/student_eval.json`

### Phase D: Quantization

Add `scripts/quantize_transformer_student.py`:

1. Export ONNX fp32 first.
2. Run dynamic int8 quantization.
3. Optionally run static int8 quantization with calibration set if dynamic int8 hurts FPR.

Keep both artifacts:

- `models/student.onnx` (fp32 baseline)
- `models/student.int8.onnx` (extension deployment candidate)

## ONNX Export Plan

Add `scripts/export_transformer_student_onnx.py` (or use Optimum CLI through wrapper script).

Implementation target:

- Opset: `17`
- Inputs: `input_ids`, `attention_mask`
- Dynamic axes for batch size
- Validate parity (`torch` vs `onnxruntime`) on at least 1k examples

Acceptance criteria for export:

- Mean absolute probability delta <= `0.01`
- Label agreement >= `99%` on validation slice

## Browser Extension Integration Plan

Keep fastText path unchanged; add transformer as optional mode.

### 1) New extension assets

Add under `extension/transformer/`:

- `student.int8.onnx`
- tokenizer files (`tokenizer.json`, etc.)
- `scam-detector-transformer.js` (ONNX runtime wrapper)

### 2) Offscreen inference integration

Update:

- `extension/offscreen.js` to dispatch by model mode:
  - existing `ic-infer-offscreen` fastText path
  - new transformer path (for example `ic-infer-offscreen` with `engine: "transformer"`)
- `extension/background.js` to forward engine selection and preserve backward-compatible messaging

### 3) Content script integration

Update `extension/content-script.js`:

- Add optional transformer detector pass for existing Janitr classes.
- Keep existing highlight behavior and thresholds semantics.
- Start with feature flag (`const ENABLE_TINY_TRANSFORMER = false`) and flip only after eval gates pass.

### 4) Manifest and resources

Update `extension/manifest.json` `web_accessible_resources` for transformer files and runtime assets.

### 5) Runtime dependency

Vendor ONNX Runtime Web assets into `extension/vendor/` (similar to existing fastText vendoring) to avoid remote fetches.

## Evaluation Strategy

### Offline metrics (required)

Add `scripts/evaluate_transformer.py` to report:

- Per-class Precision / Recall / F1 for `scam`, `topic_crypto`, `clean`
- Per-class FPR/FNR and support
- Exact-match accuracy
- Micro and macro Precision/Recall/F1
- PR-AUC for `scam`
- Calibration table (confidence bins)
- Metrics by subgroup:
  - short posts (< 40 chars)
  - with URL vs without URL
  - seen vs unseen handles

Primary release gates:

- `scam` FPR on holdout `<= 0.02`
- `scam` precision on holdout `>= 0.90`
- `scam` recall on holdout no worse than current fastText baseline by more than `2` points
- `topic_crypto` F1 non-regression vs fastText baseline

Threshold policy:

- Tune per-label thresholds with `scripts/tune_thresholds_fpr.py`.
- Store transformer thresholds under `config/thresholds.transformer.json`.
- Block promotion if thresholds cannot satisfy `scam` FPR gate.

### Browser/runtime metrics (required)

Add extension smoke benchmark similar to `extension/tests/wasm-smoke.spec.ts`:

- P50 latency <= `25 ms` per tweet (desktop)
- P95 latency <= `60 ms`
- No memory growth across 1,000 inferences in offscreen worker

### Shadow rollout plan

1. Ship behind feature flag, score-only mode (no UI action).
2. Compare transformer output vs existing fastText predictions on identical tweet batches.
3. Promote to warn/highlight mode only after 2 consecutive holdout runs pass gates.

## Estimated Model and Bundle Size

Estimated for student (4L, 192H, 4 heads, vocab 8k, seq 96):

- FP32 ONNX: ~`13-16 MB`
- INT8 ONNX: ~`3.3-4.2 MB`
- Tokenizer artifacts: ~`0.3-0.6 MB`
- Total model payload in extension: ~`3.6-4.8 MB`

Expected extension impact including ORT runtime assets:

- Additional shipped size: ~`4.8-6.2 MB`

This fits Janitr's stated browser model target range when using int8.

## Implementation Checklist (Executable)

### Milestone 1: Data + training scripts

- Create `scripts/prepare_transformer_data.py`
- Create `scripts/build_unlabeled_corpus.py`
- Create `scripts/train_teacher_dapt.py`
- Create `scripts/train_transformer_teacher.py`
- Create `scripts/calibrate_teacher.py`
- Create `scripts/train_transformer_student_distill.py`
- Create `scripts/cache_teacher_logits.py`
- Create `scripts/export_transformer_student_onnx.py`
- Create `scripts/quantize_transformer_student.py`
- Create `scripts/evaluate_transformer.py`

### Milestone 2: Model artifacts

- Train/calibrate teacher, generate soft labels
- Distill student with hidden-state matching
- Export + quantize ONNX
- Record metrics report under `docs/reports/`

### Milestone 3: Extension wiring

- Add `extension/transformer/scam-detector-transformer.js`
- Wire `background.js` + `offscreen.js` + `content-script.js`
- Add smoke/latency test in `extension/tests/`

### Milestone 4: Release decision

- Run full offline + extension benchmarks against current fastText baseline
- Confirm scam FPR gate and latency gate
- Enable feature flag by default only after both gates pass
