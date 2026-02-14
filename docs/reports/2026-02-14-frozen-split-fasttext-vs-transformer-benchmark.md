---
date: 2026-02-14
author: Bob <bob@dutifulbob.com>
title: Frozen Split Benchmark - fastText vs Transformer
tags: [janitr, benchmark, fasttext, transformer, evaluation]
updated: 2026-02-14T15:20:00Z
---

# Frozen Split Benchmark - fastText vs Transformer

## Goal

Run an apples-to-apples benchmark on a fixed split and remove leakage confusion from prior comparisons.

## Frozen Benchmark Split

Benchmark ID: `2026-02-14_expanded_holdout`

Artifacts:

- `dataset/benchmarks/2026-02-14_expanded_holdout/MANIFEST.json`
- `dataset/benchmarks/2026-02-14_expanded_holdout/train.ids.txt`
- `dataset/benchmarks/2026-02-14_expanded_holdout/valid.ids.txt`
- `dataset/benchmarks/2026-02-14_expanded_holdout/calib.ids.txt`
- `dataset/benchmarks/2026-02-14_expanded_holdout/holdout.ids.txt`

Split sizes:

- train: 2566
- valid: 428
- calib: 214
- holdout: 1069

Leakage checks:

- Current split internal overlaps are all zero (`train/valid/calib/holdout`).
- Script: `scripts/check_split_leakage.py`
- Reports:
  - `dataset/benchmarks/2026-02-14_expanded_holdout/leakage_report_current.json`
  - `dataset/benchmarks/2026-02-14_expanded_holdout/leakage_report_vs_previous.json`

Previous contamination evidence (holdout overlap with old split snapshot):

- old train overlap: 847
- old valid overlap: 72
- old holdout overlap: 10

## Rebuilds Run From Scratch

### fastText

- Train command:
  - `uv run --project scripts python scripts/train_fasttext.py --train data/train.txt --model-out models/benchmarks/2026-02-14_expanded_holdout/fasttext/scam_detector.bin`
- Threshold tuning:
  - `uv run --project scripts python scripts/tune_thresholds_fpr.py --model models/benchmarks/2026-02-14_expanded_holdout/fasttext/scam_detector.bin --data data/calib.txt --out models/benchmarks/2026-02-14_expanded_holdout/fasttext/thresholds.json --target-fpr 0.02 --labels topic_crypto,scam`
- Quantization (production profile):
  - `uv run --project scripts python scripts/reduce_fasttext.py --model models/benchmarks/2026-02-14_expanded_holdout/fasttext/scam_detector.bin --valid data/valid.txt --cutoff 1000 --dsub 8 --out models/benchmarks/2026-02-14_expanded_holdout/fasttext/scam_detector.ftz`
- Holdout eval artifact:
  - `models/benchmarks/2026-02-14_expanded_holdout/fasttext/holdout_eval.json`
  - `models/benchmarks/2026-02-14_expanded_holdout/fasttext/holdout_eval_ftz.json`

### Transformer (teacher -> student)

- Teacher: `cardiffnlp/twitter-roberta-large-2022-154m`
- Seeds: `13,42,7`
- Run root: `models/benchmarks/2026-02-14_expanded_holdout/transformer`
- Full chain completed:
  - teacher train
  - calibrate teacher
  - cache teacher logits
  - distill student
  - export ONNX
  - int8 quantize
  - evaluate torch + int8 onnx
- Holdout eval artifacts:
  - `models/benchmarks/2026-02-14_expanded_holdout/transformer/student_holdout_eval.json`
  - `models/benchmarks/2026-02-14_expanded_holdout/transformer/student_holdout_eval_int8.json`

## Side-by-Side (Same Holdout)

Operating points used:

- fastText (`.ftz`) thresholds: `scam=0.8439`, `topic_crypto=0.9957`, `clean=0.10`
- Transformer thresholds: `scam=0.87`, `topic_crypto=0.50`

Metrics:

- scam precision: fastText-ftz `0.9128` vs transformer-int8 `0.9375`
- scam recall: fastText-ftz `0.5551` vs transformer-int8 `0.6122`
- scam F1: fastText-ftz `0.6904` vs transformer-int8 `0.7407`
- scam FPR: fastText-ftz `0.0158` vs transformer-int8 `0.0121`
- topic_crypto F1: fastText-ftz `0.8258` vs transformer-int8 `0.7806`
- macro F1: fastText-ftz `0.7838` vs transformer-int8 `0.8013`
- exact match: fastText-ftz `0.7624` vs transformer-int8 `0.8241`

Result: transformer student is better on the primary scam operating point (higher precision, higher recall, lower FPR) and overall macro/exact-match. fastText-ftz is stronger only on topic_crypto F1.

## Artifact Sizes

- fastText `.bin`: `770 MB` (`models/benchmarks/2026-02-14_expanded_holdout/fasttext/scam_detector.bin`)
- fastText `.ftz` (cutoff=1000,dsub=8): `123 KB` (`models/benchmarks/2026-02-14_expanded_holdout/fasttext/scam_detector.ftz`)
- transformer student checkpoint: `16 MB`
- transformer ONNX fp32: `13 MB`
- transformer ONNX int8: `3.4 MB`

## Conclusion

- The benchmark is now reproducible and leakage-guarded.
- Comparison is no longer confounded by split reshuffling.
- On this fixed split, transformer student (int8 ONNX) outperforms fastText across key holdout metrics.
