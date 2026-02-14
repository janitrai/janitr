---
date: 2026-02-14
author: Bob <bob@dutifulbob.com>
title: Tiny Transformer Expanded Holdout Retrain Report
tags: [janitr, transformer, distillation, evaluation, holdout]
updated: 2026-02-14T13:10:00Z
---

# Tiny Transformer Expanded Holdout Retrain Report

## Scope

This run retrained the full teacher -> calibration -> logits cache -> student distillation -> ONNX -> int8 -> evaluation chain after expanding holdout size.

## Data Split Change

Rebuilt stratified splits from `data/sample.jsonl` with:

- train: 60%
- valid: 10%
- calib: 5%
- holdout: 25%

Prepared counts:

- train: 2566 (scam 587, clean 1216, topic_crypto 763)
- valid: 428 (scam 98, clean 203, topic_crypto 127)
- holdout: 1069 (scam 245, clean 506, topic_crypto 318)

## Training Configuration

- Teacher: `cardiffnlp/twitter-roberta-large-2022-154m`
- Teacher seeds: `13,42,7`
- Runtime: CUDA (GPU)
- DAPT: not used in this run

Pipeline commands executed in order:

1. `uv run --project scripts python scripts/train_transformer_teacher.py --train data/transformer/train.prepared.jsonl --valid data/transformer/valid.prepared.jsonl --holdout data/transformer/holdout.prepared.jsonl --output-dir models/student_experiments/2026-02-14-retrain-expanded-holdout/teacher --seeds 13,42,7`
2. `uv run --project scripts python scripts/calibrate_teacher.py --prepared-valid data/transformer/valid.prepared.jsonl --preds models/student_experiments/2026-02-14-retrain-expanded-holdout/teacher_valid_preds.jsonl --out models/student_experiments/2026-02-14-retrain-expanded-holdout/teacher_calibration.json --out-calibrated-preds models/student_experiments/2026-02-14-retrain-expanded-holdout/teacher_valid_preds_calibrated.jsonl`
3. `uv run --project scripts python scripts/cache_teacher_logits.py --train data/transformer/train.prepared.jsonl --valid data/transformer/valid.prepared.jsonl --teacher-dir models/student_experiments/2026-02-14-retrain-expanded-holdout/teacher --calibration models/student_experiments/2026-02-14-retrain-expanded-holdout/teacher_calibration.json --seeds 13,42,7 --train-out models/student_experiments/2026-02-14-retrain-expanded-holdout/teacher_logits_train.npz --valid-out models/student_experiments/2026-02-14-retrain-expanded-holdout/teacher_logits_valid.npz`
4. `uv run --project scripts python scripts/train_transformer_student_distill.py --train data/transformer/train.prepared.jsonl --valid data/transformer/valid.prepared.jsonl --holdout data/transformer/holdout.prepared.jsonl --cache-train models/student_experiments/2026-02-14-retrain-expanded-holdout/teacher_logits_train.npz --cache-valid models/student_experiments/2026-02-14-retrain-expanded-holdout/teacher_logits_valid.npz --output-dir models/student_experiments/2026-02-14-retrain-expanded-holdout/student`
5. `uv run --project scripts python scripts/export_transformer_student_onnx.py --student-dir models/student_experiments/2026-02-14-retrain-expanded-holdout/student --train data/transformer/train.prepared.jsonl --valid data/transformer/valid.prepared.jsonl --out models/student_experiments/2026-02-14-retrain-expanded-holdout/student.onnx`
6. `uv run --project scripts python scripts/quantize_transformer_student.py --input models/student_experiments/2026-02-14-retrain-expanded-holdout/student.onnx --output models/student_experiments/2026-02-14-retrain-expanded-holdout/student.int8.onnx --student-dir models/student_experiments/2026-02-14-retrain-expanded-holdout/student --valid data/transformer/valid.prepared.jsonl`
7. `uv run --project scripts python scripts/evaluate_transformer.py --student-dir models/student_experiments/2026-02-14-retrain-expanded-holdout/student --train data/transformer/train.prepared.jsonl --valid data/transformer/valid.prepared.jsonl --holdout data/transformer/holdout.prepared.jsonl --thresholds-out models/student_experiments/2026-02-14-retrain-expanded-holdout/thresholds.transformer.json --out models/student_experiments/2026-02-14-retrain-expanded-holdout/student_holdout_eval.json`

## Teacher Results (Expanded Holdout)

From `models/student_experiments/2026-02-14-retrain-expanded-holdout/teacher/training_summary.json` (ensemble):

- scam precision: 0.8471
- scam recall: 0.8367
- scam F1: 0.8419
- scam FPR: 0.0449
- macro F1: 0.8673
- exact match: 0.8765

## Student Results (Expanded Holdout)

From `models/student_experiments/2026-02-14-retrain-expanded-holdout/student_holdout_eval.json`:

- tuned thresholds:
  - scam: 0.69
  - topic_crypto: 0.62
- scam precision: 0.9383
- scam recall: 0.6204
- scam F1: 0.7469
- scam FPR: 0.0121
- topic F1: 0.7695
- macro F1: 0.7967
- exact match: 0.8176
- scam PR-AUC: 0.8744
- scam calibration ECE: 0.0368

## Artifacts

Run root: `models/student_experiments/2026-02-14-retrain-expanded-holdout`

- teacher checkpoints and summary: `teacher/`
- calibration file: `teacher_calibration.json`
- logits cache: `teacher_logits_train.npz`, `teacher_logits_valid.npz`
- student checkpoint: `student/pytorch_model.bin` (~16 MB)
- ONNX fp32: `student.onnx` (~13 MB)
- ONNX int8: `student.int8.onnx` (~3.4 MB)
- evaluation: `student_holdout_eval.json`
- tuned thresholds: `thresholds.transformer.json`

## Comparison Note

The previous baseline report used a much smaller holdout (214 rows, scam support 50). This run uses holdout 1069 rows (scam support 245). Because the evaluation set changed, direct metric deltas versus earlier runs are not apples-to-apples.

## Outcome

- Full GPU pipeline ran end-to-end with the required teacher and 3 seeds.
- Required artifacts were produced.
- Holdout evaluation report for the expanded-holdout run is captured in this file.
