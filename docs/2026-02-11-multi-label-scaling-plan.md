---
date: 2026-02-11
author: Bob <bob@dutifulbob.com>
title: Multi Label Scaling Plan
tags: [ml, classification, architecture]
---

# Multi-Label Scaling Plan (Plain Language)

## Why this doc exists

Janitr works today with 3 training classes (`scam`, `topic_crypto`, `clean`).
The next stage will add many labels across many topics.
If we keep the current setup as-is, quality and calibration will become unstable.

This document explains, in plain language, what we should change and why.

## Plain-language summary

- Keep scam/safety detection separate from topic detection.
- Predict topics independently (multi-label), not as one forced bucket.
- Give each label its own threshold and quality target.
- Split datasets so train/validation/holdout have similar label co-occurrence.
- Evaluate every label separately, especially rare labels.

## What we learned from recent experiments

From `docs/reports/experiments/2026-02-11/`:

- Single-stage model improved with tuning, but could not hit low scam FPR on holdout.
- Two-stage prototype worked much better than single-stage.
- Best two-stage low-FPR point reached roughly:
  - `scam` FPR: `0.05`
  - `scam` precision: `~0.85`
  - `micro-F1`: `~0.89`
- Current split construction is mismatched:
  - `valid.txt` had no `scam+topic_crypto` overlap.
  - `calib.txt` and `holdout.txt` had large overlap.
  - This makes threshold tuning transfer poorly.

## Target architecture for many labels

### 1) Safety head (strict)

A dedicated classifier for harmful behavior:

- scam
- phishing
- impersonation
- malware links
- spam abuse

Goal: minimize false positives for high-impact flags while keeping acceptable recall.

### 2) Topic head (broad multi-label)

Separate model(s) to predict topics:

- crypto, stocks, politics, sports, etc.
- allow multiple topics per post

Goal: broad topical coverage without interfering with safety decisions.

### 3) Policy layer

Combine outputs from safety + topic heads using per-label thresholds:

- `hide` rules for high-confidence safety labels
- `warn` rules for medium-confidence safety labels
- `badge` rules for topic labels

## Data strategy for taxonomy growth

### Keep full ground truth

Do not discard fine labels in source data.
Always keep full taxonomy in JSONL.

### Train with task views

Generate task-specific views from the same source:

- safety view
- topic view
- optional sub-domain views

### Fix split strategy

Use co-occurrence-aware stratification so train/valid/calib/holdout have similar distributions:

- label frequencies
- label co-occurrence pairs
- source/platform slices
- time buckets

## Evaluation strategy (must-have)

For each label:

- precision
- recall
- FPR
- support

For full model:

- micro/macro F1
- PR-AUC for imbalanced labels
- calibration checks by label

Release gates should be per-label, not just one global metric.

## Thresholding strategy for many labels

- No single global threshold.
- Use per-label thresholds tuned on `calib` only.
- Validate once on unseen holdout.
- Track drift and recalibrate periodically.

Example policy:

- safety labels: strict FPR budgets (`<= 2%` or label-specific)
- topic labels: looser budgets where recall matters more

## Rollout plan

### Phase 0: Foundation

- finalize taxonomy schema
- rebuild split pipeline with co-occurrence constraints
- add per-label dashboards

### Phase 1: Two-head baseline

- train safety head + topic head
- ship policy layer with per-label thresholds
- validate on unseen holdout

### Phase 2: Scale labels

- add new topic labels in batches
- require minimum support or weak-label bootstrapping
- monitor per-label degradation

### Phase 3: Continuous improvement

- active error mining by label
- hard-negative and hard-positive loops
- periodic retrain + recalibration

## Immediate next actions

1. Rebuild train/valid/calib/holdout with co-occurrence-aware splitting.
2. Productize two-stage inference contract in extension/offscreen pipeline.
3. Define per-label quality budgets (especially for safety labels).
4. Add evaluation report generation for all labels.
5. Start iterative error mining per new label batch.

## Success criteria

- Safety labels meet strict false-positive budgets on holdout.
- Topic labels maintain useful recall with acceptable precision.
- New labels can be added without destabilizing existing ones.
- Holdout results remain consistent across retrains.
