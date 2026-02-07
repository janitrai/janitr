# Data Mixing Strategy

> Target mixing ratios and sampling strategy for training multi-label classifiers on the Janitr label set.

## Principles

1. The dataset preserves **full multi-label ground truth** (100+ labels). Mixing and collapsing happens at training time, not at the data layer.
2. Balance by **label-occurrences**, not by posts — one post can carry multiple labels.
3. Use **per-label count floors and caps** rather than fixed percentage splits across labels, because the label distribution is long-tailed.

## Overall Mix

For training a general multi-label classifier:

| Bucket               | Share of total |
| -------------------- | -------------- |
| Clean (no positive)  | 70–85%         |
| Positives (≥1 label) | 15–30%         |

The model must learn "most things are fine" (controls false positives) while still seeing enough positive examples per label.

## Per-Label Targets

Inside the positive bucket, split by **count floors and caps**, not percentages:

| Tier | Description | Target per label      |
| ---- | ----------- | --------------------- |
| Head | Very common | Cap at 5–10% of total |
| Mid  | Moderate    | 3,000–10,000 examples |
| Tail | Rare        | 500–3,000 examples    |

**Rule of thumb:**

- **≥2,000 per label** — minimum viable for fastText-like models
- **≥10,000 per label** — performance becomes noticeably stable

## Hard Negatives

20–40% of the clean bucket should be **hard negatives**: posts that previously triggered false positives. This directly reduces FPR.

## Multi-Label Sampling

Because posts carry multiple labels:

- Sample to balance **label-occurrences** across labels after applying caps and floors
- Don't balance by raw post count — a post with 3 labels contributes to 3 label counts

## Evaluation / Threshold-Tuning Set

The eval set must reflect **realistic base rates**, not training proportions:

| Bucket        | Share  |
| ------------- | ------ |
| Clean         | 90–98% |
| All positives | 2–10%  |

Spread positive examples across labels as observed in the wild. This ensures thresholds tuned on the eval set transfer to production.

## Separate Models Per Label Group

**Recommended**: train separate models per label group rather than one monolithic classifier.

| Model               | Label group                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| Security & fraud    | scam, phishing, malware, impersonation, fake_support, recovery_scam, etc.   |
| Spam & manipulation | spam, reply_spam, promo, affiliate, engagement_bait, bot, etc.              |
| AI content          | ai_generated, ai_generated_reply, ai_slop, content_farm                     |
| Info integrity      | misinformation, civic_misinfo, manipulated_media, conspiracy, pseudoscience |
| Safety & sensitive  | hate, harassment, threat_violence, graphic_violence, etc.                   |
| Topics              | topic_crypto, topic_politics, topic_news, topic_ai, etc.                    |

Benefits:

- Each model has a simpler classification task
- Thresholds can be tuned independently per group
- Easier to satisfy per-label count minimums within each model
- Groups align with the label taxonomy in [LABELS.md](LABELS.md)

The same mixing targets apply per model, but become easier to satisfy with smaller focused datasets.

## Current State (Feb 2026)

~2,900 samples total. Most labels are well below the 2,000-per-label floor:

| Label        | Count | Status           |
| ------------ | ----- | ---------------- |
| clean        | 1,344 | Below floor      |
| topic_crypto | 1,334 | Below floor      |
| scam         | 568   | Below floor      |
| spam         | 478   | Below floor      |
| promo        | 362   | Below floor      |
| topic_news   | 354   | Below floor      |
| topic_ai     | 227   | Below floor      |
| affiliate    | 139   | Below floor      |
| phishing     | 111   | Below floor      |
| All others   | <100  | Well below floor |

**Priority**: scale up data collection before this mixing strategy can be fully applied. The current 3-class fastText model (scam, topic_crypto, clean) is a pragmatic interim approach.

## Next Steps

1. Scale data collection to reach ≥2,000 per label for head/mid labels
2. Build hard negative pipeline (collect and label false positives from production)
3. Experiment with separate models per label group
4. Implement label-occurrence-balanced sampling in `scripts/prepare_data.py`
