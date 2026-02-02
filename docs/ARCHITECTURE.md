# InternetCondom Architecture

## 1) Problem Statement
X (Twitter) is the primary distribution channel for crypto scams: shill posts, rug-pull hype, impersonators, fake airdrops, wallet drainers, and malicious links. These scams move fast, exploit platform virality, and blend into legitimate crypto chatter. InternetCondom is a browser extension that detects scam content in X feeds and hides or warns users in real time. The system must be low-latency, resilient to adversarial wording, and continuously adaptable as scam tactics evolve.

## 2) Current MVP Architecture (fastText + Rules)
The MVP uses a two-layer, on-device approach:

- **Input:** Tweet text (plus minimal metadata when available).
- **Stage A: fastText classifier**
  - Lightweight supervised model for quick text-only scoring.
- **Stage B: Rules engine**
  - Keyword patterns, URL heuristics, and hard-coded red flags.
- **Decision:** If either stage flags risk above threshold, hide or warn.
- **Output:** UI badge + optional hide in feed.

Why this works now: it is fast, cheap to iterate, and runs locally. Why it will not scale: it is brittle to obfuscation, has limited context, and cannot generalize to evolving scam campaigns.

## 3) Target Production Architecture (Multi-Stage Pipeline)
We are moving toward a modular, multi-stage pipeline that mixes fast heuristics, learned models, and campaign-level intelligence. The goal is to reduce false positives while catching coordinated scam clusters early.

**Pipeline overview:**

1. **Ingestion + Snapshotting**
   - Capture tweet text, author profile, embedded media, URLs, engagement state.
   - Normalize and hash identifiers (privacy-safe).

2. **Cheap Filters (Stage 0)**
   - URL allow/deny lists, obvious keyword patterns, known scam domains.
   - Purpose: fast reject/accept, reduce downstream load.

3. **Content Scoring (Stage 1)**
   - Text model (fastText or replacement) for scam-likeness.
   - Optional language detection + normalization.

4. **Entity + Link Enrichment (Stage 2)**
   - Resolve short links, check domain age/reputation.
   - Identify crypto tickers, contract addresses, wallet formats.

5. **Behavioral + Network Signals (Stage 3)**
   - Profile age, follower/following ratios, churn.
   - Engagement anomalies (sudden likes/retweets), bot-like patterns.
   - Cross-post similarity and coordinated campaigns.

6. **Risk Aggregation (Stage 4)**
   - Weighted fusion of content, URL, profile, and behavior scores.
   - Calibrated thresholds for warn vs hide.

7. **Feedback + Learning Loop**
   - User feedback ("safe" / "scam") and telemetry.
   - Active learning to prioritize labeling of uncertain samples.
   - Regular model refreshes with adversarial examples.

8. **Monitoring + Auditability**
   - Explainable reasons attached to warnings.
   - Drift detection on features and scam patterns.

This architecture supports fast local decisions with optional cloud enrichment when needed.

## 4) X-Specific Signals We Can Extract
We should aggressively exploit X-specific context because scam signals are rarely just text.

**Tweet-level:**
- Text, emojis, hashtags, cashtags, mentions.
- Calls-to-action ("claim", "airdrop", "mint", "DM", "connect wallet").
- Media type (image/video/GIF), repeated meme templates.
- Quoted/replied-to tweets and conversation context.

**Profile-level:**
- Account age, verified status, bio content.
- Name/handle similarity to known brands.
- Profile image reuse or stock image fingerprints.
- Follower/following ratio and growth rate.

**Engagement-level:**
- Like/retweet velocity spikes.
- Engagement sourced from low-quality/bot accounts.
- Reply patterns (copy-paste shills, identical comments).

**URL + Off-platform:**
- Shortener expansions, redirect chains.
- Domain age, registrar, known scam hosting.
- Link to wallet connect, airdrop claim, or mint pages.

**Campaign-level:**
- Near-duplicate text across many posts.
- Coordinated posting within tight time windows.
- Shared domains or wallet addresses across accounts.

## 5) Phased Migration Plan (MVP → Production)

**Phase 0: Stabilize MVP (Now)**
- Improve training data quality, tighten rules, and add minimum telemetry.
- Establish evaluation benchmarks (precision/recall on scam vs legit).

**Phase 1: Instrumentation + Data Foundation**
- Build a consistent data model for tweets, profiles, URLs.
- Start logging feature snapshots for later offline training.
- Add an internal label tool or workflow.

**Phase 2: Multi-Stage Architecture (Local First)**
- Introduce stage boundaries and score fusion.
- Add link expansion + domain reputation lookups.
- Keep everything local when possible for latency.

**Phase 3: Campaign Detection + Behavioral Signals**
- Add clustering of near-duplicate posts.
- Profile-based heuristics and bot-likeness models.
- Expand rules into data-driven heuristics.

**Phase 4: Online Learning + Personalization**
- Use user feedback and weak labels to update models.
- Risk calibration by user tolerance (warn vs hide).

## 6) Design Principles

- **Modular by default:** every stage is swappable (model, rules, enrichment). No single component should be a choke point.
- **Low-latency first:** fast local decisions; cloud only when value is clear.
- **Explainable outcomes:** show users why a post was flagged.
- **Defense in depth:** layered signals reduce single-point failure.
- **Adversarial resilience:** assume obfuscation; build for evasion.
- **Privacy-respecting:** minimal data retention, hashed identifiers.
- **Measurable progress:** every stage has its own metrics and tests.

---

This document is the north star. It is deliberately opinionated: ship fast detection locally, enrich when needed, and grow toward campaign-level intelligence without sacrificing speed or user trust.
