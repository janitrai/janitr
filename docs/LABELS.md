# Label Guide

> **Janitr Label Set v2026** — comprehensive, multi-label-friendly, aligned with X's rule structure.

## How to use these labels

- **Multi-label**: apply _all_ labels that match a post.
- **`clean` is exclusive**: use `clean` only if **no other label applies**.
- **Separate "topic" vs "behavior"**: e.g., a crypto scam is `crypto` + `scam` (+ often `impersonation`, `manipulated_media`, etc.).
- If you need to reduce complexity later, you can collapse labels by group (e.g., treat `reply_spam` as `spam`).

## Canonical labels (grouped)

```yaml
# Janitr label set v2026 (X-focused)

base:
  - clean

security_fraud:
  - scam
  - phishing
  - malware
  - impersonation
  - fake_support
  - recovery_scam
  - job_scam
  - romance_scam
  - account_compromise

spam_manipulation_commercial:
  - spam
  - reply_spam
  - dm_spam
  - promo
  - affiliate
  - lead_gen
  - engagement_bait
  - follow_train
  - giveaway
  - platform_manipulation
  - astroturf
  - bot
  - ai_generated
  - ai_slop
  - content_farm
  - copypasta
  - stolen_content
  - clickbait
  - low_effort
  - vaguepost
  - ragebait

information_integrity:
  - misinformation
  - civic_misinfo
  - manipulated_media
  - conspiracy
  - pseudoscience

safety_sensitive:
  - hate
  - harassment
  - threat_violence
  - violent_extremism
  - graphic_violence
  - self_harm
  - adult_nudity
  - nonconsensual_nudity
  - child_exploitation
  - illegal_goods
  - privacy_doxxing
  - profanity

topic_filters_optional:
  - crypto
  - politics
  - gambling
  - finance
  - health
  - adult_services
  - religion
  - celebrity
  - sports
  - language_other
```

> **Backbone alignment note**: many of these map cleanly onto X's own "Safety / Privacy / Authenticity" rule headings (e.g., violent content, child safety, abuse/harassment, hateful conduct, suicide/self-harm, adult content, illegal goods, private information, platform manipulation/spam, civic integrity, misleading identities, synthetic/manipulated media, etc.).

---

## Operational definitions

Tight, labeler-friendly definitions. Intentionally "binary-ish" to reduce subjectivity.

### Base

#### `clean`

None of the other labels apply.

---

### Security & fraud

#### `scam`

Tries to trick users into losing money/assets or taking a clearly fraudulent action (giveaway doubling, fake investment returns, fake invoices, etc.).

#### `phishing`

Attempts to capture credentials/OTP/recovery codes or send you to a fake login/verification flow.

#### `malware`

Attempts to get the user to install/run something or click a download/exploit link (cracked software, "APK", "driver update", etc.).

#### `impersonation`

Pretends to be a person/org (brand, government, creator) to mislead/confuse. If for fraud, also apply `scam`.

#### `fake_support`

A common X-specific impersonation subtype: "support" replies targeting complaint tweets, pushing DMs, links, or credential capture (often `phishing` + `impersonation`).

#### `recovery_scam`

Claims it can recover lost funds (esp. crypto) for an upfront fee / DM.

#### `job_scam`

Fake recruiting, "remote job, instant pay," suspicious hiring funnels.

#### `romance_scam`

Relationship-building with eventual money/crypto request (often in replies or DMs).

#### `account_compromise`

Explicit hacking/account takeover attempts or evidence a hacked account is being used for spam/scams (can be inferred by sudden theme change + scam links, but label conservatively).

---

### Spam, manipulation, and commercial noise

These cover what users complain about most on X day-to-day: reply spam, bot swarms, engagement farming, and AI slop.

#### `spam`

Unsolicited junk, repetitive templates, irrelevant promos, link spam; includes porn-bot style spam if you want a single bucket.

#### `reply_spam`

Spam whose _primary_ form is replies ("great post!", single emoji, generic praise, repeated templated replies, etc.). Distinct enough on X to warrant its own label.

#### `dm_spam`

DM spam (keep for future; useful for dataset continuity).

#### `promo`

Marketing/self-promotion (products, services, newsletters, "buy my course"), including astroturf-y native ads. If coordinated, add `platform_manipulation`/`astroturf`.

#### `affiliate`

Referral codes/affiliate links (can coexist with `promo`).

#### `lead_gen`

Funnel mechanics: "DM me 'GUIDE'", "comment 'X' and I'll send it", gated lead magnets.

#### `engagement_bait`

Explicit "like/RT/comment to…" prompts, forced-choice image bait, "tag 3 friends," "vote in poll," etc.

#### `follow_train`

Follow-for-follow, mutual trains, "gain followers" chains.

#### `giveaway`

Contests/giveaways even if legitimate. If fraudulent, also apply `scam`.

#### `platform_manipulation`

Explicit attempts to game ranking/reach (coordinated boosts, brigading instructions, "mass report," "block campaign," etc.).

#### `astroturf`

Coordinated inauthentic persuasion campaigns (can be human-led or bot-led). Use when there's clear coordination signals (copy/paste scripts, "talking points," synchronized posting).

#### `bot`

Automation signals dominate (high volume, templated replies, unnatural repetition).

#### `ai_generated`

Looks AI-written/AI-made (even if a human posted it).

#### `ai_slop`

Low-quality, high-volume AI output optimized for engagement/monetization. "AI slop" is now a widely used term for this category.

#### `content_farm`

Mass-produced content (AI or human) where _volume + sameness_ is the point (thread mills, clip mills, scraped summaries).

#### `copypasta`

Chain-letter style templates; repeated meme text; "post this or…" formats.

#### `stolen_content`

Uncredited reposts / content theft (maps to copyright/trademark issues).

#### `clickbait`

Intentionally misleading hook/headline; mismatch between promise and payload.

#### `low_effort`

Extremely low-signal posts ("this", "lol", emoji-only, "GM", generic praise).

#### `vaguepost`

Intentionally ambiguous personal drama / attention bait.

#### `ragebait`

Inflammatory framing designed to provoke outrage and engagement; may or may not be fact-based.

---

### Information integrity

#### `misinformation`

Materially false or unverified claims presented as true. Inaccuracy is a top complaint among people who get news on social media.

#### `civic_misinfo`

Misinformation about elections/civic processes (how/when/where to vote; participation suppression). Explicitly aligned with X civic integrity rules.

#### `manipulated_media`

Deepfakes, deceptively edited media, or out-of-context media presented to mislead. Includes AI-generated synthetic media.

#### `conspiracy`

Claims centered on secret plots and non-falsifiable narratives (often overlaps with `misinformation`).

#### `pseudoscience`

Health/science content that contradicts established evidence (often overlaps with `misinformation` and `health`).

---

### Safety & sensitive content

These align closely with X's own rule categories.

#### `hate`

Attacks on protected categories (race, religion, sexuality, etc.).

#### `harassment`

Targeted harassment, bullying, dogpiling incitement.

#### `threat_violence`

Threats, incitement, glorification/desire for violence.

#### `violent_extremism`

Support/affiliation with violent & hateful entities; terrorist propaganda; perpetrators/manifestos.

#### `graphic_violence`

Gore / graphic injury media (even if "newsworthy," users often want it filtered).

#### `self_harm`

Encourages/promotes self-harm or suicide.

#### `adult_nudity`

Consensual adult nudity/sexual behavior (NSFW).

#### `nonconsensual_nudity`

Intimate imagery without consent, including AI "nudification" and sexual deepfakes. Active 2026 regulatory and safety concern.

#### `child_exploitation`

Any child sexual exploitation content. **For safety: don't collect/store image examples; prefer text-only references or synthetic placeholders with no illegal content.**

#### `illegal_goods`

Sale/facilitation of illegal or certain regulated goods/services (drugs, weapons, etc.).

#### `privacy_doxxing`

Posting private info (address/phone), or threats/incentives to expose it.

#### `profanity`

Profanity/obscenity filter separate from harassment/hate (useful for "clean feed" mode).

---

### Topic filters (optional)

These are "user preference" filters, not moral judgments:

#### `crypto`

Crypto-related content. Any mention of a specific coin, token, NFT, wallet, exchange, DeFi, or blockchain references.

#### `politics`

Political content, partisan discussion, election-related posts.

#### `gambling`

Gambling, betting, casino content.

#### `finance`

Financial advice, stock tips, investment discussion (non-crypto).

#### `health`

Health-related content, medical advice, wellness.

#### `adult_services`

Adult services promotion (distinct from `adult_nudity`).

#### `religion`

Religious content, proselytizing.

#### `celebrity`

Celebrity gossip, fan content, parasocial posts.

#### `sports`

Sports content, game discussion, team fandom.

#### `language_other`

Non-English content (for users who want English-only feeds).

---

## Example multi-label combos

What "comprehensive" buys you:

| Example post                                                               | Labels                                                     |
| -------------------------------------------------------------------------- | ---------------------------------------------------------- |
| "Elon giveaway — send 0.1 BTC get 0.2 back" + deepfake clip                | `scam` + `crypto` + `impersonation` + `manipulated_media`  |
| Blue-check reply: "Amazing post! 🚀 DM me for business"                    | `reply_spam` + `promo` + `lead_gen` (+ `bot` if automated) |
| AI thread mill: "I analyzed 10,000 CEOs…" (obvious LLM cadence)            | `ai_slop` + `content_farm` + `clickbait`                   |
| Election suppression: "Polling stations are closed tomorrow; vote by text" | `civic_misinfo` + `misinformation` + `politics`            |
| Non-consensual sexual deepfake "nudification"                              | `nonconsensual_nudity` + `manipulated_media`               |

---

## Multi-label rules

1. Use multiple labels when **both are true** (e.g. `crypto` + `promo`).
2. `clean` should be **exclusive** (do not combine with other labels).
3. `scam` does not imply `crypto`; add `crypto` only when the topic is crypto.
4. Topic labels (`crypto`, `politics`, etc.) can combine with any behavior label.

---

## Training notes

- A 50–60 label ontology is feasible as a _schema_, but you'll likely want to:
  - Train a **coarse model** first (top ~15–25 labels)
  - Add **specialists** (rules/regex or smaller sub-models) for things like `phishing`, `crypto`, `adult_services`, `doxxing`, etc.
- That keeps model size small while still letting the product present a comprehensive set of toggles.
- You can cluster labels during training (merge into coarse super-classes) to improve performance, while keeping the dataset labels fine-grained for future remapping.

---

## Data shape

Each record is JSONL with at minimum:

- `id`: unique string
- `platform`: x | discord | web | dm | other
- `source_id`: platform-native id (tweet id, message id, etc.)
- `source_url`: canonical URL when available
- `collected_at`: ISO timestamp
- `text`: raw text (preserve exactly; do not truncate)
- `labels`: array of strings (one or more labels from this guide)

Optional fields:

- `urls`: extracted URLs
- `addresses`: extracted wallet addresses
- `notes`: short rationale
