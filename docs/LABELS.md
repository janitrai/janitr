# Label Guide

> **Janitr Label Set v2026** — comprehensive, multi-label-friendly, aligned with X's rule structure.

## How to use these labels

- **Multi-label**: apply _all_ labels that match a post.
- **`clean`**: marks a post as benign/safe. Can be combined with topic labels (e.g., `clean` + `topic_crypto` = a normal crypto tweet).
- **Separate "topic" vs "behavior"**: e.g., a crypto scam is `topic_crypto` + `scam` (+ often `impersonation`, `manipulated_media`, etc.).
- If you need to reduce complexity later, you can collapse labels by group (e.g., treat `reply_spam` as `spam`).

## 3-class training scheme (fastText)

The raw dataset (`*.jsonl`) keeps the **full, multi-label taxonomy** from this document. For fastText training we collapse those raw labels into **3 mutually-exclusive training classes** during data preparation (`scripts/prepare_data.py`):

- `scam`
  - `phishing`, `malware`, `fake_support`, `recovery_scam`, `job_scam`, `romance_scam`, `impersonation`, `account_compromise`
  - `spam`, `reply_spam`, `dm_spam`
  - `promo`, `affiliate`, `lead_gen`, `engagement_bait`, `follow_train`, `giveaway`, `bot`
- `topic_crypto`
  - `topic_crypto` (only when no `scam`-bucket label is present)
- `clean`
  - everything else, including samples with empty labels or just `clean`

**Priority rule (when multiple raw labels are present):** `scam` > `topic_crypto` > `clean`.

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
  - ai_generated_reply
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
  # News and society
  - topic_news
  - topic_world_news
  - topic_local_news
  - topic_war_conflict
  - topic_crime_truecrime
  - topic_disasters_tragedy
  - topic_law_courts
  - topic_environment_climate
  - topic_social_issues

  # Politics and governance
  - topic_politics
  - topic_elections

  # Money and commerce
  - topic_finance
  - topic_investing
  - topic_personal_finance
  - topic_crypto
  - topic_real_estate
  - topic_shopping_deals
  - topic_marketing_advertising
  - topic_gambling

  # Technology
  - topic_technology
  - topic_ai
  - topic_cybersecurity
  - topic_programming_dev
  - topic_startups_vc
  - topic_consumer_electronics

  # Entertainment and fandom
  - topic_entertainment
  - topic_tv_movies
  - topic_music
  - topic_books
  - topic_anime_manga
  - topic_gaming
  - topic_esports
  - topic_celebrity
  - topic_celebrity_gossip
  - topic_comedy_memes

  # Lifestyle
  - topic_health
  - topic_nutrition_diet
  - topic_fitness
  - topic_mental_health
  - topic_beauty_fashion
  - topic_food_drink
  - topic_travel
  - topic_home_garden
  - topic_family_parenting
  - topic_relationships_dating

  # Sports
  - topic_sports

  # Other
  - topic_religion
  - topic_adult_services
  - topic_language_other

special_modes:
  - spoiler
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

Looks AI-written/AI-made (even if a human posted it). General detection label for AI-authored content regardless of quality or intent.

#### `ai_generated_reply`

AI-generated replies specifically. **This is a distinct category and a major user complaint.** Users widely report that AI-generated replies are flooding social media, creating inauthentic engagement and drowning out genuine conversation. These are typically:

- Generic, overly flattering, or template-like replies ("Great post!", "This is so insightful!")
- Repetitive phrasing patterns across many replies from the same or different accounts
- Low specificity to the original post content
- Often combined with promotional intent or follow-baiting

**Data collection note**: AI-generated replies should be scraped **with surrounding context** (the parent post they're replying to) for efficient training. The relationship between the reply and its parent is a strong signal—genuine replies reference specific content while AI replies are generic regardless of context.

Can co-occur with: `reply_spam`, `bot`, `promo`, `lead_gen`, `ai_slop`.

#### `ai_slop`

Low-quality, high-volume AI output optimized for engagement/monetization. "AI slop" is now a widely used term for this category. Distinct from `ai_generated` (neutral detection) and `ai_generated_reply` (reply-specific)—`ai_slop` implies the content is actively unwanted due to low quality and engagement-farming intent.

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

These are "user preference" filters, not moral judgments. Use `topic_*` prefix to separate "aboutness" from "badness/behavior" labels. A post can be `topic_crypto + scam + impersonation`.

> **UI note**: Show ~15–25 top toggles, add "More topics…" search for the long tail. Internally map to IAB Tier-1 for complete topic coverage.

#### News and society

##### `topic_news`

General news content.

##### `topic_world_news`

International news, global events.

##### `topic_local_news`

Local/regional news coverage.

##### `topic_war_conflict`

War, military conflict, armed disputes. Users often want to reduce exposure even when accurate.

##### `topic_crime_truecrime`

Crime reporting, true crime content, criminal cases.

##### `topic_disasters_tragedy`

Accidents, deaths, natural disasters, tragedies. Distinct from `graphic_violence`.

##### `topic_law_courts`

Legal proceedings, court cases, judicial content.

##### `topic_environment_climate`

Environmental issues, climate change, sustainability.

##### `topic_social_issues`

Activism, protests, culture-war adjacent content. Separate from `topic_politics`.

#### Politics and governance

##### `topic_politics`

Political content, partisan discussion, government affairs.

##### `topic_elections`

Election-related content. Often treated as a special mode; can coexist with `civic_misinfo`.

#### Money and commerce

##### `topic_finance`

Financial advice, stock tips, investment discussion (non-crypto).

##### `topic_investing`

Stocks, ETFs, options discourse.

##### `topic_personal_finance`

Budgeting, debt, FIRE movement, saving strategies.

##### `topic_crypto`

Crypto-related content. Any mention of a specific coin, token, NFT, wallet, exchange, DeFi, or blockchain references.

##### `topic_real_estate`

Housing discourse, landlord/tenant content, property listings.

##### `topic_shopping_deals`

Deal spam, sales, coupons. Not necessarily "spam" but users may want to filter.

##### `topic_marketing_advertising`

Ad industry, creator economy, marketing discourse.

##### `topic_gambling`

Gambling, betting, casino content.

#### Technology

##### `topic_technology`

General technology content.

##### `topic_ai`

AI discourse. Separate from `ai_generated`/`ai_slop` behavior labels.

##### `topic_cybersecurity`

Security breaches, exploits, infosec content. Can be noisy.

##### `topic_programming_dev`

Developer content, coding, "dev Twitter."

##### `topic_startups_vc`

Founder content, VC discourse, startup ecosystem.

##### `topic_consumer_electronics`

Gadgets, devices, hardware reviews.

#### Entertainment and fandom

##### `topic_entertainment`

General entertainment content (parent category).

##### `topic_tv_movies`

Television and film content, reviews, discussions.

##### `topic_music`

Music content, artist discussion, releases.

##### `topic_books`

Book content, reading, literary discussion.

##### `topic_anime_manga`

Anime and manga content, Japanese media.

##### `topic_gaming`

Video game content, game discussion.

##### `topic_esports`

Competitive gaming, esports tournaments.

##### `topic_celebrity`

Celebrity content, famous people.

##### `topic_celebrity_gossip`

Celebrity gossip specifically. More specific than `topic_celebrity`.

##### `topic_comedy_memes`

Meme content, comedy posts. Some users want a "no memes" mode even if not `low_effort`.

#### Lifestyle

##### `topic_health`

General health-related content, medical advice, wellness.

##### `topic_nutrition_diet`

Nutrition, diet content, food health.

##### `topic_fitness`

Exercise, workout content, gym culture.

##### `topic_mental_health`

Mental health content. Optional; be careful—can correlate with sensitive user traits.

##### `topic_beauty_fashion`

Beauty, fashion, style content.

##### `topic_food_drink`

Food content, recipes, restaurants, beverages.

##### `topic_travel`

Travel content, destinations, trips.

##### `topic_home_garden`

Home improvement, gardening, domestic content.

##### `topic_family_parenting`

Family content, parenting, children.

##### `topic_relationships_dating`

Relationship content, dating, romance discussion.

#### Sports

##### `topic_sports`

General sports content, game discussion, team fandom. Can add subtopics (`topic_soccer`, `topic_basketball`, etc.) if needed.

#### Other

##### `topic_religion`

Religious content, faith discussion, proselytizing.

##### `topic_adult_services`

Adult services promotion. Distinct from `adult_nudity`.

##### `topic_language_other`

Non-English content. For users who want English-only feeds. Can split into per-language topics (`topic_language_es`, etc.) for fine control.

---

### Special modes

These are not pure "topics" but behave like topic filters in user intent.

#### `spoiler`

Content containing spoilers for media (TV, movies, games, sports). Attribute label, not a topic. Can combine with time-box controls (24h/7d mutes) similar to X's muted words feature.

---

## Example multi-label combos

What "comprehensive" buys you:

| Example post                                                               | Labels                                                          |
| -------------------------------------------------------------------------- | --------------------------------------------------------------- |
| "Elon giveaway — send 0.1 BTC get 0.2 back" + deepfake clip                | `scam` + `topic_crypto` + `impersonation` + `manipulated_media` |
| Blue-check reply: "Amazing post! 🚀 DM me for business"                    | `reply_spam` + `promo` + `lead_gen` (+ `bot` if automated)      |
| AI thread mill: "I analyzed 10,000 CEOs…" (obvious LLM cadence)            | `ai_slop` + `content_farm` + `clickbait`                        |
| Election suppression: "Polling stations are closed tomorrow; vote by text" | `civic_misinfo` + `misinformation` + `topic_politics`           |
| Non-consensual sexual deepfake "nudification"                              | `nonconsensual_nudity` + `manipulated_media`                    |
| Thread about GPT-5 capabilities with AI-generated summary                  | `topic_ai` + `ai_generated`                                     |
| Reply: "This is incredible! 🔥 The future is here!" (to any post)          | `ai_generated_reply` + `reply_spam` + `low_effort`              |
| Reply: "Amazing insights! DM me to learn more about crypto gains"          | `ai_generated_reply` + `reply_spam` + `promo` + `lead_gen`      |
| Game of Thrones finale spoiler without warning                             | `spoiler` + `topic_tv_movies`                                   |
| Breaking news about earthquake with graphic imagery                        | `topic_disasters_tragedy` + `graphic_violence` + `topic_news`   |

---

## Multi-label rules

1. Use multiple labels when **both are true** (e.g. `topic_crypto` + `promo`).
2. `clean` can combine with topic labels (e.g. `clean` + `topic_crypto` = a normal crypto tweet).
3. `scam` does not imply `topic_crypto`; add `topic_crypto` only when the topic is crypto.
4. Topic labels (`topic_*`) can combine with any behavior label.
5. Use `topic_*` prefix for "aboutness" (user preference filters), keep unprefixed labels for "badness/behavior."

---

## Training notes

- A 100+ label ontology is feasible as a _schema_, but you'll likely want to:
  - Train a **coarse model** first (top ~15–25 behavior labels)
  - Add **specialists** (rules/regex or smaller sub-models) for things like `phishing`, `topic_crypto`, `topic_adult_services`, `privacy_doxxing`, etc.
  - Train **topic classifiers** separately from behavior classifiers
- That keeps model size small while still letting the product present a comprehensive set of toggles.
- You can cluster labels during training (merge into coarse super-classes) to improve performance, while keeping the dataset labels fine-grained for future remapping.
- **Topic labels** map internally to IAB Tier-1 for complete coverage even when not all are exposed as UI toggles.
- Consider time-boxed filtering for `spoiler` labels (24h/7d auto-expiry).
- **AI-generated reply detection** benefits significantly from surrounding context. When scraping `ai_generated_reply` samples, capture the parent post—the relationship between reply and parent is a strong signal (AI replies are generic regardless of context; genuine replies reference specific content).

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
- `parent_text`: text of the parent post (for replies; important for `ai_generated_reply` detection)
- `parent_id`: source_id of the parent post
- `is_reply`: boolean indicating if this is a reply
