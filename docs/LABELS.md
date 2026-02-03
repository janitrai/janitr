# Label Guide

This repo currently uses a 5-class scheme with **multi-label support**.
Each sample can have **one or more labels** when the attributes are orthogonal
(e.g. topic + intent).

Recommended representation:
- `labels`: array of strings (one or more labels)
- `primary_label` (optional): a single label when a primary class is needed

## scam
Direct attempts to steal funds or credentials (crypto or non-crypto).
- Seed phrase / private key requests.
- "Claim airdrop" + wallet connect + signing prompts.
- Phishing links to fake exchanges, wallet drainers.

If the scam is crypto-related, also add the `crypto` label.

## crypto
Crypto-related content that is not a scam.
- Any mention of a specific coin, token, NFT, or crypto asset.
- Token promotion, price talk, meme coins, NFT drops.
- Legitimate project announcements or ecosystem updates.
- General crypto discussion, news, or education.
- Wallet, exchange, DeFi, or blockchain references.

Not crypto:
- Non-crypto content.
- Clear theft/phishing attempts (label `scam`, add `crypto` if applicable).

## ai_generated_reply
Replies that are likely automated / LLM-generated.
Signals include:
- Generic, overly flattering, or template-like replies.
- Repetitive phrasing across many replies.
- Low specificity to the original post.

Not ai_generated_reply:
- Short human replies, slang-heavy, or context-specific responses.

## promo
Promotional or advertising content.
- Product ads, affiliate pitches, newsletter promos.
- “Limited time”, “free”, “sign up now”, “get access” marketing copy.
- Generic growth/marketing hooks.

Promo can co-occur with `crypto` (e.g. a crypto product launch).
Scams should be labeled `scam` (and `crypto` if applicable), not `promo`.

## clean
Everything else that is not scam, crypto, ai_generated_reply, or promo.
This includes:
- Non-crypto content.

## Multi-label rules
- Use multiple labels when **both are true** (e.g. `crypto` + `promo`).
- `clean` should be **exclusive** (do not combine with other labels).
- `scam` does not imply `crypto`; add `crypto` only when the topic is crypto.
- Example (`crypto` + `promo`): "Claude Code/openclaw agents can now buy their own Linux VMs with USDC. Deploy code. Host apps." → labels: `["crypto", "promo"]`

**Training note:** you can cluster labels during training (merge into coarse super-classes) to improve performance, while keeping the dataset labels fine-grained for future remapping.

## Labeling rules
1) Only label `scam` when there is clear theft or phishing intent (highest priority).
2) Only label `ai_generated_reply` when there are strong stylistic cues (even if the topic is crypto).
3) Label `crypto` when the content mentions any coin, token, NFT, wallet, exchange, or crypto asset.
4) Label `promo` when the content is advertising or promotional copy.
5) Everything else is `clean`.

## Data shape
Each record is JSONL with at minimum:
- `id`: unique string
- `platform`: x | discord | web | dm | other
- `source_id`: platform-native id (tweet id, message id, etc.)
- `source_url`: canonical URL when available
- `collected_at`: ISO timestamp
- `text`: raw text (preserve exactly; do not truncate)
- `labels`: scam | crypto | ai_generated_reply | promo | clean (one or more)

Optional fields:
- `urls`: extracted URLs
- `addresses`: extracted wallet addresses
- `notes`: short rationale
