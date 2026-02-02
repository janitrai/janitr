# Label Guide

This repo currently uses a 4-class scheme:

## crypto_scam
Direct attempts to steal funds or credentials.
- Seed phrase / private key requests.
- "Claim airdrop" + wallet connect + signing prompts.
- Phishing links to fake exchanges, wallet drainers.

Not crypto_scam:
- General hype, token promotion, or price talk (label `crypto`).
- Legitimate project announcements without theft patterns (label `crypto`).

## crypto
Crypto-related content that is not a scam.
- Token promotion, price talk, meme coins, NFT drops.
- Legitimate project announcements or ecosystem updates.
- General crypto discussion, news, or education.

Not crypto:
- Non-crypto content.
- Clear theft/phishing attempts (label `crypto_scam`).

## ai_reply
Replies that are likely automated / LLM-generated.
Signals include:
- Generic, overly flattering, or template-like replies.
- Repetitive phrasing across many replies.
- Low specificity to the original post.

Not ai_reply:
- Short human replies, slang-heavy, or context-specific responses.

## clean
Everything else that is not crypto_scam, crypto, or ai_reply.
This includes:
- Non-crypto content.

## Labeling rules
1) Only label `crypto_scam` when there is clear theft or phishing intent (highest priority).
2) Only label `ai_reply` when there are strong stylistic cues (even if the topic is crypto).
3) Label `crypto` when the content is crypto-related but not a scam.
4) Everything else is `clean`.

## Data shape
Each record is JSONL with at minimum:
- `id`: unique string
- `platform`: x | discord | web | dm | other
- `source_id`: platform-native id (tweet id, message id, etc.)
- `source_url`: canonical URL when available
- `collected_at`: ISO timestamp
- `text`: raw text (preserve exactly; do not truncate)
- `label`: crypto_scam | crypto | ai_reply | clean

Optional fields:
- `urls`: extracted URLs
- `addresses`: extracted wallet addresses
- `notes`: short rationale
