# Scam Detection Data Sources

## Security Researcher Accounts (scam trackers)
These accounts expose/track scams - their posts and replies are goldmines for finding actual scammers.

| Handle | Type | Status | Notes |
|--------|------|--------|-------|
| @realScamSniffer | Phishing/drainer alerts | 🔲 TODO | Web3 Anti-Scam, postmortems |
| @zachxbt | Attribution/investigations | 🔲 TODO | Scam cluster investigations |
| @CertiKAlert | Hacks/exploits | 🔲 TODO | Security alerts |
| @PeckShieldAlert | Exploits/compromised | 🔲 TODO | Exploit warnings |
| @SlowMist_Team | Threat intel | 🔲 TODO | Incident notes |
| @MistTrack_io | Malicious fund alerts | 🔲 TODO | Fund tracing |
| @BlockSecTeam | Exploit monitoring | 🔲 TODO | Technical analysis |
| @chainabuse | Scam reports | 🔲 TODO | Reported indicators |
| @RevokeCash | Approval risks | 🔲 TODO | Revoke scam patterns |
| @wallet_guard | Phishing warnings | 🔲 TODO | Defensive guidance |
| @RektHQ | Incident reporting | 🔲 TODO | Exploit writeups |
| @web3isgreat | Failure catalog | 🔲 TODO | Curated scam/fail list |

## Indicator Sources (for ground truth)
- Chainabuse API - malicious addresses/URLs
- CryptoScamDB - malicious domains
- ScamSniffer scam-database (GitHub) - phishing domains + addresses
- PhishTank - verified phishing URLs

## Meme Coin Ticker Searches (high scam density)
Popular meme coins get impersonated constantly for fake airdrops/giveaways.

| Ticker | Search Query | Status | Notes |
|--------|--------------|--------|-------|
| $DOGE | `"$DOGE airdrop" OR "free DOGE"` | ✅ DONE | WhatsApp stock scam network (7 samples x_0509-x_0515) |
| $SHIB | `"$SHIB airdrop" OR "free SHIB"` | ✅ DONE | WhatsApp stock scam network (7 samples x_0502-x_0508) |
| $PEPE | `"$PEPE airdrop" OR "free PEPE"` | ✅ DONE | WhatsApp stock scam network (7 samples x_0516-x_0522) |
| $BONK | `"$BONK airdrop"` | ✅ DONE | WhatsApp scam (6) + 2 clean (x_0523-x_0530) |
| $FLOKI | `"$FLOKI airdrop"` | ✅ DONE | 1 fake airdrop scam + 1 clean (x_0531-x_0532) |
| $WIF | `"$WIF airdrop"` | ✅ DONE | Low yield - mostly @_free_wif_i (JP influencer), not crypto scams |

## Collection Strategy
1. Check security account posts for scam reports
2. Find the scam accounts being reported
3. Collect samples from those scam accounts
4. Document hubs in ACCOUNTS.md
5. Search meme coin tickers for fake airdrop/giveaway scams

## Session Log
### 2026-02-01 10:50 UTC - Starting diversification
- Current: 474 samples (348 scam, 125 clean)
- Problem: 73% are bankrbot token deploys
- Goal: Diversify into airdrop, recovery, DM, signal scams
- Starting with: @realScamSniffer
