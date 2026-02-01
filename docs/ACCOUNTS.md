# Scam Hub Accounts Tracker

## Overview
Tracking accounts that serve as hubs for crypto scam activity, their ecosystems, and connections.

---

## SCAM TICKER SYMBOLS ($CASHTAGS)

Ticker symbols observed in scam posts. Searching these on X surfaces coordinated campaigns.

### Drainer/Airdrop Scam Tokens (HIGH VALUE LEADS)
| Ticker | Domain | Account(s) | Notes |
|--------|--------|------------|-------|
| $USOR | us-oil.cc | @ttdomarinho, @MariahEPayne, @catalkurt35, @PTirolo (verified!), @amanda_kaianee (verified!), @itryobaseball29, @Jods_brooo, @BadhonRoy13890 | Coordinated campaign, 18+ samples. Hub accounts post drainer link, amplifiers quote-tweet with 30+ cashtags |
| $TULSA | tulsa.soldex.one, tulsa.solgrid.trade | @Dannynegty, @aogrescue, @KBlock_Chain | 10 samples |
| $PENGUIN | penguinsol.live | @sandaaramandara | 13 samples |
| $TRUST | intuition-systems.info | @tomdumoulin_eth | 4 samples |
| $GWEI | ethgasfoundetion.org | @tomdumoulin_eth | Typo in domain! 4 samples |
| $LINK | hub-chain.link | @tomdumoulin_eth | Fake Chainlink airdrop, 2 samples |
| $COPPERINU | copperinudrop.lol | @0xDaonikko | 2 samples |
| $MOLTBOOK | moltbookcoin.lol | @0xDaonikko, @tokenTr0ve, @crypticByteX | 5 samples |
| $SOSO | sosovalue.com | @Soloemon2 | Referral spam, 2 samples |

### Pump.fun / Meme Token Tickers
| Ticker | Notes |
|--------|-------|
| $ELON | Elon-themed pump tokens, 5 samples |
| $TURBO | 4 samples |
| $HACHI | 3 samples |
| $CLAW | Claw ecosystem tokens, 3 samples |
| $FUJIDOGE | Japanese pump.fun, 1 sample |
| $FARTCOIN, $FARTBOY | Meme tokens |
| $EPSTEINZE, $WHITESTEIN | "Epstein" themed scam communities |
| $GOLDFISH, $GOLDDOGE | Gold-themed pumps |

### VIP Signal Spam Cashtags (search visibility gaming)
These are legitimate token tickers used to spam searches:
- $BTC, $ETH, $SOL, $DOGE, $XRP, $ADA (major coins)
- $GUN, $FUN, $W, $RARE, $RATS, $ONDA, $GALA, $CC (altcoins)

### Search Strategy
1. Search obscure tickers ($USOR, $TULSA, $TRUST) → surfaces coordinated scam networks
2. Combine with "airdrop" or "claim" → higher precision
3. Check account networks pushing same ticker

---

## SECURITY RESEARCHER ACCOUNTS (SOURCES)

Accounts that post breach/scam alerts. Their posts are **clean/educational**, but useful for:
1. Extracting IOCs (wallet addresses, domains)
2. Finding scammers in reply sections (recovery scams target victims)
3. Named threat actors for blocklists

| Account | Followers | Focus | Best Use | Rating |
|---------|-----------|-------|----------|--------|
| **@zachxbt** | 945K | Deep attribution investigations (named actors, $M amounts) | IOCs, threat actor names, check replies | ⭐⭐⭐ |
| **@realScamSniffer** | ~150K | Phishing/drainer alerts | Check "Show probable spam" in replies | ⭐⭐ |
| **@CertiKAlert** | 68K | Real-time breach alerts with wallet addresses | Extract IOCs, protocol names | ⭐⭐⭐ |
| **@PeckShieldAlert** | 94.2K | Aggregated breach stats, cites other researchers | Annual stats, check replies | ⭐⭐ |
| **@SlowMist_Team** | 88K | Chinese security firm, Crypto Asset Tracing Handbook | Educational/reference, @MistTrack_io for tracing | ⭐⭐ |
| **@wallet_guard** | 56K | Wallet security (MetaMask integrated) | ⚠️ SUNSETTING Mar 2025, see @_SEAL_Org | ⭐ |
| **@hexagate_** | 4.6K | Smart contract exploit detection | Acquired by @Chainalysis, see @blockaid_ | ⭐⭐ |
| **@blockthreat** | 5.2K | Weekly newsletter digest of all incidents | Historical reference, newsletter.blockthreat.io | ⭐⭐ |
| **@RevokeCash** | 84.2K | Token approval management, scam awareness | Educational content, revoke.cash | ⭐⭐ |

### Related Accounts Discovered
| Account | Focus |
|---------|-------|
| **@_SEAL_Org** | Security Alliance - real-time phishing defense network |
| **@MistTrack_io** | SlowMist's tracing tool |
| **@blockaid_** | Real-time threat detection |
| **@HypernativeLabs** | DeFi security monitoring |
| **@Chainalysis** | Blockchain analytics (owns Hexagate) |
| **@AnChainAI** | AI-powered blockchain security |

### Key Insight
Security researcher posts are **not scam content** — they're warnings. But they attract scammers in replies who target victims seeking help. Use "Show probable spam" button on X to reveal hidden scam replies.

### Notes
- **@revaboris** — account doesn't exist (may have been suspended/deleted)

---

## BASE CHAIN ECOSYSTEM

### @bankrbot
- **Type**: Token deployment bot (AI agent)
- **Followers**: 78.5K
- **Posts**: 592.1K (extremely high volume)
- **Joined**: December 2024
- **Chain**: Base (ETH L2)
- **Website**: bankr.bot
- **Parent account**: Automated by @0xDeployer
- **Pattern**: Users mention @bankrbot with "deploy [token] with ticker [X]"
- **Connected accounts**: @moltbook, @0xDeployer, @clanker_world, @myk_clawd
- **Samples collected**: 200 (x_0011 to x_0257)
- **Notes**: Extremely high volume. New posts every few minutes. Primary scam vector is token deployment spam. Claims to be "open agent ecosystem" for financially self-sufficient agents.

### @moltbook
- **Type**: AI agent aggregator / "front page of agent internet"
- **Followers**: ~180K
- **Chain**: Base
- **Pattern**: Legitimate platform but attracts scam replies
- **Connected accounts**: @bankrbot, @openclaw
- **Notes**: Many scammers reply to moltbook posts requesting token deployments

### @0xDeployer
- **Type**: Bankr ecosystem creator/operator
- **Followers**: 31K
- **Posts**: 9,064
- **Joined**: July 2021
- **Location**: Berkeley, CA
- **Website**: bankr.bot
- **Chain**: Base
- **Token**: $BNKR (listed on Coinbase!)
- **Pattern**: Parent account that automates @bankrbot
- **Notes**: Legitimate builder - the ecosystem itself isn't scam, but attracts scam spam from users requesting random token deployments. Key distinction: @0xDeployer/bankrbot = legitimate infra, spammers using it = scam content.

---

## SOLANA ECOSYSTEM

### @isaiahbami / @youralphacaller
- **Type**: Pump.fun token shill bot
- **Followers**: TBD (need to check)
- **Chain**: Solana
- **Pattern**: "pump.fun/[CA] CA - [address] $TOKEN Follow for more cooks 🔥"
- **TG Channel**: t.me/moonmemesscouts ("Moon Memes Scouts")
- **Connected accounts**: @youralphacaller (automation parent)
- **Monetization**: VIP group upsell for "early CA"
- **Samples collected**: 2 (x_0336, x_0337)
- **Notes**: High-volume automated shilling, claims "degen alphas"

### @IT95045755
- **Type**: Japanese pump.fun promoter  
- **Handle**: "next 114514 is $FUJIDOGE"
- **Chain**: Solana
- **Pattern**: Scheduled CA drops, community chat coordination
- **Samples collected**: 1 (x_0338)

### @wilzowishere
- **Type**: Fake airdrop promoter
- **Chain**: Solana
- **Pattern**: Meme coin launch + fake airdrop link (CRABAGENT.fun/airdrop.html)
- **X Label**: "Commentary account" (X flagged it!)
- **Samples collected**: 1 (x_0339)

### @web3kingmajor
- **Type**: Pump.fun signal bot
- **Chain**: Solana
- **Pattern**: Fake trading signals ("KOL Wallet Buy", "Vibe Buy Signal", "Max Pump: 6x")
- **Connected tools**: pro.ave.ai (tracking tool)
- **Samples collected**: 1 (x_0345)

### @Cynerion1
- **Type**: Pump.fun community promoter
- **Chain**: Solana
- **Pattern**: Creates X Communities for pump.fun tokens, posts "Dev is live" and buy calls
- **Communities**: CLAUDECRAFT, The White Epstein
- **Samples collected**: 2 (x_0348, x_0350)

### @AfrujaIewd / @Tregjb
- **Type**: Pump.fun community shillers
- **Chain**: Solana
- **Pattern**: "EPSTEIN SLEAZE" community coordination, fake ATH claims
- **Community**: EPSTEIN SLEAZE (2017825997760962918)
- **Samples collected**: 2 (x_0347, x_0349)

---

## ETHEREUM MAINNET

### TBD - Need to search for:
- Uniswap scam tokens
- ETH airdrop scams

---

## CROSS-CHAIN SCAM PATTERNS

### Airdrop Scams
- Search terms: "airdrop claim now", "free airdrop", "airdrop alert"
- **Pattern**: Fake claim links, .lol domains, urgency language, excessive cashtags for search gaming
- **Primary hub accounts (with followers)**:
  - **@ttdomarinho (Marin 🎭)** — 1,734 followers — USOR airdrop hub (us-oil.cc)
  - **@MariahEPayne (🌹 Mariah 🌹 Payne)** — 1,198 followers — USOR hub (us-oil.cc)
  - **@catalkurt35 (alcata.sol)** — 4,664 followers — USOR hub (us-oil.cc)
  - **@0xDaonikko (Daonikko)** — 868 followers — serial .lol scammer (elonsol.lol, copperinudrop.lol)
  - **@tokenTr0ve (tokenTrove)** — 1,125 followers — moltbookcoin.lol/claim
- **Amplifier network (USOR)**:
  - @itsyrceiram, @AllieDierschke, @SGenslerLPL, @jarrad1117 (cashtag-spam quote-tweets)
- **$TULSA coordinated campaign**:
  - **@Dannynegty** — 36 followers — tulsa.soldex.one
  - **@aogrescue** — 30 followers — tulsa.soldex.one
  - **@KBlock_Chain** — 228 followers — tulsa.solgrid.trade
- **Referral/airdrop spam bots**:
  - **@Soloemon2** — 7 followers — SoSoValue $SOSO referral spam (sosovalue.com/join/...)
- **Samples collected**: ~20+ (x_0262 onwards)

### DM Engagement Bots
- Pattern: "DM me for gains", "impressive project", generic praise
- Found in reply sections of popular accounts
- Hub accounts: TBD

### Wallet Drainers
- Pattern: Links to fake claim sites
- Often impersonate legitimate projects
- Hub accounts: TBD

### Cloud Mining Investment Scams (NEW)
- **@ChainsCloud** — 294 followers — chainscloud.com
- Pattern: "cloud mining apps" + "daily profit" + "trusted" language; often used in cloud-mining scams
- Samples: x_0569-x_0570

### WhatsApp Stock Signal Scam Network (NEW - Meme Coin Tag Spam)
- **Pattern**: AI-generated "stock analysis" praise → WhatsApp link → "Send 'stocks' to get them for free"
- **Hashtag gaming**: Uses #DOGE, #SHIB, #LDO, #AVAX to hijack meme coin searches
- **Cashtag spam**: Legitimate stock tickers ($META, $TSLA, $JPM) for search visibility
- **Phone numbers (2 shared across network)**:
  - `+1 (213) 462-6461` — @murli_bhai, @Dilipmakwana197, @Abhisek93371673, @pkallappa01, @VimalSi30944018, @Manoran05401242
  - `+1 (303) 419-5364` — @KAyyappa20, @KandilalK, @AyanAcharya5, @Riteshvish55681, @VivekKu83554512, @Muskaan66202721, @Devbrat06653615, @padaliya_hitesh
- **Accounts (26 documented)**:
  | Handle | Phone | Tickers Used |
  |--------|-------|--------------|
  | @Manoran05401242 | +12134626461 | #SHIB |
  | @Riteshvish55681 | +13034195364 | #SHIB |
  | @VimalSi30944018 | +12134626461 | #SHIB |
  | @VivekKu83554512 | +13034195364 | #SHIB |
  | @Devbrat06653615 | +13034195364 | #SHIB |
  | @Muskaan66202721 | +13034195364 | #SHIB |
  | @padaliya_hitesh | +13034195364 | #SHIB |
  | @KAyyappa20 | +13034195364 | #DOGE #SHIB |
  | @murli_bhai | +12134626461 | #DOGE |
  | @Dilipmakwana197 | +12134626461 | #DOGE #PEPE |
  | @KandilalK | +13034195364 | #DOGE |
  | @Abhisek93371673 | +12134626461 | #DOGE |
  | @pkallappa01 | +12134626461 | #DOGE |
  | @AyanAcharya5 | +13034195364 | #DOGE |
  | @Rahul182020 | +12134626461 | #PEPE |
  | @Mj31Jangid | +13034195364 | #PEPE |
  | @KameshKing6 | +12134626461 | #PEPE |
  | @SatishVulisett2 | +13034195364 | #PEPE |
  | @SureshS95360800 | +12134626461 | #PEPE |
  | @VikashK06132386 | +13034195364 | #PEPE |
  | @BDhalayat52584 | +12134626461 | #BONK |
  | @Chiragkhant19 | +12134626461 | #BONK |
  | @suleman59639 | +13034195364 | #BONK |
  | @SANOJKU33878929 | +12134626461 | #BONK |
  | @MDHamidJamal3 | +13034195364 | #BONK |
  | @Pradeep38758428 | +12134626461 | #BONK |
- **Samples collected**: 28 (x_0502 to x_0528, excluding 2 clean)
- **Notes**: Classic advance-fee fraud — victims DM WhatsApp, get pitched fake stock/crypto signals. Same 2 US phone numbers shared across 26+ accounts = centralized scam operation.

### Recovery Scams (Advance-Fee Fraud)
- Pattern: Spam targeting scam victims, "DM me for assistance in recovering your funds"
- Uses hashtags: #CryptoRecovery, #CryptoScam, #[ScamPlatformName]Scam
- Coordinated campaigns: Multiple accounts spam identical template about same "scam platform" (e.g., Bitfiz.net)
- **Hub accounts**:
  - **@Fante_ni** ("Coby Forensic 🕵️‍♂️ & Crypto Recovery") — 2,774 followers — high-volume spam
  - **@Mr_CThru** ("Chris_Xevi||CryptoRecovery expert") — 2,669 followers — coordinated template spam
  - **@tao_eliana** ("Philip Higman ~ Crypto Recovery") — coordinated network
  - **@Vincent_plr** ("Royce Vincent") — coordinated network
  - **@_iGoByEbo** ("Sli Shady Crypto Recovery Expert") — ironically honest name!
  - **@FrederickGLC** ("Frederick Gonzalez") — multiple samples in network
  - **@MarkEricCr0nqt** ("Mark Eric ~ Crypto Recovery") — XYZverse warning bait
  - **@LayTeaci** ("Robert_Mason||Cryptorecovery expert") — SERIAL SCAMMER, 5+ samples in 1 hour
    - Platform hashtags: #VaultCapital, #Acmexpe, #Gelvixinvest, #Diphaswengfin, #heritageprofit
    - Domain: tgexnode.com
  - **@gustaborecalde** ("Michael Bratton || Asset Recovery") — SERIAL SCAMMER, 5+ samples in 1 hour
    - Platform hashtags: #SonnenAfinitor, #ImpetuValtrix, #DonTradezip, #FrameAvageLab, #ArgentoLuxeron, #GiustoWex, #EstableCorexis, #FuturixBitport
- **Pattern analysis**: All use invented/obscure platform names as hashtags to poison search results. Victims searching for platform legitimacy find these "warnings" and DM for help.
- Samples collected: 15+ (x_0371-x_0372, x_0475-x_0487)
- Notes: These are scams targeting previous scam victims - they charge upfront "recovery fees" and never deliver

---

## CLEAN SOURCE ACCOUNTS

### Tech Discussions
- Python/JS/Rust programming searches
- Compiler optimization discussions

### Legitimate Crypto
- @jessepollak (Base founder) - legitimate
- @vikiival - legitimate dev

---

## Data Collection Progress

| Category | Target | Current | % |
|----------|--------|---------|---|
| Token deployment (bankrbot) | 500 | ~253 | 50% |
| Airdrop scams | 500 | ~30 | 6% |
| Recovery scams | 300 | ~35 | 12% |
| VIP signal groups | 200 | ~10 | 5% |
| DM engagement bots | 300 | ~6 | 2% |
| Wallet drainer/phishing | 200 | ~5 | 2.5% |
| **Total Scam** | **2000** | **375** | **18.75%** |
| Tech discussions | 300 | ~70 | 23% |
| Casual conversation | 300 | ~15 | 5% |
| Legit crypto | 200 | ~25 | 12.5% |
| Multilingual | 200 | ~15 | 7.5% |
| **Total Clean** | **1000** | **125** | **12.5%** |

---

## Session Log

### 2026-02-01 01:33 UTC - Starting extended collection
- Chairman requested 3000 total samples
- Currently at 257 (200 scam, 56 clean, 1 ai_reply)
- Need: 1800 more scam, ~935 more clean

### 2026-02-01 09:06 UTC - Airdrop scam expansion + follower counts
- Total now: 391 (299 scam, 91 clean)
- Discovered coordinated **USOR** campaign (us-oil.cc) with verified hubs
- Documented **TULSA** campaign (tulsa.soldex.one / tulsa.solgrid.trade)
- Added follower counts for key hubs (ttdomarinho, MariahEPayne, catalkurt35, 0xDaonikko, tokenTr0ve, Dannynegty, aogrescue, KBlock_Chain, Fante_ni, Mr_CThru, Soloemon2)
- Fixed bad IDs for x_0381 and x_0384
- Next ID: x_0392

### 2026-02-01 10:25 UTC - Session 4: Serial scammers + recovery scam network
- Total now: 445 (331 scam, 113 clean)
- **New serial scammer hub**: @tomdumoulin_eth
  - Domains: intuition-systems.info/claim ($TRUST), hub-chain.link/rewards ($LINK)
  - High engagement (28-29 RT, 118-126 likes) - likely amplifier network
- **New airdrop scam hub**: @sandaaramandara (verified!)
  - Domain: airdrop.penguinsol.live ($PENGUIN)
  - 8 RT, 58 likes
- **Recovery scam network discovered** (all posted ~32-33 min apart - coordinated):
  - @tao_eliana (Philip Higman ~ Crypto Recovery)
  - @Vincent_plr (Royce Vincent)
  - @_iGoByEbo (Sli Shady Crypto Recovery Expert - ironically honest!)
  - @FrederickGLC (Frederick Gonzalez)
  - Pattern: Same template structure, commas as separators, generic platform warnings (#Pairvex, #cortexdlt, #Blockwave, #AccGn), DM bait
- **DM engagement scam bots** (verified accounts spamming identical templates):
  - @CryptoKing_2020 (THE CRYPTO KING) — "Your Project really got me Excited 🤑 Follow me & DM Me Please"
  - @faveecryptoo (Fave Crypto) — "dm me or follow me back"
  - @MotherOfCrypto_ (Mother Of Crypto) — "moon plan" + "DM me"
  - @CryptoChad003 (Crypto Chad) — pump.fun CA spam
- **moltbookcoin.lol network** (coordinated same-drainer campaign):
  - @0xDaonikko, @tokenTr0ve, @crypticByteX — all pushing moltbookcoin.lol/claim
- **@tomdumoulin_eth** — SERIAL SCAMMER with 4+ drainer domains:
  - intuition-systems.info/claim ($TRUST)
  - hub-chain.link/rewards ($LINK)  
  - ethgasfoundetion.org/token ($GWEI) — note typo!
  - High engagement (28-38 RT, 118-133 likes) — has amplifier network

### 2026-02-01 10:55 UTC - Session 5: Recovery scam goldmine + VIP signals
- Total now: 501 (375 scam, 125 clean) **🎉 500+ MILESTONE**
- **Search query success**: "DM for recovery" OR "recover stolen crypto" OR "wallet recovery expert"
- **New serial recovery scammers discovered**:
  - **@MarkEricCr0nqt** ("Mark Eric ~ Crypto Recovery") — fake scam alert + DM bait
  - **@LayTeaci** ("Robert_Mason||Cryptorecovery expert") — 5 samples in 1 hour
  - **@gustaborecalde** ("Michael Bratton || Asset Recovery") — 5 samples in 1 hour
  - **@juanpi_s_c** ("Patrick Coman") — unusual ",,,,," separator pattern (automation template)
  - **@IRONGATE05** ("IRON GATE FUNDS RECOVERY" - verified!) — "no upfront payment" angle
  - **@CharlenePtns** ("Charles_Wills") — Torpex + Elon Musk impersonation targeting
- **Pattern**: Create fake platform names as hashtags (e.g., #Gelvixinvest, #SonnenAfinitor), spam warnings, harvest DMs from victims searching for platform reviews
- **Coordinated hashtag sharing**: @Fante_ni and @juanpi_s_c use identical platform hashtags (#SonnenAfinitor, #ImpetuValtrix, #DonTradezip) — same network
- **VIP Signal Group scams discovered**:
  - **@WealthSignal888** — DM-based VIP group recruitment
  - **@spideycrypt** (verified!) — TG @spideyadmin1, urgency "closes February 8!"
  - **@_callmebuchi** — fake testimonial shill boosting @heis_samueljay
  - **@OderaI72943** ("JACKSON WILLIAMS") — +277% profit claims, TG link, cashtag spam
- **Security researcher source strategy**: ScamSniffer post had spam replies hidden under "Show probable spam" button — X auto-filters scam replies!
- Next ID: x_0502

## NFT Scam Networks (2026-02-01)

### nerdsoneth.fun Drainer Campaign
**Hub Account:**
- @nerdsoneth (verified) — nerdsoneth.fun domain, "FREE MINT on ETH" with WL harvesting

**Amplifier/Shill Accounts:**
- @mmjpursuit — quotes hub, promotes WL
- @patecostd — referral link nerdsoneth.fun/?ref=
- @Drew0x27 — "This is cool af" shill post
- @e4ma_officiall (verified) — step-by-step WL instructions, credential harvesting
- @0xClaudz (verified) — manufactured enthusiasm post
- @Sophia_Claire22 — WL instructions with referral
- @LuvianElcy — "best nft free mint" shill
- @Rave_murphy — shill with #NFTs tag
- @KingsEcheeh (verified) — 10x WL giveaway with wallet harvesting
- @Andramichael5 (verified) — urgency tactics "ending in few hours"

### Fake OpenSea x InkChain Campaign
Coordinated campaign promoting fake "Fresh INK" free NFT mint, likely phishing opensea.io links:
- @xptoplayici — fake announcement with opensea.io/collection/fre
- @Trustway4 — "Airdrop Alert" persona, urgency + fake opensea link
- @Kaitor_eth (verified) — "Web3Moon" persona, same fake announcement
- @Harven85791878 — copy-paste of fake announcement

### NFT Giveaway Wallet Harvesting
- @Give2Caesar — "NFT Giveaway" with "drop wallet" + "tag fren"
- @Megaeth_Punks (verified) — "NFT GIVEAWAY" 5K views, wallet harvesting
- @_morkie (verified) — "Mint Free NFT" on Arc Testnet, morkie.xyz drainer, 15K views

### Other NFT Scam Domains
- nerdsoneth.fun
- morkie.xyz/vale
- arcflow.finance

