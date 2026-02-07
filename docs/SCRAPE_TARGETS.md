# Scrape Targets

> Curated lists of X accounts for collecting training data. Goal: increase clean and legit topic_crypto samples to reduce FPR.

## Clean — Mainstream Accounts

Diverse, high-volume accounts that produce mostly clean, non-crypto content.

### News

- @BBCBreaking — hard news, global
- @Reuters — wire service, factual
- @AP — wire service, factual
- @nytimes — general news
- @guardian — general news, international
- @CNN — general news
- @WIRED — tech/science journalism
- @TheEconomist — economics, world affairs
- @NPR — public radio, diverse topics
- @WSJ — finance + general

### Sports

- @espn — sports news
- @NBA — basketball
- @NFL — football
- @premierleague — soccer
- @UEFA — European soccer
- @F1 — Formula 1
- @MLB — baseball
- @FIFAWorldCup — soccer, global
- @UFC — MMA

### Entertainment

- @Variety — film/TV industry
- @RollingStone — music + culture
- @IGN — gaming
- @RottenTomatoes — movie reviews
- @billboard — music charts
- @netflix — streaming, pop culture
- @NintendoAmerica — gaming
- @PlayStation — gaming
- @MarvelStudios — entertainment

### Science & Tech (non-crypto)

- @NASA — space, science
- @NatGeo — nature, science
- @verge — tech news
- @TechCrunch — tech/startup news
- @github — developer community
- @MIT — academic, research
- @nature — scientific journal
- @newscientist — science news

### Food & Lifestyle

- @GordonRamsay — food, personality
- @nytcooking — food/recipes
- @Airbnb — travel
- @NatGeoTravel — travel
- @Nike — sports/lifestyle brand

### Comedy & Culture

- @TheOnion — satire
- @dril — shitposting (clean-weird)
- @MKBHD — tech reviews, personality
- @elonmusk — high volume, mixed topics
- @BillGates — philanthropy, tech, books

### Politics & Opinion (diverse viewpoints)

- @POTUS — US politics
- @BBCPolitics — UK politics
- @politico — political news
- @FoxNews — conservative news
- @MSNBC — liberal news

## Legit Crypto — Project & Community Accounts

High-signal crypto accounts that produce legitimate, non-scam content. These help the model learn "crypto does not equal scam."

### Layer 1 / Major Protocols

- @ethereum — Ethereum foundation
- @solana — Solana foundation
- @bitcoin — Bitcoin community
- @arbitrum — L2, active community
- @optimismFND — Optimism L2
- @base — Coinbase L2
- @0xPolygon — Polygon
- @SuiNetwork — Sui
- @avax — Avalanche
- @cosmos — Cosmos ecosystem
- @cardano — Cardano

### Exchanges & Infrastructure

- @coinbase — major exchange
- @binance — major exchange
- @krakenfx — Kraken exchange
- @Uniswap — DEX
- @AaveAave — DeFi lending
- @MakerDAO — DeFi stablecoin
- @chainlink — oracle network
- @etherscan — block explorer

### Crypto Analysts & Builders (legit)

- @VitalikButerin — Ethereum co-founder
- @balajis — tech/crypto thought leader
- @punk6529 — NFT/crypto culture
- @cburniske — Placeholder VC
- @hasufl — crypto researcher
- @sassal0x — Ethereum educator
- @ljin18 — crypto VC (Variant)
- @rleshner — Compound founder
- @StarkWareLtd — StarkWare, ZK research

### Crypto Media

- @CoinDesk — crypto journalism
- @theblock\_\_ — crypto journalism
- @DeFi_Dad — DeFi educator
- @messaricrypto — crypto research/data
- @glassnode — on-chain analytics
- @DuneAnalytics — on-chain data
- @Bankless — crypto media

## Scraping Notes

- **Volume target**: aim for 2,000+ clean and 2,000+ topic_crypto samples (current: ~1,344 and ~1,334)
- **Scrape posts, not replies** (for this round — reply scraping is a separate effort for scam labels)
- **Include retweets/quotes** — they reflect what the account amplifies
- **Preserve full text** — do not truncate
- **Tag source account** in metadata for provenance
- **Mix time ranges** — do not just scrape today's posts, get a spread across recent weeks
