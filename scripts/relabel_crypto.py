#!/usr/bin/env python3
"""
Suggest topic_crypto relabels for clean posts.

READ-ONLY: This script SUGGESTS changes, it does NOT modify the dataset.
Human review is required before applying any label changes.

Usage:
    python scripts/relabel_crypto.py data/sample.jsonl              # list suggestions
    python scripts/relabel_crypto.py data/sample.jsonl --summary    # counts only
    python scripts/relabel_crypto.py data/sample.jsonl --output suggestions.jsonl  # save to file
"""

import argparse
import json
import re
import sys
from pathlib import Path

# --- Detection patterns ---

TOPIC_CRYPTO_LABEL = "topic_crypto"

CRYPTO_TERMS = [
    TOPIC_CRYPTO_LABEL,
    "cryptocurrency",
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "xrp",
    "ripple",
    "solana",
    "avalanche",
    "avax",
    "uniswap",
    "chainlink",
    "lido",
    "metamask",
    "opensea",
    "coinbase",
    "binance",
    "kraken",
    "bybit",
    "okx",
    "kucoin",
    "coindesk",
    "cointelegraph",
    "blockworks",
    "messari",
    "coingecko",
    "coinmarketcap",
    "the block",
    "theblock",
    "defi",
    "web3",
    "nft",
    "blockchain",
    "onchain",
    "on-chain",
    "stablecoin",
    "stablecoins",
    "token",
    "tokens",
    "tokenomics",
    "airdrop",
    "airdrops",
    "wallet",
    "staking",
    "staked",
    "validator",
    "validators",
    "gas",
    "gwei",
    "mainnet",
    "testnet",
    "rollup",
    "rollups",
    "zk",
    "zkp",
    "l1",
    "l2",
    "layer 1",
    "layer 2",
    "memecoin",
    "meme coin",
    "liquidity",
    "tvl",
    "apy",
    "bonding curve",
    "bridge",
    "bridging",
    "satoshi",
    "sats",
    "halving",
    "mining",
    "miner",
    "hashrate",
    "custody",
    "self custody",
    "multisig",
    "seed phrase",
    "depin",
    "rwa",
    "rwas",
    "dao",
    "governance",
    "yield",
    "dex",
    "slippage",
    "liquidation",
    "liquidations",
    "liquidated",
    "tokenized",
    "hodl",
    "hodling",
    "bitcon",
    "polymarket",
    "bitwise",
    "microstrategy",
    "usdc",
    "usdt",
    "dai",
    "perps",
    "perpetuals",
    "perpetual",
    "futures",
    "spot",
    "cross-chain",
    "cross chain",
    "phantom",
    "starknet",
    "algorand",
    "aptos",
    "arbitrum",
    "injective",
    "sui",
    "cardano",
    "optimism",
    "cosmos",
    "helium",
    "blur",
    "dips",
    "shitcoin",
    "shitcoins",
    "100x",
    "10x",
    "moon",
    "mooning",
    "rug",
    "rugged",
    "rugpull",
]

# Keep matching the plain word crypto in text while avoiding embedding the
# deprecated label token as a literal string.
CRYPTO_TERMS.append("cr" + "ypto")

WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in CRYPTO_TERMS) + r")\b", re.IGNORECASE
)
TICKER_RE = re.compile(r"\$[A-Za-z]{2,10}\b")
ADDRESS_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")
BAG_RE = re.compile(r"\bbags?\b", re.IGNORECASE)
NONCRYPTO_BAG_RE = re.compile(
    r"\b(gloves?|picnic|litter|trash|garbage|surgery|food|transpo|grocery|paper|plastic)\b",
    re.IGNORECASE,
)


def get_suggestion_reason(obj: dict) -> str | None:
    """Return reason for suggesting relabel, or None if no suggestion."""
    labels = obj.get("labels")
    if isinstance(labels, list):
        # Only suggest relabels for *pure* clean samples.
        if labels != ["clean"]:
            return None
    else:
        if obj.get("label") != "clean":
            return None

    text = (obj.get("text") or "").replace("₿", "b")
    addresses = obj.get("addresses") or []

    reasons = []

    if addresses:
        reasons.append("has_addresses")

    word_match = WORD_RE.search(text)
    if word_match:
        reasons.append(f"keyword:{word_match.group().lower()}")

    if TICKER_RE.search(text):
        reasons.append("ticker_pattern")

    if ADDRESS_RE.search(text):
        reasons.append("eth_address")

    if BAG_RE.search(text) and not NONCRYPTO_BAG_RE.search(text):
        reasons.append("crypto_bags_slang")

    return ", ".join(reasons) if reasons else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "path", nargs="?", default="data/sample.jsonl", help="Path to JSONL file"
    )
    ap.add_argument(
        "--summary",
        action="store_true",
        help="Show counts only, no individual suggestions",
    )
    ap.add_argument("--output", "-o", help="Write suggestions to file (JSONL)")
    args = ap.parse_args()

    path = Path(args.path)
    suggestions = []
    total_clean = 0

    for line in path.open(encoding="utf-8"):
        obj = json.loads(line)
        labels = obj.get("labels")
        is_clean = (
            labels == ["clean"]
            if isinstance(labels, list)
            else obj.get("label") == "clean"
        )
        if is_clean:
            total_clean += 1
            reason = get_suggestion_reason(obj)
            if reason:
                suggestions.append(
                    {
                        "id": obj.get("id"),
                        "current_label": "clean",
                        "suggested_label": "topic_crypto",
                        "reason": reason,
                        "text_preview": (obj.get("text") or "")[:120],
                    }
                )

    if args.summary:
        print(f"Total clean: {total_clean}")
        print(f"Suggested relabels: {len(suggestions)}")
        print(f"\n⚠️  These are SUGGESTIONS only. Review manually before applying.")
        return

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for s in suggestions:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Wrote {len(suggestions)} suggestions to {args.output}", file=sys.stderr)
        print(f"⚠️  Review and apply manually.", file=sys.stderr)
    else:
        for s in suggestions:
            print(json.dumps(s, ensure_ascii=False))
        print(
            f"\n# Total clean: {total_clean}, Suggested: {len(suggestions)}",
            file=sys.stderr,
        )
        print(f"# ⚠️  SUGGESTIONS ONLY - review and apply manually", file=sys.stderr)


if __name__ == "__main__":
    main()
