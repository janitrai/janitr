#!/usr/bin/env python3
"""Relabel clean posts with crypto-related content to 'crypto' label."""
import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

CRYPTO_TERMS = [
    "crypto","cryptocurrency","bitcoin","btc","ethereum","eth","xrp","ripple","solana","avalanche","avax",
    "uniswap","chainlink","lido","metamask","opensea","coinbase","binance","kraken","bybit","okx","kucoin",
    "coindesk","cointelegraph","blockworks","messari","coingecko","coinmarketcap","the block","theblock",
    "defi","web3","nft","blockchain","onchain","on-chain","stablecoin","stablecoins","token","tokens","tokenomics",
    "airdrop","airdrops","wallet","staking","staked","validator","validators","gas","gwei","mainnet","testnet",
    "rollup","rollups","zk","zkp","l1","l2","layer 1","layer 2","memecoin","meme coin","liquidity","tvl","apy",
    "bonding curve","bridge","bridging","satoshi","sats","halving","mining","miner","hashrate","custody","self custody",
    "multisig","seed phrase","depin","rwa","rwas","dao","governance","yield","dex","slippage","liquidation",
    "liquidations","liquidated","tokenized","hodl","hodling","bitcon","polymarket","bitwise","microstrategy",
    "usdc","usdt","dai","perps","perpetuals","perpetual","futures","spot","cross-chain","cross chain",
    "phantom","starknet","algorand","aptos","arbitrum","injective","sui","cardano","optimism","cosmos",
]

WORD_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in CRYPTO_TERMS) + r")\b", re.IGNORECASE)
TICKER_RE = re.compile(r"\$[A-Za-z]{2,10}\b")
ADDRESS_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")
BAG_RE = re.compile(r"\bbags?\b", re.IGNORECASE)
NONCRYPTO_BAG_RE = re.compile(r"\b(gloves?|picnic|litter|trash|garbage|surgery|food|transpo|grocery|paper|plastic)\b", re.IGNORECASE)

def should_relabel(obj: dict) -> bool:
    if obj.get("label") != "clean":
        return False
    text = (obj.get("text") or "").replace("₿", "b")
    addresses = obj.get("addresses") or []
    if addresses:
        return True
    if WORD_RE.search(text) or TICKER_RE.search(text) or ADDRESS_RE.search(text):
        return True
    if BAG_RE.search(text) and not NONCRYPTO_BAG_RE.search(text):
        return True
    return False

def relabel_file(path: Path, out_f):
    changed = 0
    total_clean = 0
    for line in path.open(encoding="utf-8"):
        obj = json.loads(line)
        if obj.get("label") == "clean":
            total_clean += 1
            if should_relabel(obj):
                obj["label"] = "crypto"
                changed += 1
        out_f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    return total_clean, changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/sample.jsonl")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--output")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.path)

    if args.dry_run:
        total_clean = 0
        changed = 0
        for line in path.open(encoding="utf-8"):
            obj = json.loads(line)
            if obj.get("label") == "clean":
                total_clean += 1
                if should_relabel(obj):
                    changed += 1
        print(f"clean_total={total_clean}")
        print(f"relabeled_count={changed}")
        return

    if args.in_place:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=path.parent) as tmp:
            total_clean, changed = relabel_file(path, tmp)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
        print(f"clean_total={total_clean}")
        print(f"relabeled_count={changed}")
        return

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_f:
            total_clean, changed = relabel_file(path, out_f)
    else:
        total_clean, changed = relabel_file(path, sys.stdout)

    print(f"clean_total={total_clean}", file=sys.stderr)
    print(f"relabeled_count={changed}", file=sys.stderr)

if __name__ == "__main__":
    main()
