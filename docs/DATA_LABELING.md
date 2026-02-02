# Data Labeling Guidelines

## Philosophy

**Scripts suggest, humans decide.**

Automated labeling scripts can help identify candidates for relabeling, but they MUST NOT modify the dataset directly. Any script that cannot achieve 100% accuracy should be read-only.

Why:
- Training data quality directly impacts model quality (garbage in, garbage out)
- Edge cases require human judgment (e.g., posts *about* scams vs actual scams)
- False labels compound into systematic model errors
- Manual review catches nuances that regex/heuristics miss

## Label Hierarchy

Priority order (highest to lowest):
1. `crypto_scam` - Malicious crypto content (drainers, fake airdrops, phishing)
2. `ai_reply` - AI-generated promotional replies
3. `crypto` - Legitimate crypto discussion (news, trading, projects)
4. `clean` - Non-crypto content

## Labeling Process

### Step 1: Run suggestion scripts (read-only)
```bash
# Get suggestions for clean→crypto relabeling
python scripts/relabel_crypto.py data/sample.jsonl --summary

# Export suggestions for review
python scripts/relabel_crypto.py data/sample.jsonl -o /tmp/suggestions.jsonl
```

### Step 2: Human review
- Review each suggestion manually
- Check for false positives (scams labeled as just "crypto")
- Check for false negatives (crypto content still labeled "clean")
- Apply changes manually via direct JSONL editing or helper scripts

### Step 3: Commit with clear message
```bash
git add data/sample.jsonl
git commit -m "Manual relabel: X entries clean→crypto, Y entries clean→crypto_scam"
```

## Common Edge Cases

### Scam patterns (should be `crypto_scam`, not `crypto`)
- "Connect wallet" + "claim" + rewards/airdrop
- Urgency language ("limited time", "act now")
- Too-good-to-be-true offers ("double your ETH")
- Impersonation of known projects/people

### Educational content (should be `crypto`, not `crypto_scam`)
- Posts *warning about* scams
- Explanations of how scams work
- Security advice ("never share your seed phrase")

### Legitimate crypto (should be `crypto`, not `clean`)
- Project announcements from real teams
- Price discussion, trading analysis
- Hiring posts from crypto companies
- Blockchain technical discussion

## Scripts Inventory

| Script | Mode | Purpose |
|--------|------|---------|
| `relabel_crypto.py` | Read-only | Suggest clean→crypto based on keywords/patterns |
| `mine_errors.py` | Read-only | Extract false positives/negatives from model |

## Anti-patterns

❌ `--in-place` flags that auto-modify data  
❌ Scripts that write directly to `data/sample.jsonl`  
❌ Bulk relabeling without human review  
❌ Trusting keyword matching for scam detection  
