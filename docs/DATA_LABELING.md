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
2. `ai_generated_reply` - AI-generated promotional replies
3. `crypto` - Legitimate crypto discussion (news, trading, projects)
4. `promo` - Non-crypto promotional/advertising copy
5. `clean` - Non-crypto content

Multi-label note:
- Use multiple labels when the attributes are orthogonal (e.g. `crypto` + `promo`).
- `clean` should be exclusive (no other labels).

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

### Legitimate crypto (should be `crypto`, not `clean`/`promo`)
- Project announcements from real teams
- Price discussion, trading analysis
- Hiring posts from crypto companies
- Blockchain technical discussion

### Non-crypto promos (should be `promo`, not `clean`)
- Product ads, affiliate pitches, newsletter promos
- “Limited time”, “free”, “sign up now” style marketing copy

## Scripts Inventory

| Script | Mode | Purpose |
|--------|------|---------|
| `relabel_crypto.py` | Read-only | Suggest clean→crypto based on keywords/patterns |
| `mine_errors.py` | Read-only | Extract false positives/negatives from model |
| `manual_relabel.py` | Read-only (--apply to write) | Apply reviewed label changes from a changes file |
| `check_integrity.py` | Read-only | Validate dataset integrity (unique IDs, valid labels, etc.) |
| `fix_duplicate_ids.py` | Read-only (--apply to write) | Fix duplicate IDs by assigning unique suffixes |
| `validate_data.sh` | Read-only | Quick validation for CI/pre-commit |

## Integrity Checks

Run before committing any data changes:
```bash
python scripts/check_integrity.py
```

Checks performed:
- Valid JSON on every line
- Required fields present (id, label, text)
- No duplicate IDs
- Valid label values in `labels[]` (clean, crypto, crypto_scam, ai_generated_reply, promo)
- No empty text
- ID format consistency

For CI/pre-commit:
```bash
./scripts/validate_data.sh
```

## Anti-patterns

❌ `--in-place` flags that auto-modify data  
❌ Scripts that write directly to `data/sample.jsonl`  
❌ Bulk relabeling without human review  
❌ Trusting keyword matching for scam detection  
