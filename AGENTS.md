# Agent Instructions

## Project Overview

**Janitr** is a browser extension that detects crypto scams, shill posts, and other undesirable content on X (Twitter) in real-time. It runs entirely on-device using a fastText classifier compiled to WebAssembly—no network calls required for classification.

### Goals

- **Low false positives** (FPR < 2%) — users tolerate missing some scams better than wrongly flagging legit content
- **Fast local inference** — classification happens in the browser, no cloud dependency
- **Small model size** — target 3-5MB for the browser extension (currently 122KB!)

### Labels

The dataset uses a **100+ label multi-label taxonomy** (see [LABELS.md](docs/LABELS.md) for the full guide). Ground truth is always preserved at full granularity.

For training, labels are collapsed into **3 mutually-exclusive classes**:

| Training class | Description                                                            |
| -------------- | ---------------------------------------------------------------------- |
| `scam`         | All bad-behavior labels (phishing, spam, promo, affiliate, bots, etc.) |
| `topic_crypto` | Crypto-related content with no bad behavior                            |
| `clean`        | Everything else                                                        |

---

## Directory Structure

```
Janitr/
├── extension/           # Chrome extension (MV3)
│   ├── manifest.json    # Extension manifest
│   ├── content-script.js # Injected into X pages, scans tweets
│   ├── background.js    # Service worker
│   ├── offscreen.js     # Offscreen doc for WASM inference
│   ├── fasttext/        # WASM model + JS bindings
│   │   ├── model.ftz    # Quantized fastText model (122KB)
│   │   ├── thresholds.json # Per-label confidence thresholds
│   │   ├── classifier.js # High-level classification API
│   │   └── fasttext.js  # WASM loader
│   ├── vendor/          # Third-party WASM bindings
│   └── tests/           # Extension smoke tests
│
├── scripts/             # Python ML pipeline
│   ├── prepare_data.py  # Convert JSONL → fastText format
│   ├── train_fasttext.py # Train classifier
│   ├── evaluate.py      # Eval metrics, confusion matrix
│   ├── inference.py     # Run inference on text
│   ├── reduce_fasttext.py # Quantize models for size
│   ├── tune_thresholds_fpr.py # Find thresholds for target FPR
│   └── ...              # Various data processing scripts
│
├── data/                # Training data (git-ignored)
│   ├── snapshots/       # Raw X post snapshots (JSONL)
│   ├── train.txt        # fastText training format
│   └── valid.txt        # fastText validation format
│
├── dataset/             # Curated labeled datasets
│
├── models/              # Trained models (git-ignored)
│   ├── scam_detector.bin # Full model (~97MB)
│   ├── scam_detector.ftz # Quantized model
│   └── experiments/     # Grid search results
│
├── config/
│   └── thresholds.json  # Production threshold config
│
├── docs/                # Documentation (SimpleDoc conventions)
│   ├── ARCHITECTURE.md  # System design
│   ├── MODEL.md         # Current model specs
│   ├── QUANTIZATION.md  # Size optimization guide
│   ├── LABELS.md        # Labeling guidelines
│   ├── DATA_MODEL.md    # Data schemas
│   └── logs/            # Daily development logs
│
├── tests/               # Integration tests
└── .agent/skills/       # Agent skills (SimpleDoc, etc.)
```

---

## Key Components

### 1. Browser Extension (`extension/`)

Chrome MV3 extension that:

- Injects `content-script.js` into X pages
- Extracts tweet text from DOM
- Runs fastText inference via WASM in an offscreen document
- Hides or badges flagged content based on confidence thresholds

**Key files:**

- `fasttext/classifier.js` — main classification API (`predictClassifier()`)
- `fasttext/thresholds.json` — per-label thresholds (scam: 0.93, topic_crypto: 0.91)
- `fasttext/model.ftz` — quantized model (123KB)

### 2. ML Pipeline (`scripts/`)

Python scripts for training and evaluation. Uses `fasttext-wheel` bindings.

**Typical workflow:**

```bash
# Prepare data
python scripts/prepare_data.py

# Train model
python scripts/train_fasttext.py

# Evaluate
python scripts/evaluate.py

# Quantize for browser
python scripts/reduce_fasttext.py --cutoff 1000 --dsub 8
```

**Key metrics to watch:**

- **FPR (False Positive Rate)** — must stay ≤ 2%
- **Scam Precision** — currently 95%
- **Scam Recall** — currently 64%
- **Model Size** — target < 6MB for extension (currently 123KB)

### 3. Data (`data/`, `dataset/`)

Training data collected via browser automation (OpenClaw), not X API.

**Format:** JSONL snapshots with fields:

- `id`, `text`, `authorHandle`, `timestamp`
- `label` (human or AI-assigned)
- `source` (provenance tracking)

**Note:** `data/` and `models/` are git-ignored due to size. Regenerate from scripts.

---

## Development Notes

### Pre-commit Hooks

Husky runs on every commit:

- `ruff format` — Python formatting
- `prettier` — JS/TS/JSON/MD formatting
- `simpledoc check` — Doc convention validation

### Quantization

See `docs/QUANTIZATION.md` for size optimization. Key finding: **smaller vocabulary (cutoff=1000) actually improves crypto recall** by forcing the model to focus on discriminative features.

### Thresholds

Thresholds are tuned per-label to maintain FPR < 2%. The `promo` label is currently disabled (threshold=1.0) to reduce noise.

---

## SimpleDoc Conventions

**Attention agent!** Before creating ANY documentation, use the `simpledoc` skill in `.agent/skills/simpledoc/SKILL.md`.

Key rules:

- Filenames: `SCREAMING_SNAKE_CASE.md`
- Daily logs: `docs/logs/YYYY-MM-DD.md` with YAML frontmatter
- No hyphens in doc filenames (use underscores)

---

## Daily Logs (SimpleLog) — NON-NEGOTIABLE

**You MUST use `npx @simpledoc/simpledoc log` to record everything worth noting.** This is not optional. Every significant change, decision, discovery, tradeoff, assumption, error, workaround, and clarification gets logged. If in doubt, log it.

The spec lives at `docs/SIMPLELOG_SPEC.md`.

### Where logs live

- Default location: `docs/logs/YYYY-MM-DD.md`.
- The CLI writes to `<repo-root>/docs/logs/` by default when inside a git repo.
- You can set a shared default in `simpledoc.json` and override locally in `.simpledoc.local.json` (see `docs/SIMPLEDOC_CONFIG_SPEC.md`).

### Create a daily log entry

Use the CLI to create the file and append entries:

```bash
npx @simpledoc/simpledoc log "Entry text here"
```

Notes:

- The CLI creates the daily log file if missing, including required frontmatter.
- It adds a new session section only when the threshold is exceeded (default 5 minutes).
- It preserves the text you type and inserts a blank line before each new entry.

### Multiline entries

Pipe or heredoc input (stdin) for multiline entries:

```bash
cat <<'EOF' | npx @simpledoc/simpledoc log
Multiline entry.
- line two
- line three
EOF
```

You can also use `--stdin` explicitly:

```bash
npx @simpledoc/simpledoc log --stdin <<'EOF'
Another multiline entry.
EOF
```

### Manual edits (if needed)

- Keep the YAML frontmatter intact (`title`, `author`, `date`, `tz`, `created`, optional `updated`).
- Ensure a blank line separates entries.
- Session sections must be `## HH:MM` (local time of the first entry in that section).

### What to log

- Significant changes, decisions, discoveries, tradeoffs, and assumptions
- Ongoing progress and small but real steps (changes, commands, tests, doc updates)
- Errors, failures, workarounds, and clarifications
- Anything you'd want to know if you came back to this in a week

Log each entry after completing the step or realizing the insight. No exceptions.

---

## Skill Breadcrumbs

- `general-visualization` (`.agent/skills/general-visualization/SKILL.md`): use for aligned plain-text metric comparisons (no markdown tables), explicit better/worse verdicts, and optional deltas. Trigger with `general-visualization` or `$general-visualization`.
