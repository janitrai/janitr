# Internet Condom

<img src="assets/logo-white.svg" alt="Internet Condom logo" width="200">

A browser extension that filters crypto scams, AI-generated replies, and promotional spam from your social media feeds — entirely locally, with no network calls.

## Install

1. Clone or download this repo
2. Open Chrome → `chrome://extensions`
3. Enable **Developer mode** (top right)
4. Click **Load unpacked** → select the `extension/` folder
5. Pin the extension to your toolbar

## How It Works

- **fastText model** runs in-browser via WebAssembly
- **Content scripts** scan posts and DMs as you scroll
- **Classification** into: `scam`, `crypto`, `promo`, `clean`
- **Thresholds** are tunable per-class to control false positive rate
- **Zero network calls** — all inference happens on your CPU

## Model Performance

| Metric              | Value              |
| ------------------- | ------------------ |
| Model size          | 122 KB (quantized) |
| Crypto recall       | 89%                |
| False positive rate | < 2%               |

Current thresholds (`extension/fasttext/thresholds.json`):

- `crypto`: 0.74
- `scam`: 0.93
- `promo`: 1.0 (disabled)
- `clean`: 0.1

## Development

### Training pipeline

```bash
python -m venv .venv
source .venv/bin/activate
pip install fasttext-wheel

make prepare   # prepare train/valid splits
make train     # train fastText model
make eval      # evaluate on test set
```

### Extension development

The extension lives in `extension/`. Key files:

- `manifest.json` — Chrome extension manifest (MV3)
- `content-script.js` — injected into pages, scans DOM
- `background.js` — service worker
- `fasttext/` — WASM runtime + quantized model + thresholds

### Quantization

Models are quantized to ~120KB using fastText's built-in quantization:

- `cutoff=1000` (prune rare words)
- `dsub=8` (product quantization)

See `docs/QUANTIZATION.md` for the full grid search results.

## Documentation

| Doc                                     | Description                          |
| --------------------------------------- | ------------------------------------ |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and data flow          |
| [WASM_PLAN.md](docs/WASM_PLAN.md)       | WebAssembly integration strategy     |
| [QUANTIZATION.md](docs/QUANTIZATION.md) | Model compression experiments        |
| [LABELS.md](docs/LABELS.md)             | Labeling rules and class definitions |
| [DATA_MODEL.md](docs/DATA_MODEL.md)     | Schemas and storage patterns         |
| [DATA_SOURCES.md](docs/DATA_SOURCES.md) | Where training data comes from       |
| [PLAN.md](docs/PLAN.md)                 | Project roadmap                      |
| [EVAL_RESULTS.md](docs/EVAL_RESULTS.md) | Model evaluation results             |

## Data Collection

Training data is collected via browser automation (UI scraping), not APIs. We use AI models to label scams at scale. All samples include provenance (platform, source URL, timestamp).

See `docs/DATA_LABELING.md` for the labeling workflow.

## Local-First

No network calls. No telemetry. No cloud. Your browsing stays private.
