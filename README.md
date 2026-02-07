# Janitr

<img src="assets/logo.svg" alt="Janitr logo" width="200">

A browser extension that filters crypto scams, AI-generated replies, and promotional spam from your social media feeds — entirely locally, with no network calls.

> **⚠️ Work in Progress**: This is an MVP. Currently it only works on X (Twitter) for demoing scam detection. Try it out, and if you have ideas for new content categories or improvements, tag or DM [@onusoz](https://x.com/onusoz) on X.

**Website:** [janitr.ai](https://janitr.ai)

## Vision

Janitr currently detects **crypto scams, spam, and promotional noise**, but the goal is much broader: build a comprehensive filtering system for all types of unwanted social media content.

**Ground-truth dataset:**

Every sample in the dataset is labeled with the full, fine-grained [label taxonomy](docs/LABELS.md) — 100+ labels spanning security & fraud, spam & manipulation, AI-generated content, information integrity, safety, and 40+ topic filters. This rich ground truth is preserved as-is and never simplified at the data layer.

Models trained on top of this dataset are a separate concern. Different models will collapse, group, or subset these labels in whatever way is most practical for their use case. The current fastText model, for example, uses a simple 3-class scheme (`scam`, `topic_crypto`, `clean`), but future models may use more classes, different groupings, or the full label set — the data supports all of these.

- 💡 **Have an idea for a new category?** — propose it via [@onusoz](https://x.com/onusoz)

**Local-first, lightweight models:**

A core principle is that models must run **locally on your device** — no cloud, no API calls, no data leaving your browser. This means optimizing for small model sizes and fast inference so detection works on everything from phones to older laptops. Privacy isn't optional.

The current implementation uses **fastText** (123KB quantized model running via WebAssembly), but the underlying ML approach may evolve as we expand to more content categories.

**Current dataset:**

~2,900 multi-label samples, all sourced from X via browser automation and human-verified. See [LABELS.md](docs/LABELS.md) for the full label guide.

This entire project — data collection, labeling, model training, and the extension itself — was built using [OpenClaw](https://github.com/openclaw/openclaw), an open framework for personal AI assistants.

**Open datasets:**

A key goal is to create and release **large, high-quality labeled datasets** for unwanted content detection. Training data is collected via browser automation (not APIs), labeled at scale using AI models, and every sample includes full provenance (platform, source URL, timestamp). These datasets will be open for researchers and developers building healthier social media experiences.

The approach: start narrow (crypto scams have clear ground truth), prove the pipeline works, then expand to fuzzier categories where labeling is more subjective.

## Install

1. Clone or download this repo
2. Open Chrome → `chrome://extensions`
3. Enable **Developer mode** (top right)
4. Click **Load unpacked** → select the `extension/` folder
5. Pin the extension to your toolbar

## How It Works

- **fastText model** runs in-browser via WebAssembly
- **Content scripts** scan posts and DMs as you scroll
- **3-class detection**: `scam`, `topic_crypto`, `clean` (backed by a [100+ label taxonomy](docs/LABELS.md))
- **Thresholds** are tunable per-class to control false positive rate
- **Zero network calls** — all inference happens on your CPU

## Model Performance

| Metric                 | Value              |
| ---------------------- | ------------------ |
| Model size             | 123 KB (quantized) |
| Scam precision         | 95%                |
| Scam recall            | 64%                |
| topic_crypto precision | 79%                |
| topic_crypto recall    | 37%                |
| Target FPR             | ≤ 2%               |

Current thresholds (`extension/fasttext/thresholds.json`), tuned for ≤ 2% FPR:

- `scam`: 0.93
- `topic_crypto`: 0.91
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

## Origin Story

This project was originally called **Internet Condom**, inspired by [@tszzl](https://x.com/tszzl)'s "internet condom" concept from mid-2025 — the idea of an AI layer between you and the web that filters, rewrites, and protects you from ads, spam, dark patterns, and low-signal content. We liked the metaphor (a protective barrier so _your agent_ touches the web, not you), but renamed to **Janitr** because... well, try explaining "Internet Condom" in a work meeting.

## License

[MIT](LICENSE)
