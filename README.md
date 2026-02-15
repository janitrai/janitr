# Janitr

<img src="assets/logo.svg" alt="Janitr logo" width="200">

A browser extension that filters crypto scams, AI-generated replies, and promotional spam from your social media feeds. Inference runs locally on-device.

> **⚠️ Work in Progress**: This is an MVP. Currently it only works on X (Twitter) for demoing scam detection. Try it out, and if you have ideas for new content categories or improvements, tag or DM [@janitr_ai](https://x.com/janitr_ai) on X.

**Website:** [janitr.ai](https://janitr.ai)

## Vision

Janitr currently detects **crypto scams, spam, and promotional noise**, but the goal is much broader: build a comprehensive filtering system for all types of unwanted social media content.

**Ground-truth dataset:**

Every sample in the dataset is labeled with the full, fine-grained [label taxonomy](docs/LABELS.md) — 100+ labels spanning security & fraud, spam & manipulation, AI-generated content, information integrity, safety, and 40+ topic filters. This rich ground truth is preserved as-is and never simplified at the data layer.

Models trained on top of this dataset are a separate concern. Different models will collapse, group, or subset these labels in whatever way is most practical for their use case. The current fastText model, for example, uses a simple 3-class scheme (`scam`, `topic_crypto`, `clean`), but future models may use more classes, different groupings, or the full label set — the data supports all of these.

- 💡 **Have an idea for a new category?** — propose it via [@janitr_ai](https://x.com/janitr_ai)

**Local-first, lightweight models:**

A core principle is that models must run **locally on your device** — no cloud, no API calls, no data leaving your browser. This means optimizing for small model sizes and fast inference so detection works on everything from phones to older laptops. Privacy isn't optional.

The current implementation supports **two local backends**:

- **Transformer (default):** ONNX Runtime Web, stronger scam detection, larger model
- **fastText:** ultra-small WASM path, useful as a lightweight fallback

**Current dataset:**

~4,200+ multi-label samples, all sourced from X via browser automation and human-verified. See [LABELS.md](docs/LABELS.md) for the full label guide.

This entire project — data collection, labeling, model training, and the extension itself — was built using [OpenClaw](https://github.com/openclaw/openclaw), an open framework for personal AI assistants.

**Open datasets:**

A key goal is to create and release **large, high-quality labeled datasets** for unwanted content detection. Training data is collected via browser automation (not APIs), labeled at scale using AI models, and every sample includes full provenance (platform, source URL, timestamp). These datasets will be open for researchers and developers building healthier social media experiences.

The approach: start narrow (crypto scams have clear ground truth), prove the pipeline works, then expand to fuzzier categories where labeling is more subjective.

## Install

1. Clone or download this repo
2. Run setup commands:

```bash
corepack enable
pnpm install
pnpm extension:build
```

3. Open Chrome → `chrome://extensions`
4. Enable **Developer mode** (top right)
5. Click **Load unpacked** → select the `extension/` folder
6. Pin the extension to your toolbar

## How It Works

- **Transformer model** runs in-browser via ONNX Runtime Web (default backend)
- **fastText model** runs in-browser via WebAssembly (optional fallback backend)
- **Content scripts** scan posts and DMs as you scroll
- **3-class detection**: `scam`, `topic_crypto`, `clean` (backed by a [100+ label taxonomy](docs/LABELS.md))
- **Thresholds** are tunable per-class to control false positive rate
- **No network calls during inference** — classification happens on your CPU (network is only used when you explicitly download remote runs in advanced mode)

## Model Performance

Frozen-split benchmark (expanded holdout, see `docs/reports/2026-02-14-frozen-split-fasttext-vs-transformer-benchmark.md`):

| Metric      | fastText (ftz) | Transformer (int8 ONNX) |
| ----------- | -------------- | ----------------------- |
| Scam P      | 0.9128         | 0.9375                  |
| Scam R      | 0.5551         | 0.6122                  |
| Scam F1     | 0.6904         | 0.7407                  |
| Scam FPR    | 0.0158         | 0.0121                  |
| Macro F1    | 0.7838         | 0.8013                  |
| Exact Match | 0.7624         | 0.8241                  |

Model artifact sizes from the same benchmark:

- fastText `.ftz`: ~123 KB
- transformer int8 ONNX: ~3.4 MB

## Hugging Face Experiment Runs

Janitr keeps a rolling artifact repo on Hugging Face for large model files and dataset checkpoints:

- **Repo:** [`janitr/experiments`](https://huggingface.co/janitr/experiments)
- **Purpose:** store versioned experiment runs and dataset snapshots without bloating the main git repo
- **Structure:** runs are indexed with `runs/INDEX.json`; each run has `RUN_INFO.json` plus model/eval files
- **Scope boundary:** Hugging Face stores model/dataset artifacts only (ONNX, tokenizer, thresholds, eval, dataset checkpoints). Runtime app assets (ONNX Runtime Web JS/WASM) stay in the main `janitr` repo build pipeline.

### Extension approach (advanced mode)

- The extension ships with local runtime assets and supports bundled or cached transformer model artifacts.
- Default backend is `transformer`; you can switch backend from the popup or options page.
- In advanced mode (`Options` page), you can:

1. list remote runs from Hugging Face
2. inspect run metadata/eval
3. download and activate a selected run

- Downloaded run artifacts are integrity-checked (size + SHA-256) and cached locally in IndexedDB.
- If a selected remote run cannot load, the extension falls back to the bundled transformer.
- If bundled transformer artifacts are not present in your checkout, Janitr auto-selects the newest cached Hugging Face run. If none is cached yet, download and activate a run from Options.

## Development

### Training pipeline

```bash
# FastText path
uv run --project scripts python scripts/prepare_data.py
uv run --project scripts python scripts/train_fasttext.py
uv run --project scripts python scripts/evaluate.py

# Transformer path (teacher -> student -> eval)
uv run --project scripts python scripts/train_transformer_teacher.py --seeds 13,42,7
uv run --project scripts python scripts/calibrate_teacher.py
uv run --project scripts python scripts/cache_teacher_logits.py
uv run --project scripts python scripts/train_transformer_student_distill.py
uv run --project scripts python scripts/export_transformer_student_onnx.py
uv run --project scripts python scripts/quantize_transformer_student.py
uv run --project scripts python scripts/evaluate_transformer.py
```

### Extension development

The extension lives in `extension/`. Key files:

- `manifest.json` — Chrome extension manifest (MV3)
- `src/` — TypeScript source for background/content/offscreen/options/popup
- `transformer/` — bundled transformer runtime + model loader
- `fasttext/` — WASM runtime + quantized fallback model + thresholds

Build command:

- `pnpm extension:build` transpiles TS and stages required ONNX Runtime Web assets from `node_modules/onnxruntime-web/dist` into `extension/vendor/onnxruntime-web/`.

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

No telemetry. No cloud inference. Your browsing stays private.

By default, inference is fully local. Optional network access is only used when you explicitly download remote experiment artifacts in advanced mode.

## Similar Projects

| Project                           | Author                                  | Approach                                                                                                                                         |
| --------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Reply Bye](https://replybye.run) | [@iannuttall](https://x.com/iannuttall) | Chrome extension that scores replies using language detection, AI analysis, and community votes. Highlights, hides, or blocks AI spam reply guys |

> Know of another project fighting AI spam on social media? Open an issue or DM [@janitr_ai](https://x.com/janitr_ai)

## Origin Story

This project was originally called **Internet Condom**, inspired by [@tszzl](https://x.com/tszzl)'s "internet condom" concept from mid-2025 — the idea of an AI layer between you and the web that filters, rewrites, and protects you from ads, spam, dark patterns, and low-signal content. We liked the metaphor (a protective barrier so _your agent_ touches the web, not you), but renamed to **Janitr** because... well, try explaining "Internet Condom" in a work meeting.

## License

[MIT](LICENSE)
