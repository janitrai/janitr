# WASM Implementation Plan for Scam Detector

> **Reviewed 2026-02-02**: API verified against `fasttext.wasm.js` v1.0.0 source. Fixed Pair access (`[0]`/`[1]` not `.first`/`.second`).
>
> **Updated 2026-02-02**: Applied 4 fixes from pi review: (1) scan predictions for `__label__scam` explicitly, (2) call `predictions.delete()` to avoid WASM heap leak, (3) manifest pattern `fastText/**` for subfolders, (4) filename case verified correct.

## Overview

Load the trained fastText model (`quant-cutoff10k.ftz`, 690KB) in the browser extension using WebAssembly.

## Recommended Library

**[fasttext.wasm.js](https://github.com/yunsii/fasttext.wasm.js)** (v1.0.0)

- ✅ TypeScript support
- ✅ Explicit browser extension support
- ✅ Custom model loading
- ✅ Active maintenance (2024)
- ✅ ~423KB WASM binary (very reasonable!)

## Architecture

```
extension/
├── public/
│   └── fastText/
│       ├── fastText.common.wasm    # WASM binary from fasttext.wasm.js
│       └── models/
│           └── scam-detector.ftz   # Our model (690KB)
├── src/
│   ├── lib/
│   │   └── scam-detector.ts        # ScamDetector class wrapper
│   └── background.ts               # Background service worker
├── package.json
└── wxt.config.ts                   # WXT config (or manifest.json)
```

## Implementation Steps

### Phase 1: Dependencies & Setup

```bash
cd extension
npm install fasttext.wasm.js
```

### Phase 2: Extract WASM Assets

Copy from `node_modules/fasttext.wasm.js`:
- `dist/core/fastText.common.wasm` → `public/fastText/fastText.common.wasm`

Or use a build script to copy automatically:
```bash
mkdir -p extension/public/fastText/models
cp node_modules/fasttext.wasm.js/dist/core/fastText.common.wasm extension/public/fastText/
cp models/reduced/quant-cutoff10k.ftz extension/public/fastText/models/scam-detector.ftz
```

**WASM Size**: 423KB
**Model Size**: 690KB
**Total**: ~1.1MB (very reasonable for a browser extension!)

### Phase 3: Create ScamDetector Wrapper

**`src/lib/scam-detector.ts`**:

```typescript
import { getFastTextModule, getFastTextClass, type FastTextModel } from 'fasttext.wasm.js/common';

const PRODUCTION_THRESHOLD = 0.985;

export interface PredictionResult {
  isScam: boolean;
  confidence: number;
  label: string;
  rawProbability: number;
}

export class ScamDetector {
  private model: FastTextModel | null = null;
  private loaded = false;

  async load(options?: { wasmPath?: string; modelPath?: string }): Promise<void> {
    if (this.loaded) return;

    const wasmPath = options?.wasmPath ?? chrome.runtime.getURL('fastText/fastText.common.wasm');
    const modelPath = options?.modelPath ?? chrome.runtime.getURL('fastText/models/scam-detector.ftz');

    // Step 1: Initialize the WASM module with custom path
    const getFastTextModuleWithPath = () => getFastTextModule({ wasmPath });
    
    // Step 2: Get the FastText class
    const FastText = await getFastTextClass({ getFastTextModule: getFastTextModuleWithPath });
    
    // Step 3: Load the model
    const ft = new FastText();
    this.model = await ft.loadModel(modelPath);
    this.loaded = true;
  }

  async predict(text: string, threshold = PRODUCTION_THRESHOLD): Promise<PredictionResult> {
    if (!this.model) {
      throw new Error('Model not loaded. Call load() first.');
    }

    // fastText predict returns Vector<Pair<number, string>> where Pair is a tuple [number, string]:
    // - [0] is probability
    // - [1] is label (e.g., "__label__scam")
    // Use -1 to get ALL labels (don't assume binary normalized probabilities!)
    const predictions = this.model.predict(text, -1);
    
    try {
      // Scan predictions for __label__scam explicitly
      // (fastText doesn't guarantee p(scam)+p(not_scam)=1)
      let scamProb = 0;
      for (let i = 0; i < predictions.size(); i++) {
        const pair = predictions.get(i);
        const prob = pair[0];
        const label = pair[1];
        if (label === '__label__scam') {
          scamProb = prob;
          break;
        }
      }
      
      return {
        isScam: scamProb >= threshold,
        confidence: scamProb >= threshold ? scamProb : 1 - scamProb,
        label: scamProb >= threshold ? 'scam' : 'not_scam',
        rawProbability: scamProb,
      };
    } finally {
      // IMPORTANT: Free embind Vector to avoid WASM heap leak
      predictions.delete();
    }
  }

  isLoaded(): boolean {
    return this.loaded;
  }
}

// Singleton instance for background script
let instance: ScamDetector | null = null;

export async function getScamDetector(): Promise<ScamDetector> {
  if (!instance) {
    instance = new ScamDetector();
    await instance.load();
  }
  return instance;
}
```

**Note**: The `predict()` method returns a C++ Vector wrapper. Access elements with `.get(i)`. Pair types are tuples `[T1, T2]` - access with `[0]`/`[1]`, NOT `.first`/`.second`! **IMPORTANT**: Call `predictions.delete()` after use to free WASM heap memory!

### Phase 4: Background Script Integration

**`src/background.ts`**:

```typescript
import { getScamDetector, PredictionResult } from './lib/scam-detector';

// Initialize on install/startup
chrome.runtime.onInstalled.addListener(async () => {
  console.log('[ScamDetector] Initializing...');
  const detector = await getScamDetector();
  console.log('[ScamDetector] Model loaded!');
});

// Message handler for content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'CHECK_SCAM') {
    (async () => {
      try {
        const detector = await getScamDetector();
        const result = await detector.predict(message.text);
        sendResponse({ success: true, result });
      } catch (error) {
        sendResponse({ success: false, error: String(error) });
      }
    })();
    return true; // Keep channel open for async response
  }
});
```

### Phase 5: Content Script Usage

**`src/content.ts`** (example):

```typescript
async function checkText(text: string) {
  const response = await chrome.runtime.sendMessage({
    type: 'CHECK_SCAM',
    text,
  });
  
  if (response.success && response.result.isScam) {
    console.warn('🚨 Potential scam detected:', response.result);
    // Show warning UI
  }
}
```

### Phase 6: Manifest Configuration

**`manifest.json`** additions:

```json
{
  "permissions": [],
  "web_accessible_resources": [
    {
      "resources": ["fastText/**"],
      "matches": ["<all_urls>"]
    }
  ],
  "background": {
    "service_worker": "background.js",
    "type": "module"
  }
}
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `extension/package.json` | Modify | Add `fasttext.wasm.js` dependency |
| `extension/public/fastText/fastText.common.wasm` | Create | Copy WASM binary |
| `extension/public/fastText/models/scam-detector.ftz` | Create | Copy our model |
| `extension/src/lib/scam-detector.ts` | Create | ScamDetector wrapper class |
| `extension/src/background.ts` | Modify | Initialize detector, handle messages |
| `extension/manifest.json` | Modify | Add web_accessible_resources |

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Threshold | 0.985 | Tuned for 4.7% FPR |
| Model | `quant-cutoff10k.ftz` | 690KB, quantized |
| WASM | `fastText.common.wasm` | 423KB |

**Total bundle size: ~1.1MB** - very reasonable for a browser extension!

## Potential Gotchas

1. **Vector/Pair API**: fasttext.wasm.js uses C++ STL-like containers. Vector uses `.get(i)`, `.size()`, and **`.delete()` for cleanup**. **Pair is a tuple** `[T1, T2]` - access with `[0]`/`[1]`, NOT `.first`/`.second`!

2. **WASM Loading**: In Manifest V3 service workers, ensure WASM is loaded correctly. May need to use `chrome.runtime.getURL()` for paths.

3. **Model Path**: The `loadModel()` method expects a URL, not a file path. Use absolute URLs in browser context.

4. **Memory / Heap Leak**: WASM modules can be memory-intensive. **Critical**: embind objects like `Vector` returned by `predict()` must be freed with `.delete()` or you'll leak WASM heap memory! Use try/finally to ensure cleanup.

5. **Service Worker Lifecycle**: Background service workers can be terminated. Model may need to be re-loaded on wake.

6. **Probability Values**: fastText doesn't guarantee `p(scam) + p(not_scam) = 1`. Don't use `1 - topProb` as the complement. Instead, scan predictions for the specific label you want (use `predict(text, -1)` to get all labels).

## Testing Checklist

- [ ] Model loads successfully in background script
- [ ] Prediction returns correct format (check Vector/Pair access)
- [ ] Threshold of 0.985 works correctly
- [ ] Content script can communicate with background
- [ ] No CORS issues with WASM/model loading
- [ ] Memory usage is acceptable
- [ ] Service worker restart handles model reload

## Example Predictions

```typescript
await detector.predict("FREE AIRDROP! Connect wallet now!")
// → { isScam: true, confidence: 0.999, label: 'scam', rawProbability: 0.999 }

await detector.predict("The meeting is scheduled for tomorrow at 3pm")
// → { isScam: false, confidence: 0.95, label: 'not_scam', rawProbability: 0.05 }
```

## References

- [fasttext.wasm.js](https://github.com/yunsii/fasttext.wasm.js)
- [Browser extension example](https://github.com/yunsii/browser-extension-with-fasttext.wasm.js)
- [WXT Framework](https://wxt.dev/) (optional, for easier extension development)
