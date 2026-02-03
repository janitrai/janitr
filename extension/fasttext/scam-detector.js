import { getFastTextClass, getFastTextModule } from '../vendor/fasttext/main/common.mjs';

const MODEL_FILENAME = 'quant-cutoff10k.ftz';
const PRODUCTION_THRESHOLD = 0.985;

let modelPromise = null;

const normalizeText = (text) => {
  if (!text) return '';
  return String(text).replace(/\s+/g, ' ').trim();
};

const defaultModelUrl = () => {
  if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.getURL) {
    return chrome.runtime.getURL(`models/${MODEL_FILENAME}`);
  }
  if (typeof browser !== 'undefined' && browser.runtime && browser.runtime.getURL) {
    return browser.runtime.getURL(`models/${MODEL_FILENAME}`);
  }
  return new URL(`../models/${MODEL_FILENAME}`, import.meta.url).toString();
};

const defaultWasmUrl = () => {
  if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.getURL) {
    return chrome.runtime.getURL('vendor/fasttext/core/fastText.common.wasm');
  }
  if (typeof browser !== 'undefined' && browser.runtime && browser.runtime.getURL) {
    return browser.runtime.getURL('vendor/fasttext/core/fastText.common.wasm');
  }
  return new URL('../vendor/fasttext/core/fastText.common.wasm', import.meta.url).toString();
};

export const loadScamModel = async ({ modelUrl } = {}) => {
  if (!modelPromise) {
    modelPromise = (async () => {
      const wasmUrl = defaultWasmUrl();
      const getFastTextModuleWithPath = () => getFastTextModule({ wasmPath: wasmUrl });
      const FastText = await getFastTextClass({ getFastTextModule: getFastTextModuleWithPath });
      const ft = new FastText();
      return ft.loadModel(modelUrl || defaultModelUrl());
    })();
  }
  return modelPromise;
};

export const resetScamModel = () => {
  modelPromise = null;
};

export const predictScam = async (text, { threshold = PRODUCTION_THRESHOLD, k = 3 } = {}) => {
  const model = await loadScamModel();
  const cleaned = normalizeText(text).replace(/\n/g, ' ');
  const rawPredictions = model.predict(cleaned, k, 0.0);
  const scores = {};
  if (rawPredictions && typeof rawPredictions.size === 'function') {
    try {
      for (let i = 0; i < rawPredictions.size(); i += 1) {
        const item = rawPredictions.get(i);
        if (!item) continue;
        const prob = Number(item[0]);
        const label = String(item[1]);
        const key = label.replace(/^__label__/, '');
        if (!Number.isNaN(prob)) {
          scores[key] = prob;
        }
      }
    } finally {
      if (typeof rawPredictions.delete === 'function') {
        rawPredictions.delete();
      }
    }
  }
  const pScam = scores.scam ?? 0;
  const isScam = pScam >= threshold;

  return {
    isScam,
    probability: pScam,
    threshold,
    label: isScam ? 'scam' : 'clean',
    scores,
  };
};

export { PRODUCTION_THRESHOLD };
