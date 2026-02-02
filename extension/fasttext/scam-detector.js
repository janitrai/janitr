import { FastText, fastTextReady } from './fasttext.js';

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

const predictionsToArray = (predictions) => {
  if (!predictions) return [];
  if (Array.isArray(predictions)) return predictions;
  if (typeof predictions.size === 'function' && typeof predictions.get === 'function') {
    const items = [];
    for (let i = 0; i < predictions.size(); i += 1) {
      items.push(predictions.get(i));
    }
    return items;
  }
  return [];
};

const parseScores = (items) => {
  const scores = {};
  for (const item of items) {
    if (!item) continue;
    if (Array.isArray(item) && item.length >= 2) {
      const prob = Number(item[0]);
      const label = String(item[1]);
      const key = label.replace(/^__label__/, '');
      if (!Number.isNaN(prob)) {
        scores[key] = prob;
      }
    } else if (typeof item === 'object' && 'label' in item && 'prob' in item) {
      const prob = Number(item.prob);
      const label = String(item.label || '');
      const key = label.replace(/^__label__/, '');
      if (!Number.isNaN(prob)) {
        scores[key] = prob;
      }
    }
  }
  return scores;
};

export const loadScamModel = async ({ modelUrl } = {}) => {
  if (!modelPromise) {
    modelPromise = (async () => {
      await fastTextReady;
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
  const items = predictionsToArray(rawPredictions);
  const scores = parseScores(items);
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
