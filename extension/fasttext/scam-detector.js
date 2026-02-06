import {
  getFastTextClass,
  getFastTextModule,
} from "../vendor/fasttext/main/common.mjs";

const MODEL_FILENAME = "model.ftz";
const THRESHOLDS_FILENAME = "thresholds.json";
const CLASSES = ["clean", "topic_crypto", "scam", "promo"];

let modelPromise = null;
let thresholdsPromise = null;
let cachedThresholds = null;

const normalizeText = (text) => {
  if (!text) return "";
  return String(text).replace(/\s+/g, " ").trim();
};

const defaultAssetUrl = (filename) =>
  new URL(`./${filename}`, import.meta.url).toString();

const defaultModelUrl = () => defaultAssetUrl(MODEL_FILENAME);

const defaultThresholdsUrl = (modelUrl) => {
  if (modelUrl) {
    try {
      return new URL(THRESHOLDS_FILENAME, modelUrl).toString();
    } catch (err) {
      // fall through to default asset URL
    }
  }
  return defaultAssetUrl(THRESHOLDS_FILENAME);
};

const defaultWasmUrl = () => {
  if (
    typeof chrome !== "undefined" &&
    chrome.runtime &&
    chrome.runtime.getURL
  ) {
    return chrome.runtime.getURL("vendor/fasttext/core/fastText.common.wasm");
  }
  if (
    typeof browser !== "undefined" &&
    browser.runtime &&
    browser.runtime.getURL
  ) {
    return browser.runtime.getURL("vendor/fasttext/core/fastText.common.wasm");
  }
  return new URL(
    "../vendor/fasttext/core/fastText.common.wasm",
    import.meta.url,
  ).toString();
};

const parseThresholds = (payload) => {
  if (!payload || typeof payload !== "object") return null;
  const raw =
    payload.thresholds && typeof payload.thresholds === "object"
      ? payload.thresholds
      : payload;
  if (!raw || typeof raw !== "object") return null;
  const parsed = {};
  for (const label of CLASSES) {
    if (raw[label] === undefined) continue;
    const value = Number(raw[label]);
    if (Number.isFinite(value)) {
      parsed[label] = value;
    }
  }
  return Object.keys(parsed).length > 0 ? parsed : null;
};

const loadThresholds = async ({ thresholdsUrl, modelUrl } = {}) => {
  const url = thresholdsUrl || defaultThresholdsUrl(modelUrl);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to load thresholds from ${url} (${response.status})`,
    );
  }
  const payload = await response.json();
  const parsed = parseThresholds(payload);
  if (!parsed) {
    throw new Error(`Invalid thresholds payload from ${url}`);
  }
  return parsed;
};

const buildThresholds = (perLabel, globalThreshold) => {
  const thresholds = {};
  const useGlobal = Number.isFinite(globalThreshold);
  for (const label of CLASSES) {
    if (useGlobal) {
      thresholds[label] = globalThreshold;
      continue;
    }
    const value = perLabel ? perLabel[label] : undefined;
    const numeric = Number(value);
    thresholds[label] = Number.isFinite(numeric) ? numeric : 1.0;
  }
  return thresholds;
};

export const loadScamThresholds = async ({ thresholdsUrl, modelUrl } = {}) => {
  if (!thresholdsPromise) {
    thresholdsPromise = loadThresholds({ thresholdsUrl, modelUrl }).then(
      (thresholds) => {
        cachedThresholds = thresholds;
        return thresholds;
      },
    );
  }
  return thresholdsPromise;
};

export const getScamThresholds = () => cachedThresholds;

export const loadScamModel = async ({ modelUrl, thresholdsUrl } = {}) => {
  const resolvedModelUrl = modelUrl || defaultModelUrl();
  if (!modelPromise) {
    modelPromise = (async () => {
      const wasmUrl = defaultWasmUrl();
      const getFastTextModuleWithPath = () =>
        getFastTextModule({ wasmPath: wasmUrl });
      const FastText = await getFastTextClass({
        getFastTextModule: getFastTextModuleWithPath,
      });
      const ft = new FastText();
      return ft.loadModel(resolvedModelUrl);
    })();
  }
  await loadScamThresholds({ thresholdsUrl, modelUrl: resolvedModelUrl });
  return modelPromise;
};

export const resetScamModel = () => {
  modelPromise = null;
  thresholdsPromise = null;
  cachedThresholds = null;
};

export const predictScam = async (
  text,
  { thresholds, threshold, k = CLASSES.length, allowEmpty = false } = {},
) => {
  const model = await loadScamModel();
  const cleaned = normalizeText(text).replace(/\n/g, " ");
  const rawPredictions = model.predict(cleaned, k, 0.0);
  const scores = {};
  for (const label of CLASSES) {
    scores[label] = 0;
  }
  if (rawPredictions && typeof rawPredictions.size === "function") {
    try {
      for (let i = 0; i < rawPredictions.size(); i += 1) {
        const item = rawPredictions.get(i);
        if (!item) continue;
        const prob = Number(item[0]);
        const label = String(item[1]);
        const key = label.replace(/^__label__/, "");
        if (Number.isFinite(prob)) {
          const bounded = Math.min(1, Math.max(0, prob));
          if (key in scores) {
            scores[key] = bounded;
          }
        }
      }
    } finally {
      if (typeof rawPredictions.delete === "function") {
        rawPredictions.delete();
      }
    }
  }
  const perLabelThresholds = thresholds || (await loadScamThresholds());
  const appliedThresholds = buildThresholds(perLabelThresholds, threshold);
  const predicted = new Set();
  for (const label of CLASSES) {
    if ((scores[label] ?? 0) >= appliedThresholds[label]) {
      predicted.add(label);
    }
  }
  if (predicted.size === 0 && !allowEmpty) {
    if (CLASSES.includes("clean")) {
      predicted.add("clean");
    } else {
      const bestLabel = Object.entries(scores).reduce(
        (best, entry) => (entry[1] > best[1] ? entry : best),
        ["", -Infinity],
      )[0];
      if (bestLabel) {
        predicted.add(bestLabel);
      }
    }
  }
  if (predicted.has("clean") && predicted.size > 1) {
    predicted.delete("clean");
  }
  const labels = CLASSES.filter((label) => predicted.has(label));
  const pScam = scores.scam ?? 0;
  const isScam = predicted.has("scam");

  return {
    isScam,
    probability: pScam,
    threshold: appliedThresholds.scam,
    thresholds: appliedThresholds,
    label: isScam ? "scam" : labels[0] || "clean",
    labels,
    scores,
  };
};
