// Generated from extension/src/*.ts by `npm run extension:build`.
import {
  loadClassifierModel,
  loadClassifierThresholds,
  predictClassifier,
  resetClassifierModel,
} from "./fasttext/classifier.js";
import {
  loadTransformerModel,
  loadTransformerThresholds,
  predictTransformerBatch,
  resetTransformerModel,
} from "./transformer/classifier-transformer.js";
import {
  getActiveTransformerSource,
  resolveTransformerAssetsForSource,
  setActiveTransformerSource,
  transformerSourceKey,
} from "./transformer/model-repo.js";
const ENGINE_FASTTEXT = "fasttext";
const ENGINE_TRANSFORMER = "transformer";
const ENGINE_AUTO = "auto";
const normalizeEngine = (value) => {
  const candidate = String(value ?? "")
    .trim()
    .toLowerCase();
  if (candidate === ENGINE_FASTTEXT) return ENGINE_FASTTEXT;
  if (candidate === ENGINE_TRANSFORMER) return ENGINE_TRANSFORMER;
  if (candidate === ENGINE_AUTO) return ENGINE_AUTO;
  return ENGINE_FASTTEXT;
};
let queue = Promise.resolve();
let activeTransformerSourceKey = null;
let activeTransformerAssets = null;
const enqueue = (task) => {
  const next = queue.then(task, task);
  queue = next.then(
    () => void 0,
    () => void 0,
  );
  return next;
};
const classifyTextsFasttext = async (texts) => {
  await loadClassifierModel();
  const thresholds = await loadClassifierThresholds();
  const results = [];
  for (const text of texts) {
    results.push(
      await predictClassifier(text, { thresholds, allowEmpty: false }),
    );
  }
  return { results, engine: ENGINE_FASTTEXT };
};
const classifyTextsTransformer = async (texts) => {
  const source = await getActiveTransformerSource();
  const sourceKey = transformerSourceKey(source);
  if (!activeTransformerAssets || activeTransformerSourceKey !== sourceKey) {
    try {
      activeTransformerAssets = await resolveTransformerAssetsForSource(source);
    } catch (error) {
      if (source.type === "hf_run") {
        if (typeof console !== "undefined" && console.warn) {
          console.warn(
            "Unable to load selected transformer run; switching back to bundled transformer.",
            error,
          );
        }
        await setActiveTransformerSource({ type: "builtin" });
        activeTransformerAssets = await resolveTransformerAssetsForSource({
          type: "builtin",
        });
      } else {
        throw error;
      }
    }
    activeTransformerSourceKey = activeTransformerAssets.sourceKey;
    resetTransformerModel();
  }
  const [model, thresholds] = await Promise.all([
    loadTransformerModel(activeTransformerAssets.modelLoadOptions),
    loadTransformerThresholds(activeTransformerAssets.thresholdLoadOptions),
  ]);
  const results = await predictTransformerBatch(texts, {
    model,
    thresholds,
  });
  return { results, engine: ENGINE_TRANSFORMER };
};
const classifyTexts = async (texts, engine) => {
  const requested = normalizeEngine(engine);
  if (requested === ENGINE_FASTTEXT) {
    return classifyTextsFasttext(texts);
  }
  if (requested === ENGINE_TRANSFORMER) {
    try {
      return await classifyTextsTransformer(texts);
    } catch (err) {
      if (typeof console !== "undefined" && console.warn) {
        console.warn(
          "Transformer inference failed, falling back to fastText.",
          err,
        );
      }
      const fallback = await classifyTextsFasttext(texts);
      return {
        ...fallback,
        fallbackFrom: ENGINE_TRANSFORMER,
        fallbackReason: String(err && err.message ? err.message : err),
      };
    }
  }
  try {
    return await classifyTextsTransformer(texts);
  } catch (err) {
    if (typeof console !== "undefined" && console.warn) {
      console.warn(
        "Transformer inference failed in auto mode, falling back to fastText.",
        err,
      );
    }
    const fallback = await classifyTextsFasttext(texts);
    return {
      ...fallback,
      fallbackFrom: ENGINE_TRANSFORMER,
      fallbackReason: String(err && err.message ? err.message : err),
    };
  }
};
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || message.type !== "ic-infer-offscreen") {
    return void 0;
  }
  const texts = Array.isArray(message.texts)
    ? message.texts.map((text) => String(text ?? ""))
    : [];
  const engine = normalizeEngine(message.engine);
  void enqueue(() => classifyTexts(texts, engine))
    .then((response) => {
      sendResponse({
        ok: true,
        ...response,
      });
    })
    .catch((err) => {
      resetClassifierModel();
      resetTransformerModel();
      activeTransformerSourceKey = null;
      activeTransformerAssets = null;
      sendResponse({
        ok: false,
        error: String(err && err.stack ? err.stack : err),
      });
    });
  return true;
});
