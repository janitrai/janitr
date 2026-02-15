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

const ENGINE_FASTTEXT = "fasttext";
const ENGINE_TRANSFORMER = "transformer";
const ENGINE_AUTO = "auto";

const normalizeEngine = (value) => {
  const candidate = String(value || "")
    .trim()
    .toLowerCase();
  if (candidate === ENGINE_FASTTEXT) return ENGINE_FASTTEXT;
  if (candidate === ENGINE_TRANSFORMER) return ENGINE_TRANSFORMER;
  if (candidate === ENGINE_AUTO) return ENGINE_AUTO;
  return ENGINE_FASTTEXT;
};

let queue = Promise.resolve();

const enqueue = (task) => {
  const next = queue.then(task, task);
  queue = next.then(
    () => undefined,
    () => undefined,
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
  await loadTransformerModel();
  const thresholds = await loadTransformerThresholds();
  const results = await predictTransformerBatch(texts, { thresholds });
  return { results, engine: ENGINE_TRANSFORMER };
};

const classifyTexts = async (texts, engine) => {
  const requested = normalizeEngine(engine);
  if (requested === ENGINE_FASTTEXT) {
    return classifyTextsFasttext(texts);
  }
  if (requested === ENGINE_TRANSFORMER) {
    return classifyTextsTransformer(texts);
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
    return undefined;
  }

  const texts = Array.isArray(message.texts) ? message.texts : [];
  const engine = normalizeEngine(message.engine);
  enqueue(() => classifyTexts(texts, engine))
    .then((response) => {
      sendResponse({
        ok: true,
        ...response,
      });
    })
    .catch((err) => {
      resetClassifierModel();
      resetTransformerModel();
      sendResponse({
        ok: false,
        error: String(err && err.stack ? err.stack : err),
      });
    });

  return true;
});
