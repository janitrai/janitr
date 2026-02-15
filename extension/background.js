const OFFSCREEN_URL = "offscreen.html";
const ENGINE_FASTTEXT = "fasttext";
const ENGINE_TRANSFORMER = "transformer";
const ENGINE_AUTO = "auto";
const DEFAULT_ENGINE = ENGINE_TRANSFORMER;
const STORAGE_MODEL_BACKEND_KEY = "ic_model_backend";

const normalizeEngine = (value) => {
  const candidate = String(value || "")
    .trim()
    .toLowerCase();
  if (candidate === ENGINE_FASTTEXT) return ENGINE_FASTTEXT;
  if (candidate === ENGINE_TRANSFORMER) return ENGINE_TRANSFORMER;
  if (candidate === ENGINE_AUTO) return ENGINE_AUTO;
  return DEFAULT_ENGINE;
};

const getStorageArea = () => {
  if (typeof chrome !== "undefined" && chrome?.storage?.local) {
    return chrome.storage.local;
  }
  if (typeof browser !== "undefined" && browser?.storage?.local) {
    return browser.storage.local;
  }
  return null;
};

const storageGet = async (area, key) => {
  if (!area) return {};
  if (typeof area.get === "function" && area.get.length < 2) {
    return area.get(key);
  }
  return new Promise((resolve, reject) => {
    area.get(key, (value) => {
      const err = globalThis.chrome?.runtime?.lastError;
      if (err) {
        reject(new Error(err.message || String(err)));
        return;
      }
      resolve(value || {});
    });
  });
};

const storageSet = async (area, payload) => {
  if (!area) return;
  if (typeof area.set === "function" && area.set.length < 2) {
    await area.set(payload);
    return;
  }
  await new Promise((resolve, reject) => {
    area.set(payload, () => {
      const err = globalThis.chrome?.runtime?.lastError;
      if (err) {
        reject(new Error(err.message || String(err)));
        return;
      }
      resolve();
    });
  });
};

const getConfiguredEngine = async () => {
  const area = getStorageArea();
  if (!area) return DEFAULT_ENGINE;
  const payload = await storageGet(area, STORAGE_MODEL_BACKEND_KEY);
  return normalizeEngine(payload?.[STORAGE_MODEL_BACKEND_KEY]);
};

const setConfiguredEngine = async (engine) => {
  const area = getStorageArea();
  if (!area) return;
  const normalized = normalizeEngine(engine);
  await storageSet(area, {
    [STORAGE_MODEL_BACKEND_KEY]: normalized,
  });
};

const ensureOffscreen = async () => {
  if (!chrome?.offscreen?.createDocument) {
    throw new Error("chrome.offscreen API is unavailable.");
  }

  if (chrome.offscreen.hasDocument) {
    const hasDoc = await chrome.offscreen.hasDocument();
    if (hasDoc) return;
  }

  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ["WORKERS"],
    justification:
      "Run local classifier inference (fastText/transformer) without page CSP limitations.",
  });
};

const sendToOffscreen = (message) =>
  new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      const err = chrome.runtime.lastError;
      if (err) {
        reject(err);
        return;
      }
      resolve(response);
    });
  });

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message) return undefined;

  if (message.type === "ic-set-model-backend") {
    (async () => {
      try {
        const engine = normalizeEngine(message.engine);
        await setConfiguredEngine(engine);
        sendResponse({ ok: true, engine });
      } catch (err) {
        sendResponse({
          ok: false,
          error: String(err && err.stack ? err.stack : err),
        });
      }
    })();
    return true;
  }

  if (message.type === "ic-get-model-backend") {
    (async () => {
      try {
        const engine = await getConfiguredEngine();
        sendResponse({ ok: true, engine });
      } catch (err) {
        sendResponse({
          ok: false,
          error: String(err && err.stack ? err.stack : err),
        });
      }
    })();
    return true;
  }

  if (message.type !== "ic-infer") {
    return undefined;
  }

  const texts = Array.isArray(message.texts) ? message.texts : [];
  (async () => {
    try {
      const configuredEngine = await getConfiguredEngine();
      const requestedEngine = normalizeEngine(
        message.engine || configuredEngine,
      );
      await ensureOffscreen();
      const response = await sendToOffscreen({
        type: "ic-infer-offscreen",
        texts,
        engine: requestedEngine,
      });
      sendResponse(
        response || { ok: false, error: "No response from offscreen" },
      );
    } catch (err) {
      sendResponse({
        ok: false,
        error: String(err && err.stack ? err.stack : err),
      });
    }
  })();

  return true;
});
