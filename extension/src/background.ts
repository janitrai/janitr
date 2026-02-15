import {
  DEFAULT_HF_EXPERIMENTS_REPO,
  downloadAndActivateTransformerRun,
  fetchRemoteRuns,
  getActiveTransformerSource,
  listCachedTransformerRuns,
  removeCachedTransformerRun,
  setActiveTransformerSource,
  summarizeRemoteRun,
  transformerSourceKey,
} from "./transformer/model-repo.js";
import type { Engine } from "./types.js";

const OFFSCREEN_URL = "offscreen.html";
const ENGINE_FASTTEXT: Engine = "fasttext";
const ENGINE_TRANSFORMER: Engine = "transformer";
const ENGINE_AUTO: Engine = "auto";
const DEFAULT_ENGINE: Engine = ENGINE_TRANSFORMER;
const STORAGE_MODEL_BACKEND_KEY = "ic_model_backend";

type StoragePayload = Record<string, unknown>;
type StorageAreaLike = {
  get: (
    key: string,
    callback?: (value: StoragePayload) => void,
  ) => Promise<StoragePayload> | void;
  set: (payload: StoragePayload, callback?: () => void) => Promise<void> | void;
};

const normalizeEngine = (value: unknown): Engine => {
  const candidate = String(value ?? "")
    .trim()
    .toLowerCase();
  if (candidate === ENGINE_FASTTEXT) return ENGINE_FASTTEXT;
  if (candidate === ENGINE_TRANSFORMER) return ENGINE_TRANSFORMER;
  if (candidate === ENGINE_AUTO) return ENGINE_TRANSFORMER;
  return DEFAULT_ENGINE;
};

const getStorageArea = (): StorageAreaLike | null => {
  if (typeof chrome !== "undefined" && chrome?.storage?.local) {
    return chrome.storage.local as StorageAreaLike;
  }
  if (typeof browser !== "undefined" && browser?.storage?.local) {
    return browser.storage.local as StorageAreaLike;
  }
  return null;
};

const storageGet = async (
  area: StorageAreaLike | null,
  key: string,
): Promise<StoragePayload> => {
  if (!area) return {};
  if (typeof area.get === "function" && area.get.length < 2) {
    const payload = await area.get(key);
    return (payload || {}) as StoragePayload;
  }
  return new Promise<StoragePayload>((resolve, reject) => {
    area.get(key, (value: StoragePayload) => {
      const err = chrome?.runtime?.lastError || browser?.runtime?.lastError;
      if (err) {
        reject(new Error(err.message || String(err)));
        return;
      }
      resolve((value || {}) as StoragePayload);
    });
  });
};

const storageSet = async (
  area: StorageAreaLike | null,
  payload: StoragePayload,
): Promise<void> => {
  if (!area) return;
  if (typeof area.set === "function" && area.set.length < 2) {
    await area.set(payload);
    return;
  }
  await new Promise<void>((resolve, reject) => {
    area.set(payload, () => {
      const err = chrome?.runtime?.lastError || browser?.runtime?.lastError;
      if (err) {
        reject(new Error(err.message || String(err)));
        return;
      }
      resolve();
    });
  });
};

const getConfiguredEngine = async (): Promise<Engine> => {
  const area = getStorageArea();
  if (!area) return DEFAULT_ENGINE;
  const payload = await storageGet(area, STORAGE_MODEL_BACKEND_KEY);
  return normalizeEngine(payload?.[STORAGE_MODEL_BACKEND_KEY]);
};

const setConfiguredEngine = async (engine: unknown): Promise<void> => {
  const area = getStorageArea();
  if (!area) return;
  const normalized = normalizeEngine(engine);
  await storageSet(area, {
    [STORAGE_MODEL_BACKEND_KEY]: normalized,
  });
};

const ensureOffscreen = async (): Promise<void> => {
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

const sendToOffscreen = (message: unknown): Promise<unknown> =>
  new Promise<unknown>((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response: unknown) => {
      const err = chrome.runtime.lastError;
      if (err) {
        reject(err);
        return;
      }
      resolve(response);
    });
  });

const buildModelState = async () => {
  const [engine, transformerSource, cachedRuns] = await Promise.all([
    getConfiguredEngine(),
    getActiveTransformerSource(),
    listCachedTransformerRuns(),
  ]);

  return {
    engine,
    transformerSource,
    transformerSourceKey: transformerSourceKey(transformerSource),
    cachedRuns,
    defaultRepo: DEFAULT_HF_EXPERIMENTS_REPO,
  };
};

chrome.runtime.onMessage.addListener(
  (message: any, _sender: any, sendResponse: (response: unknown) => void) => {
    if (!message) return undefined;

    if (message.type === "ic-set-model-backend") {
      (async () => {
        try {
          const engine = normalizeEngine(message.engine);
          await setConfiguredEngine(engine);
          sendResponse({ ok: true, engine });
        } catch (err: any) {
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
        } catch (err: any) {
          sendResponse({
            ok: false,
            error: String(err && err.stack ? err.stack : err),
          });
        }
      })();
      return true;
    }

    if (message.type === "ic-get-model-state") {
      (async () => {
        try {
          sendResponse({
            ok: true,
            ...(await buildModelState()),
          });
        } catch (err: any) {
          sendResponse({
            ok: false,
            error: String(err && err.stack ? err.stack : err),
          });
        }
      })();
      return true;
    }

    if (message.type === "ic-list-remote-model-runs") {
      (async () => {
        try {
          const repo = String(
            message.repo || DEFAULT_HF_EXPERIMENTS_REPO,
          ).trim();
          const runs = await fetchRemoteRuns(repo);
          sendResponse({ ok: true, repo, runs });
        } catch (err: any) {
          sendResponse({
            ok: false,
            error: String(err && err.stack ? err.stack : err),
          });
        }
      })();
      return true;
    }

    if (message.type === "ic-get-remote-model-run") {
      (async () => {
        try {
          const repo = String(
            message.repo || DEFAULT_HF_EXPERIMENTS_REPO,
          ).trim();
          const runId = String(message.runId || "").trim();
          if (!runId) {
            throw new Error("runId is required.");
          }
          const summary = await summarizeRemoteRun(repo, runId);
          sendResponse({
            ok: true,
            repo,
            runId,
            ...summary,
          });
        } catch (err: any) {
          sendResponse({
            ok: false,
            error: String(err && err.stack ? err.stack : err),
          });
        }
      })();
      return true;
    }

    if (message.type === "ic-download-and-activate-transformer-run") {
      (async () => {
        try {
          const repo = String(
            message.repo || DEFAULT_HF_EXPERIMENTS_REPO,
          ).trim();
          const runId = String(message.runId || "").trim();
          if (!runId) {
            throw new Error("runId is required.");
          }

          const result = await downloadAndActivateTransformerRun(repo, runId);
          const setBackendToTransformer =
            message.setBackendToTransformer !== false;
          if (setBackendToTransformer) {
            await setConfiguredEngine(ENGINE_TRANSFORMER);
          }

          sendResponse({
            ok: true,
            download: result,
            ...(await buildModelState()),
          });
        } catch (err: any) {
          sendResponse({
            ok: false,
            error: String(err && err.stack ? err.stack : err),
          });
        }
      })();
      return true;
    }

    if (message.type === "ic-use-builtin-transformer") {
      (async () => {
        try {
          await setActiveTransformerSource({ type: "builtin" });
          if (message.setBackendToTransformer === true) {
            await setConfiguredEngine(ENGINE_TRANSFORMER);
          }
          sendResponse({
            ok: true,
            ...(await buildModelState()),
          });
        } catch (err: any) {
          sendResponse({
            ok: false,
            error: String(err && err.stack ? err.stack : err),
          });
        }
      })();
      return true;
    }

    if (message.type === "ic-list-cached-transformer-runs") {
      (async () => {
        try {
          const cachedRuns = await listCachedTransformerRuns();
          sendResponse({ ok: true, cachedRuns });
        } catch (err: any) {
          sendResponse({
            ok: false,
            error: String(err && err.stack ? err.stack : err),
          });
        }
      })();
      return true;
    }

    if (message.type === "ic-remove-cached-transformer-run") {
      (async () => {
        try {
          const repo = String(
            message.repo || DEFAULT_HF_EXPERIMENTS_REPO,
          ).trim();
          const runId = String(message.runId || "").trim();
          if (!runId) {
            throw new Error("runId is required.");
          }

          await removeCachedTransformerRun(repo, runId);
          sendResponse({
            ok: true,
            ...(await buildModelState()),
          });
        } catch (err: any) {
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

    const rawTexts = Array.isArray(message.texts) ? message.texts : [];
    const texts = rawTexts.map((text: unknown) => String(text ?? ""));

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
      } catch (err: any) {
        sendResponse({
          ok: false,
          error: String(err && err.stack ? err.stack : err),
        });
      }
    })();

    return true;
  },
);
