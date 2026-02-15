import {
  getFastTextClass,
  getFastTextModule,
} from "../vendor/fasttext/main/common.mjs";
import type { ClassifierResult, ScoreMap, ThresholdMap } from "../types.js";

const MODEL_FILENAME = "model.ftz";
const MODEL_STAGE1_FILENAME = "model.stage1.ftz";
const MODEL_STAGE2_FILENAME = "model.stage2.ftz";
const THRESHOLDS_FILENAME = "thresholds.json";
const MODE_SINGLE = "single_stage";
const MODE_TWO_STAGE = "two_stage";
const CLASSES = ["clean", "topic_crypto", "scam", "promo"] as const;

type ClassLabel = (typeof CLASSES)[number];
type ClassifierMode = typeof MODE_SINGLE | typeof MODE_TWO_STAGE;
type FastTextModel = any;

interface ClassifierModels {
  stage1: string;
  stage2: string;
}

interface ClassifierConfig {
  mode: ClassifierMode;
  thresholds: ThresholdMap;
  models: ClassifierModels;
  fallbackThresholds: ThresholdMap | null;
}

interface SingleModelBundle {
  mode: typeof MODE_SINGLE;
  model: FastTextModel;
}

interface TwoStageModelBundle {
  mode: typeof MODE_TWO_STAGE;
  stage1: FastTextModel;
  stage2: FastTextModel;
}

type ModelBundle = SingleModelBundle | TwoStageModelBundle;

interface LoadConfigOptions {
  thresholdsUrl?: string;
  modelUrl?: string;
}

interface PredictClassifierOptions {
  thresholds?: ThresholdMap;
  threshold?: number;
  k?: number;
  allowEmpty?: boolean;
}

let modelPromise: Promise<ModelBundle> | null = null;
let configPromise: Promise<ClassifierConfig> | null = null;
let cachedThresholds: ThresholdMap | null = null;
let cachedConfig: ClassifierConfig | null = null;

const normalizeText = (text: unknown): string => {
  if (!text) return "";
  return String(text).replace(/\s+/g, " ").trim();
};

const defaultAssetUrl = (filename: string): string =>
  new URL(`./${filename}`, import.meta.url).toString();

const defaultModelUrl = (): string => defaultAssetUrl(MODEL_FILENAME);

const defaultThresholdsUrl = (modelUrl?: string): string => {
  if (modelUrl) {
    try {
      return new URL(THRESHOLDS_FILENAME, modelUrl).toString();
    } catch {
      // Fall through to static asset path.
    }
  }
  return defaultAssetUrl(THRESHOLDS_FILENAME);
};

const resolveAssetUrl = (
  assetRef: string | undefined,
  baseUrl: string | undefined,
  fallbackFilename: string,
): string => {
  const candidate = assetRef || fallbackFilename;
  try {
    return new URL(candidate, baseUrl || defaultModelUrl()).toString();
  } catch {
    return defaultAssetUrl(fallbackFilename);
  }
};

const defaultWasmUrl = (): string => {
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

const parseThresholds = (payload: unknown): ThresholdMap | null => {
  if (!payload || typeof payload !== "object") return null;

  const payloadRecord = payload as Record<string, unknown>;
  const raw =
    payloadRecord.thresholds && typeof payloadRecord.thresholds === "object"
      ? (payloadRecord.thresholds as Record<string, unknown>)
      : payloadRecord;

  if (!raw || typeof raw !== "object") return null;
  const parsed: ThresholdMap = {};
  for (const label of CLASSES) {
    if (raw[label] === undefined) continue;
    const value = Number(raw[label]);
    if (Number.isFinite(value)) {
      parsed[label] = value;
    }
  }
  return Object.keys(parsed).length > 0 ? parsed : null;
};

const parseMode = (payload: unknown): ClassifierMode => {
  const mode = String(
    (payload as Record<string, unknown>)?.mode || "",
  ).toLowerCase();
  return mode === MODE_TWO_STAGE ? MODE_TWO_STAGE : MODE_SINGLE;
};

const parseModels = (payload: unknown): ClassifierModels => {
  const raw = (payload as Record<string, unknown>)?.models as
    | Record<string, unknown>
    | undefined;
  if (!raw || typeof raw !== "object") {
    return {
      stage1: MODEL_STAGE1_FILENAME,
      stage2: MODEL_STAGE2_FILENAME,
    };
  }
  return {
    stage1: String(raw.stage1 || raw.scam || MODEL_STAGE1_FILENAME),
    stage2: String(raw.stage2 || raw.topic_crypto || MODEL_STAGE2_FILENAME),
  };
};

const parseConfig = (payload: unknown, url: string): ClassifierConfig => {
  const thresholds = parseThresholds(payload);
  if (!thresholds) {
    throw new Error(`Invalid thresholds payload from ${url}`);
  }
  const fallbackThresholds = parseThresholds(
    (payload as Record<string, unknown>)?.fallback_thresholds,
  );
  return {
    mode: parseMode(payload),
    thresholds,
    models: parseModels(payload),
    fallbackThresholds,
  };
};

const loadConfig = async ({
  thresholdsUrl,
  modelUrl,
}: LoadConfigOptions = {}): Promise<ClassifierConfig> => {
  const url = thresholdsUrl || defaultThresholdsUrl(modelUrl);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to load thresholds from ${url} (${response.status})`,
    );
  }
  const payload = await response.json();
  return parseConfig(payload, url);
};

const buildThresholds = (
  perLabel: ThresholdMap | null | undefined,
  globalThreshold: number | undefined,
): ThresholdMap => {
  const thresholds: ThresholdMap = {};
  const useGlobal = Number.isFinite(globalThreshold);
  for (const label of CLASSES) {
    if (useGlobal) {
      thresholds[label] = Number(globalThreshold);
      continue;
    }
    const value = perLabel ? perLabel[label] : undefined;
    const numeric = Number(value);
    thresholds[label] = Number.isFinite(numeric) ? numeric : 1.0;
  }
  return thresholds;
};

export const loadClassifierThresholds = async ({
  thresholdsUrl,
  modelUrl,
}: LoadConfigOptions = {}): Promise<ThresholdMap> => {
  if (!configPromise) {
    configPromise = loadConfig({ thresholdsUrl, modelUrl }).then((config) => {
      cachedConfig = config;
      cachedThresholds = config.thresholds;
      return config;
    });
  }
  const config = await configPromise;
  return cachedThresholds || config.thresholds;
};

export const getClassifierThresholds = (): ThresholdMap | null =>
  cachedThresholds;
export const getClassifierConfig = (): ClassifierConfig | null => cachedConfig;

const readPredictionScores = (
  model: FastTextModel,
  text: string,
  k: number,
): ScoreMap => {
  const rawPredictions = model.predict(text, k, 0.0) as {
    size?: () => number;
    get?: (index: number) => unknown;
    delete?: () => void;
  } | null;
  const scores: ScoreMap = {};
  if (rawPredictions && typeof rawPredictions.size === "function") {
    try {
      const size = rawPredictions.size();
      for (let i = 0; i < size; i += 1) {
        const item = rawPredictions.get?.(i) as [unknown, unknown] | undefined;
        if (!item) continue;
        const prob = Number(item[0]);
        const label = String(item[1]);
        const key = label.replace(/^__label__/, "");
        if (Number.isFinite(prob)) {
          scores[key] = Math.min(1, Math.max(0, prob));
        }
      }
    } finally {
      if (typeof rawPredictions.delete === "function") {
        rawPredictions.delete();
      }
    }
  }
  return scores;
};

const buildSingleScores = (
  model: FastTextModel,
  text: string,
  k: number,
): ScoreMap => {
  const rawScores = readPredictionScores(model, text, k);
  const scores: ScoreMap = {};
  for (const label of CLASSES) {
    const value = Number(rawScores[label]);
    scores[label] = Number.isFinite(value)
      ? Math.min(1, Math.max(0, value))
      : 0;
  }
  return scores;
};

const buildTwoStageScores = (
  stage1Model: FastTextModel,
  stage2Model: FastTextModel,
  text: string,
): ScoreMap => {
  const stage1Scores = readPredictionScores(stage1Model, text, 2);
  const stage2Scores = readPredictionScores(stage2Model, text, 2);
  const scamScore = Number(stage1Scores.scam);
  const topicScore = Number(stage2Scores.topic_crypto);
  const scam = Number.isFinite(scamScore)
    ? Math.min(1, Math.max(0, scamScore))
    : 0;
  const topic = Number.isFinite(topicScore)
    ? Math.min(1, Math.max(0, topicScore))
    : 0;
  const clean = Math.min(1, Math.max(0, 1 - Math.max(scam, topic)));
  return {
    clean,
    topic_crypto: topic,
    scam,
    promo: 0,
  };
};

const pickLabels = (
  scores: ScoreMap,
  appliedThresholds: ThresholdMap,
  options: { allowEmpty: boolean; mode: ClassifierMode },
): string[] => {
  const { allowEmpty, mode } = options;
  if (mode === MODE_TWO_STAGE) {
    if ((scores.scam ?? 0) >= Number(appliedThresholds.scam ?? 1)) {
      return ["scam"];
    }
    if (
      (scores.topic_crypto ?? 0) >= Number(appliedThresholds.topic_crypto ?? 1)
    ) {
      return ["topic_crypto"];
    }
    return allowEmpty ? [] : ["clean"];
  }

  const predicted = new Set<string>();
  for (const label of CLASSES) {
    if ((scores[label] ?? 0) >= Number(appliedThresholds[label] ?? 1)) {
      predicted.add(label);
    }
  }
  if (predicted.size === 0 && !allowEmpty) {
    if (CLASSES.includes("clean")) {
      predicted.add("clean");
    } else {
      const bestLabel = (
        Object.entries(scores) as Array<[string, number]>
      ).reduce(
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
  return CLASSES.filter((label) => predicted.has(label));
};

export const loadClassifierModel = async ({
  modelUrl,
  thresholdsUrl,
}: LoadConfigOptions = {}): Promise<ModelBundle> => {
  const resolvedModelUrl = modelUrl || defaultModelUrl();
  await loadClassifierThresholds({ thresholdsUrl, modelUrl: resolvedModelUrl });
  if (!modelPromise) {
    modelPromise = (async () => {
      const config = cachedConfig || {
        mode: MODE_SINGLE,
        models: {
          stage1: MODEL_STAGE1_FILENAME,
          stage2: MODEL_STAGE2_FILENAME,
        },
        thresholds: {},
        fallbackThresholds: null,
      };

      const wasmUrl = defaultWasmUrl();
      const getFastTextModuleWithPath = () =>
        getFastTextModule({ wasmPath: wasmUrl });
      const FastText = await getFastTextClass({
        getFastTextModule: getFastTextModuleWithPath,
      });

      const loadModelFromUrl = async (url: string): Promise<FastTextModel> => {
        const ft = new FastText();
        return ft.loadModel(url);
      };

      if (config.mode === MODE_TWO_STAGE) {
        const stage1Url = resolveAssetUrl(
          config.models.stage1,
          resolvedModelUrl,
          MODEL_STAGE1_FILENAME,
        );
        const stage2Url = resolveAssetUrl(
          config.models.stage2,
          resolvedModelUrl,
          MODEL_STAGE2_FILENAME,
        );
        try {
          const [stage1, stage2] = await Promise.all([
            loadModelFromUrl(stage1Url),
            loadModelFromUrl(stage2Url),
          ]);
          return {
            mode: MODE_TWO_STAGE,
            stage1,
            stage2,
          };
        } catch (err) {
          if (config.fallbackThresholds) {
            cachedThresholds = config.fallbackThresholds;
          }
          if (typeof console !== "undefined" && console.warn) {
            console.warn(
              "Failed to load two-stage models, falling back to single model:",
              err,
            );
          }
        }
      }

      const model = await loadModelFromUrl(resolvedModelUrl);
      return {
        mode: MODE_SINGLE,
        model,
      };
    })();
  }
  return modelPromise;
};

export const resetClassifierModel = (): void => {
  modelPromise = null;
  configPromise = null;
  cachedThresholds = null;
  cachedConfig = null;
};

export const predictClassifier = async (
  text: unknown,
  {
    thresholds,
    threshold,
    k = CLASSES.length,
    allowEmpty = false,
  }: PredictClassifierOptions = {},
): Promise<ClassifierResult> => {
  const modelBundle = await loadClassifierModel();
  const cleaned = normalizeText(text).replace(/\n/g, " ");
  const perLabelThresholds = thresholds || (await loadClassifierThresholds());
  const appliedThresholds = buildThresholds(perLabelThresholds, threshold);
  const scores =
    modelBundle.mode === MODE_TWO_STAGE
      ? buildTwoStageScores(modelBundle.stage1, modelBundle.stage2, cleaned)
      : buildSingleScores(modelBundle.model, cleaned, k);

  const labels = pickLabels(scores, appliedThresholds, {
    allowEmpty,
    mode: modelBundle.mode,
  });
  const predictedLabels = new Set(labels);
  const primaryProbability = scores.scam ?? 0;
  const isFlagged = predictedLabels.has("scam");

  return {
    isFlagged,
    probability: primaryProbability,
    threshold: Number(appliedThresholds.scam ?? 1),
    thresholds: appliedThresholds,
    label: isFlagged ? "scam" : labels[0] || "clean",
    labels,
    scores,
  };
};
