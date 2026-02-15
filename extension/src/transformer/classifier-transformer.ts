import type { ClassifierResult, ScoreMap, ThresholdMap } from "../types.js";

const MODEL_FILENAME = "student.int8.onnx";
const STUDENT_CONFIG_FILENAME = "student_config.json";
const THRESHOLDS_FILENAME = "thresholds.json";
const VOCAB_FILENAME = "tokenizer/vocab.txt";
const ORT_MODULE_RELATIVE_PATH = "../vendor/onnxruntime-web/ort.wasm.min.mjs";

const ENGINE = "transformer";
const CLASSES = ["clean", "topic_crypto", "scam"] as const;
const DEFAULT_MAX_LENGTH = 96;

const SPECIAL_TOKENS = {
  pad: "[PAD]",
  unk: "[UNK]",
  cls: "[CLS]",
  sep: "[SEP]",
} as const;

const ZERO_WIDTH_RE = /[\u200B-\u200D\u2060\uFEFF]/g;
const WHITESPACE_RE = /\s+/g;
const CONTROL_CODE_RE = /[\u0000-\u001F\u007F-\u009F]/;
const PUNCT_OR_SYMBOL_RE = /[\p{P}\p{S}]/u;

type TransformerThresholds = ThresholdMap & {
  scam: number;
  topic_crypto: number;
};

interface LoadTransformerOptions {
  modelUrl?: string;
  modelData?: ArrayBuffer | Uint8Array;
  studentConfigUrl?: string;
  studentConfig?: Record<string, unknown>;
  vocabUrl?: string;
  vocabText?: string;
  ortWasmPathPrefix?: string;
  cacheKey?: string;
}

interface LoadThresholdOptions {
  thresholdsUrl?: string;
  thresholdsPayload?: Record<string, unknown>;
  cacheKey?: string;
}

interface TokenizerState {
  vocabMap: Map<string, number>;
  vocabSize: number;
  maxLength: number;
  padId: number;
  unkId: number;
  clsId: number;
  sepId: number;
}

interface TransformerModel {
  ort: any;
  session: any;
  tokenizer: TokenizerState;
}

interface PredictTransformerOptions {
  thresholds?: TransformerThresholds;
  threshold?: number;
  model?: TransformerModel;
  loadOptions?: LoadTransformerOptions;
  thresholdLoadOptions?: LoadThresholdOptions;
}

let ortPromise: Promise<any> | null = null;
let modelPromise: Promise<TransformerModel> | null = null;
let configPromise: Promise<Record<string, unknown>> | null = null;
let thresholdsPromise: Promise<TransformerThresholds> | null = null;
let tokenizerPromise: Promise<TokenizerState> | null = null;
let modelCacheKey: string | null = null;
let thresholdsCacheKey: string | null = null;
let tokenizerCacheKey: string | null = null;

const defaultAssetUrl = (filename: string): string => {
  if (typeof chrome !== "undefined" && chrome.runtime?.getURL) {
    return chrome.runtime.getURL(`transformer/${filename}`);
  }
  if (typeof browser !== "undefined" && browser.runtime?.getURL) {
    return browser.runtime.getURL(`transformer/${filename}`);
  }
  return new URL(filename, import.meta.url).toString();
};

const resolveRelativeAssetUrl = (relativePath: string): string =>
  new URL(relativePath, import.meta.url).toString();

const fetchText = async (url: string, label: string): Promise<string> => {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load ${label}: ${res.status} ${res.statusText}`);
  }
  return res.text();
};

const fetchJson = async (
  url: string,
  label: string,
): Promise<Record<string, unknown>> => {
  const text = await fetchText(url, label);
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch (err: any) {
    throw new Error(
      `Invalid JSON for ${label}: ${String(err && err.message ? err.message : err)}`,
    );
  }
};

const clampProb = (value: unknown): number => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.min(1, Math.max(0, numeric));
};

const parseThresholds = (payload: unknown): TransformerThresholds => {
  const payloadRecord =
    payload && typeof payload === "object"
      ? (payload as Record<string, unknown>)
      : {};
  const source =
    payloadRecord.thresholds && typeof payloadRecord.thresholds === "object"
      ? (payloadRecord.thresholds as Record<string, unknown>)
      : payloadRecord;
  const scam = Number(source.scam);
  const topic = Number(source.topic_crypto);
  return {
    scam: Number.isFinite(scam) ? clampProb(scam) : 0.5,
    topic_crypto: Number.isFinite(topic) ? clampProb(topic) : 0.5,
  };
};

const normalizeText = (text: unknown): string =>
  String(text || "")
    .normalize("NFKC")
    .replace(ZERO_WIDTH_RE, "")
    .replace(WHITESPACE_RE, " ")
    .trim()
    .toLowerCase();

const isWhitespace = (char: string): boolean => /\s/u.test(char);

const isControl = (char: string): boolean => {
  if (char === "\t" || char === "\n" || char === "\r") {
    return false;
  }
  return CONTROL_CODE_RE.test(char);
};

const isCjkCodePoint = (codePoint: number): boolean =>
  (codePoint >= 0x4e00 && codePoint <= 0x9fff) ||
  (codePoint >= 0x3400 && codePoint <= 0x4dbf) ||
  (codePoint >= 0x20000 && codePoint <= 0x2a6df) ||
  (codePoint >= 0x2a700 && codePoint <= 0x2b73f) ||
  (codePoint >= 0x2b740 && codePoint <= 0x2b81f) ||
  (codePoint >= 0x2b820 && codePoint <= 0x2ceaf) ||
  (codePoint >= 0xf900 && codePoint <= 0xfaff) ||
  (codePoint >= 0x2f800 && codePoint <= 0x2fa1f);

const isPunctuationOrSymbol = (char: string): boolean =>
  PUNCT_OR_SYMBOL_RE.test(char);

const basicTokenize = (normalized: string): string[] => {
  const tokens: string[] = [];
  let current = "";

  const flush = (): void => {
    if (!current) return;
    tokens.push(current);
    current = "";
  };

  for (const char of normalized) {
    if (isControl(char)) {
      continue;
    }
    if (isWhitespace(char)) {
      flush();
      continue;
    }

    const codePoint = char.codePointAt(0) || 0;
    if (isCjkCodePoint(codePoint) || isPunctuationOrSymbol(char)) {
      flush();
      tokens.push(char);
      continue;
    }

    current += char;
  }
  flush();
  return tokens;
};

const wordpieceTokenize = (
  token: string,
  vocabMap: Map<string, number>,
  maxInputCharsPerWord = 100,
): string[] => {
  const tokenChars = Array.from(token);
  if (!tokenChars.length || tokenChars.length > maxInputCharsPerWord) {
    return [SPECIAL_TOKENS.unk];
  }

  const pieces: string[] = [];
  let start = 0;
  while (start < tokenChars.length) {
    let end = tokenChars.length;
    let found: string | null = null;

    while (start < end) {
      const subword = tokenChars.slice(start, end).join("");
      const candidate = start > 0 ? `##${subword}` : subword;
      if (vocabMap.has(candidate)) {
        found = candidate;
        break;
      }
      end -= 1;
    }

    if (!found) {
      return [SPECIAL_TOKENS.unk];
    }
    pieces.push(found);
    start = end;
  }

  return pieces;
};

const parseVocab = (text: string): Map<string, number> => {
  const vocabMap = new Map<string, number>();
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const token = lines[i];
    if (token.length === 0 && i === lines.length - 1) continue;
    if (!vocabMap.has(token)) {
      vocabMap.set(token, i);
    }
  }
  return vocabMap;
};

const loadTokenizer = async ({
  vocabUrl,
  vocabText,
  maxLength,
  cacheKey,
}: {
  vocabUrl?: string;
  vocabText?: string;
  maxLength?: number;
  cacheKey?: string;
} = {}): Promise<TokenizerState> => {
  const resolvedCacheKey = String(cacheKey || "");
  if (tokenizerPromise && tokenizerCacheKey !== resolvedCacheKey) {
    tokenizerPromise = null;
  }

  if (!tokenizerPromise) {
    tokenizerCacheKey = resolvedCacheKey;
    tokenizerPromise = (async () => {
      let resolvedVocabText = vocabText;
      if (typeof resolvedVocabText !== "string") {
        const resolvedVocabUrl = vocabUrl || defaultAssetUrl(VOCAB_FILENAME);
        resolvedVocabText = await fetchText(
          resolvedVocabUrl,
          "transformer vocab",
        );
      }
      const vocabMap = parseVocab(resolvedVocabText);
      const vocabSize = vocabMap.size;

      const padId = vocabMap.get(SPECIAL_TOKENS.pad);
      const unkId = vocabMap.get(SPECIAL_TOKENS.unk);
      const clsId = vocabMap.get(SPECIAL_TOKENS.cls);
      const sepId = vocabMap.get(SPECIAL_TOKENS.sep);
      if (
        !Number.isInteger(padId) ||
        !Number.isInteger(unkId) ||
        !Number.isInteger(clsId) ||
        !Number.isInteger(sepId)
      ) {
        throw new Error(
          "Tokenizer vocab is missing required special tokens ([PAD], [UNK], [CLS], [SEP]).",
        );
      }

      return {
        vocabMap,
        vocabSize,
        maxLength:
          Number.isInteger(maxLength) && maxLength > 0
            ? maxLength
            : DEFAULT_MAX_LENGTH,
        padId,
        unkId,
        clsId,
        sepId,
      };
    })();
  }
  return tokenizerPromise;
};

const encodeSingle = (
  text: string,
  tokenizerState: TokenizerState,
): { tokenIds: number[]; attentionMask: number[] } => {
  const normalized = normalizeText(text);
  const basicTokens = basicTokenize(normalized);
  const subwordTokens: string[] = [];
  for (const token of basicTokens) {
    const pieces = wordpieceTokenize(token, tokenizerState.vocabMap);
    for (const piece of pieces) subwordTokens.push(piece);
  }

  const tokenIds = [tokenizerState.clsId];
  for (const token of subwordTokens) {
    const id = tokenizerState.vocabMap.get(token);
    tokenIds.push(Number.isInteger(id) ? id : tokenizerState.unkId);
  }
  tokenIds.push(tokenizerState.sepId);

  if (tokenIds.length > tokenizerState.maxLength) {
    tokenIds.length = tokenizerState.maxLength;
    tokenIds[tokenizerState.maxLength - 1] = tokenizerState.sepId;
  }

  const attentionMask = new Array(tokenIds.length).fill(1);
  while (tokenIds.length < tokenizerState.maxLength) {
    tokenIds.push(tokenizerState.padId);
    attentionMask.push(0);
  }

  return { tokenIds, attentionMask };
};

const encodeBatch = (
  texts: string[],
  tokenizerState: TokenizerState,
): { inputIds: BigInt64Array; attentionMask: BigInt64Array } => {
  const flatInputIds: number[] = [];
  const flatAttention: number[] = [];

  for (const text of texts) {
    const encoded = encodeSingle(text, tokenizerState);
    flatInputIds.push(...encoded.tokenIds);
    flatAttention.push(...encoded.attentionMask);
  }

  return {
    inputIds: BigInt64Array.from(flatInputIds, (value) => BigInt(value)),
    attentionMask: BigInt64Array.from(flatAttention, (value) => BigInt(value)),
  };
};

const loadOrt = async (): Promise<any> => {
  if (!ortPromise) {
    ortPromise = import(ORT_MODULE_RELATIVE_PATH)
      .then((mod: any) => mod.default || mod)
      .catch((err: any) => {
        throw new Error(
          `Failed to import onnxruntime-web runtime (${ORT_MODULE_RELATIVE_PATH}). ` +
            "Ensure ort.wasm.min.mjs is packaged in extension/vendor/onnxruntime-web/. " +
            String(err && err.message ? err.message : err),
        );
      });
  }
  return ortPromise;
};

const resolveMaxLength = (config: Record<string, unknown>): number => {
  const architecture =
    config.architecture && typeof config.architecture === "object"
      ? (config.architecture as Record<string, unknown>)
      : null;
  const configured = Number(architecture?.max_length);
  if (Number.isFinite(configured) && configured > 0) {
    return Math.floor(configured);
  }
  return DEFAULT_MAX_LENGTH;
};

const resolveModelCacheKey = (options: LoadTransformerOptions): string => {
  if (typeof options.cacheKey === "string" && options.cacheKey.trim()) {
    return options.cacheKey.trim();
  }
  const modelPart = options.modelUrl || "default_model";
  const configPart = options.studentConfigUrl || "default_config";
  const vocabPart = options.vocabUrl || "default_vocab";
  const ortWasmPart = options.ortWasmPathPrefix || "default_ort_wasm";
  const hasInlineModel = options.modelData ? "inline_model" : "";
  const hasInlineConfig = options.studentConfig ? "inline_config" : "";
  const hasInlineVocab = options.vocabText ? "inline_vocab" : "";
  return [
    modelPart,
    configPart,
    vocabPart,
    ortWasmPart,
    hasInlineModel,
    hasInlineConfig,
    hasInlineVocab,
  ]
    .join("|")
    .trim();
};

const resolveThresholdCacheKey = (options: LoadThresholdOptions): string => {
  if (typeof options.cacheKey === "string" && options.cacheKey.trim()) {
    return options.cacheKey.trim();
  }
  const thresholdsPart = options.thresholdsUrl || "default_thresholds";
  const inlinePart = options.thresholdsPayload ? "inline_thresholds" : "";
  return [thresholdsPart, inlinePart].join("|").trim();
};

export const loadTransformerThresholds = async ({
  thresholdsUrl,
  thresholdsPayload,
  cacheKey,
}: LoadThresholdOptions = {}): Promise<TransformerThresholds> => {
  const resolvedCacheKey = resolveThresholdCacheKey({
    thresholdsUrl,
    thresholdsPayload,
    cacheKey,
  });
  if (thresholdsPromise && thresholdsCacheKey !== resolvedCacheKey) {
    thresholdsPromise = null;
  }

  if (!thresholdsPromise) {
    thresholdsCacheKey = resolvedCacheKey;
    thresholdsPromise = (async () => {
      let payload = thresholdsPayload;
      if (!payload) {
        const resolvedThresholdsUrl =
          thresholdsUrl || defaultAssetUrl(THRESHOLDS_FILENAME);
        payload = await fetchJson(
          resolvedThresholdsUrl,
          "transformer thresholds",
        );
      }
      return parseThresholds(payload);
    })();
  }
  return thresholdsPromise;
};

export const loadTransformerModel = async ({
  modelUrl,
  modelData,
  studentConfigUrl,
  studentConfig,
  vocabUrl,
  vocabText,
  ortWasmPathPrefix,
  cacheKey,
}: LoadTransformerOptions = {}): Promise<TransformerModel> => {
  const resolvedCacheKey = resolveModelCacheKey({
    modelUrl,
    modelData,
    studentConfigUrl,
    studentConfig,
    vocabUrl,
    vocabText,
    ortWasmPathPrefix,
    cacheKey,
  });
  if (modelPromise && modelCacheKey !== resolvedCacheKey) {
    modelPromise = null;
    configPromise = null;
    tokenizerPromise = null;
  }

  if (!modelPromise) {
    modelCacheKey = resolvedCacheKey;
    modelPromise = (async () => {
      const resolvedModelUrl = modelUrl || defaultAssetUrl(MODEL_FILENAME);
      const resolvedConfigUrl =
        studentConfigUrl || defaultAssetUrl(STUDENT_CONFIG_FILENAME);

      if (!configPromise) {
        if (studentConfig && typeof studentConfig === "object") {
          configPromise = Promise.resolve(studentConfig);
        } else {
          configPromise = fetchJson(
            resolvedConfigUrl,
            "transformer student config",
          );
        }
      }
      const [ort, config] = await Promise.all([loadOrt(), configPromise]);

      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      const defaultWasmPathPrefix = resolveRelativeAssetUrl(
        "../vendor/onnxruntime-web/",
      );
      const configuredWasmPathPrefix =
        typeof ortWasmPathPrefix === "string" && ortWasmPathPrefix.trim()
          ? ortWasmPathPrefix.trim()
          : defaultWasmPathPrefix;
      ort.env.wasm.wasmPaths = configuredWasmPathPrefix.endsWith("/")
        ? configuredWasmPathPrefix
        : `${configuredWasmPathPrefix}/`;

      const maxLength = resolveMaxLength(config);
      const tokenizer = await loadTokenizer({
        vocabUrl,
        vocabText,
        maxLength,
        cacheKey: resolvedCacheKey,
      });
      const architecture =
        config.architecture && typeof config.architecture === "object"
          ? (config.architecture as Record<string, unknown>)
          : null;
      const expectedVocabSize = Number(architecture?.vocab_size);
      if (
        Number.isFinite(expectedVocabSize) &&
        expectedVocabSize > 0 &&
        tokenizer.vocabSize !== expectedVocabSize
      ) {
        throw new Error(
          `Tokenizer vocab size mismatch: expected ${expectedVocabSize}, got ${tokenizer.vocabSize}.`,
        );
      }

      const modelSource = modelData
        ? modelData instanceof Uint8Array
          ? modelData
          : new Uint8Array(modelData)
        : resolvedModelUrl;
      const session = await ort.InferenceSession.create(modelSource, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      return {
        ort,
        session,
        tokenizer,
      };
    })();
  }
  return modelPromise;
};

const applyThresholds = (
  thresholds: TransformerThresholds | undefined,
  threshold: number | undefined,
): TransformerThresholds => {
  const base = {
    scam: Number(thresholds?.scam),
    topic_crypto: Number(thresholds?.topic_crypto),
  };
  if (Number.isFinite(threshold)) {
    const value = clampProb(threshold);
    return {
      scam: value,
      topic_crypto: value,
    };
  }
  return {
    scam: Number.isFinite(base.scam) ? clampProb(base.scam) : 0.5,
    topic_crypto: Number.isFinite(base.topic_crypto)
      ? clampProb(base.topic_crypto)
      : 0.5,
  };
};

const softmaxBinaryPositive = (
  negLogit: unknown,
  posLogit: unknown,
): number => {
  const a = Number(negLogit);
  const b = Number(posLogit);
  const max = Math.max(a, b);
  const expNeg = Math.exp(a - max);
  const expPos = Math.exp(b - max);
  return expPos / (expNeg + expPos);
};

const sigmoid = (value: unknown): number => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  if (numeric >= 0) {
    const z = Math.exp(-numeric);
    return 1 / (1 + z);
  }
  const z = Math.exp(numeric);
  return z / (1 + z);
};

const toResult = (
  scamProb: number,
  topicProb: number,
  thresholds: TransformerThresholds,
): ClassifierResult => {
  const scam = clampProb(scamProb);
  const topic = clampProb(topicProb);
  const clean = clampProb(1 - Math.max(scam, topic));

  let label = "clean";
  if (scam >= thresholds.scam) {
    label = "scam";
  } else if (topic >= thresholds.topic_crypto) {
    label = "topic_crypto";
  }

  const scores: ScoreMap = {
    clean,
    topic_crypto: topic,
    scam,
  };

  return {
    isFlagged: label === "scam",
    probability: scam,
    threshold: thresholds.scam,
    thresholds,
    label,
    labels: [label],
    scores,
    mode: ENGINE,
    classes: CLASSES,
  };
};

export const predictTransformerBatch = async (
  texts: unknown,
  {
    thresholds,
    threshold,
    model,
    loadOptions,
    thresholdLoadOptions,
  }: PredictTransformerOptions = {},
): Promise<ClassifierResult[]> => {
  const safeTexts = Array.isArray(texts)
    ? texts.map((text) => String(text ?? ""))
    : [];
  if (safeTexts.length === 0) return [];

  const [loadedModel, loadedThresholds] = await Promise.all([
    model ? Promise.resolve(model) : loadTransformerModel(loadOptions),
    thresholds
      ? Promise.resolve(thresholds)
      : loadTransformerThresholds(thresholdLoadOptions),
  ]);
  const appliedThresholds = applyThresholds(loadedThresholds, threshold);

  const batchSize = safeTexts.length;
  const maxLength = loadedModel.tokenizer.maxLength;
  const encoded = encodeBatch(safeTexts, loadedModel.tokenizer);

  const feeds = {
    input_ids: new loadedModel.ort.Tensor("int64", encoded.inputIds, [
      batchSize,
      maxLength,
    ]),
    attention_mask: new loadedModel.ort.Tensor("int64", encoded.attentionMask, [
      batchSize,
      maxLength,
    ]),
  };

  const outputs = await loadedModel.session.run(feeds, [
    "scam_logits",
    "topic_logits",
  ]);
  const scamLogits = outputs?.scam_logits?.data as
    | ArrayLike<number>
    | undefined;
  const topicLogits = outputs?.topic_logits?.data as
    | ArrayLike<number>
    | undefined;

  if (!scamLogits || !topicLogits) {
    throw new Error("Transformer inference did not return scam/topic logits.");
  }

  const results: ClassifierResult[] = [];
  for (let i = 0; i < batchSize; i += 1) {
    const scamProb = softmaxBinaryPositive(
      scamLogits[i * 2],
      scamLogits[i * 2 + 1],
    );
    const topicProb = sigmoid(topicLogits[i]);
    results.push(toResult(scamProb, topicProb, appliedThresholds));
  }
  return results;
};

export const predictTransformer = async (
  text: unknown,
  options: PredictTransformerOptions = {},
): Promise<ClassifierResult> => {
  const [result] = await predictTransformerBatch([text], options);
  return result;
};

export const resetTransformerModel = (): void => {
  ortPromise = null;
  modelPromise = null;
  configPromise = null;
  thresholdsPromise = null;
  tokenizerPromise = null;
  modelCacheKey = null;
  thresholdsCacheKey = null;
  tokenizerCacheKey = null;
};
