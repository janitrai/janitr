const MODEL_FILENAME = "student.int8.onnx";
const STUDENT_CONFIG_FILENAME = "student_config.json";
const THRESHOLDS_FILENAME = "thresholds.json";
const VOCAB_FILENAME = "tokenizer/vocab.txt";
const ORT_MODULE_RELATIVE_PATH = "../vendor/onnxruntime-web/ort.wasm.min.mjs";

const ENGINE = "transformer";
const CLASSES = ["clean", "topic_crypto", "scam"];
const DEFAULT_MAX_LENGTH = 96;

const SPECIAL_TOKENS = {
  pad: "[PAD]",
  unk: "[UNK]",
  cls: "[CLS]",
  sep: "[SEP]",
};

const ZERO_WIDTH_RE = /[\u200B-\u200D\u2060\uFEFF]/g;
const WHITESPACE_RE = /\s+/g;
const CONTROL_CODE_RE = /[\u0000-\u001F\u007F-\u009F]/;
const PUNCT_OR_SYMBOL_RE = /[\p{P}\p{S}]/u;

let ortPromise = null;
let modelPromise = null;
let configPromise = null;
let thresholdsPromise = null;
let tokenizerPromise = null;

const defaultAssetUrl = (filename) => {
  if (typeof chrome !== "undefined" && chrome.runtime?.getURL) {
    return chrome.runtime.getURL(`transformer/${filename}`);
  }
  if (typeof browser !== "undefined" && browser.runtime?.getURL) {
    return browser.runtime.getURL(`transformer/${filename}`);
  }
  return new URL(filename, import.meta.url).toString();
};

const resolveRelativeAssetUrl = (relativePath) =>
  new URL(relativePath, import.meta.url).toString();

const fetchText = async (url, label) => {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load ${label}: ${res.status} ${res.statusText}`);
  }
  return res.text();
};

const fetchJson = async (url, label) => {
  const text = await fetchText(url, label);
  try {
    return JSON.parse(text);
  } catch (err) {
    throw new Error(
      `Invalid JSON for ${label}: ${String(err && err.message ? err.message : err)}`,
    );
  }
};

const clampProb = (value) => {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, Number(value)));
};

const parseThresholds = (payload) => {
  const source =
    payload && typeof payload === "object" && payload.thresholds
      ? payload.thresholds
      : payload;
  const scam = Number(source?.scam);
  const topic = Number(source?.topic_crypto);
  return {
    scam: Number.isFinite(scam) ? clampProb(scam) : 0.5,
    topic_crypto: Number.isFinite(topic) ? clampProb(topic) : 0.5,
  };
};

const normalizeText = (text) =>
  String(text || "")
    .normalize("NFKC")
    .replace(ZERO_WIDTH_RE, "")
    .replace(WHITESPACE_RE, " ")
    .trim()
    .toLowerCase();

const isWhitespace = (char) => /\s/u.test(char);

const isControl = (char) => {
  if (char === "\t" || char === "\n" || char === "\r") {
    return false;
  }
  return CONTROL_CODE_RE.test(char);
};

const isCjkCodePoint = (codePoint) =>
  (codePoint >= 0x4e00 && codePoint <= 0x9fff) ||
  (codePoint >= 0x3400 && codePoint <= 0x4dbf) ||
  (codePoint >= 0x20000 && codePoint <= 0x2a6df) ||
  (codePoint >= 0x2a700 && codePoint <= 0x2b73f) ||
  (codePoint >= 0x2b740 && codePoint <= 0x2b81f) ||
  (codePoint >= 0x2b820 && codePoint <= 0x2ceaf) ||
  (codePoint >= 0xf900 && codePoint <= 0xfaff) ||
  (codePoint >= 0x2f800 && codePoint <= 0x2fa1f);

const isPunctuationOrSymbol = (char) => PUNCT_OR_SYMBOL_RE.test(char);

const basicTokenize = (normalized) => {
  const tokens = [];
  let current = "";

  const flush = () => {
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

const wordpieceTokenize = (token, vocabMap, maxInputCharsPerWord = 100) => {
  const tokenChars = Array.from(token);
  if (!tokenChars.length || tokenChars.length > maxInputCharsPerWord) {
    return [SPECIAL_TOKENS.unk];
  }

  const pieces = [];
  let start = 0;
  while (start < tokenChars.length) {
    let end = tokenChars.length;
    let found = null;

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

const parseVocab = (text) => {
  const vocabMap = new Map();
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

const loadTokenizer = async ({ vocabUrl, maxLength } = {}) => {
  if (!tokenizerPromise) {
    tokenizerPromise = (async () => {
      const resolvedVocabUrl = vocabUrl || defaultAssetUrl(VOCAB_FILENAME);
      const vocabText = await fetchText(resolvedVocabUrl, "transformer vocab");
      const vocabMap = parseVocab(vocabText);
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

const encodeSingle = (text, tokenizerState) => {
  const normalized = normalizeText(text);
  const basicTokens = basicTokenize(normalized);
  const subwordTokens = [];
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

const encodeBatch = (texts, tokenizerState) => {
  const flatInputIds = [];
  const flatAttention = [];

  for (const text of texts) {
    const encoded = encodeSingle(text, tokenizerState);
    flatInputIds.push(...encoded.tokenIds);
    flatAttention.push(...encoded.attentionMask);
  }

  return {
    inputIds: BigInt64Array.from(flatInputIds, (v) => BigInt(v)),
    attentionMask: BigInt64Array.from(flatAttention, (v) => BigInt(v)),
  };
};

const loadOrt = async () => {
  if (!ortPromise) {
    ortPromise = import(ORT_MODULE_RELATIVE_PATH)
      .then((mod) => mod.default || mod)
      .catch((err) => {
        throw new Error(
          `Failed to import onnxruntime-web runtime (${ORT_MODULE_RELATIVE_PATH}). ` +
            "Add runtime assets under extension/vendor/onnxruntime-web/. " +
            String(err && err.message ? err.message : err),
        );
      });
  }
  return ortPromise;
};

const resolveMaxLength = (config) => {
  const configured = Number(config?.architecture?.max_length);
  if (Number.isFinite(configured) && configured > 0) {
    return Math.floor(configured);
  }
  return DEFAULT_MAX_LENGTH;
};

export const loadTransformerThresholds = async ({ thresholdsUrl } = {}) => {
  if (!thresholdsPromise) {
    thresholdsPromise = (async () => {
      const resolvedThresholdsUrl =
        thresholdsUrl || defaultAssetUrl(THRESHOLDS_FILENAME);
      const payload = await fetchJson(
        resolvedThresholdsUrl,
        "transformer thresholds",
      );
      return parseThresholds(payload);
    })();
  }
  return thresholdsPromise;
};

export const loadTransformerModel = async ({
  modelUrl,
  studentConfigUrl,
  vocabUrl,
} = {}) => {
  if (!modelPromise) {
    modelPromise = (async () => {
      const resolvedModelUrl = modelUrl || defaultAssetUrl(MODEL_FILENAME);
      const resolvedConfigUrl =
        studentConfigUrl || defaultAssetUrl(STUDENT_CONFIG_FILENAME);

      if (!configPromise) {
        configPromise = fetchJson(
          resolvedConfigUrl,
          "transformer student config",
        );
      }
      const [ort, config] = await Promise.all([loadOrt(), configPromise]);

      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.wasm.wasmPaths = resolveRelativeAssetUrl(
        "../vendor/onnxruntime-web/",
      );

      const maxLength = resolveMaxLength(config);
      const tokenizer = await loadTokenizer({ vocabUrl, maxLength });
      const expectedVocabSize = Number(config?.architecture?.vocab_size);
      if (
        Number.isFinite(expectedVocabSize) &&
        expectedVocabSize > 0 &&
        tokenizer.vocabSize !== expectedVocabSize
      ) {
        throw new Error(
          `Tokenizer vocab size mismatch: expected ${expectedVocabSize}, got ${tokenizer.vocabSize}.`,
        );
      }

      const session = await ort.InferenceSession.create(resolvedModelUrl, {
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

const applyThresholds = (thresholds, threshold) => {
  const base = {
    scam: Number(thresholds?.scam),
    topic_crypto: Number(thresholds?.topic_crypto),
  };
  if (Number.isFinite(threshold)) {
    const v = clampProb(threshold);
    return {
      scam: v,
      topic_crypto: v,
    };
  }
  return {
    scam: Number.isFinite(base.scam) ? clampProb(base.scam) : 0.5,
    topic_crypto: Number.isFinite(base.topic_crypto)
      ? clampProb(base.topic_crypto)
      : 0.5,
  };
};

const softmaxBinaryPositive = (negLogit, posLogit) => {
  const a = Number(negLogit);
  const b = Number(posLogit);
  const max = Math.max(a, b);
  const expNeg = Math.exp(a - max);
  const expPos = Math.exp(b - max);
  return expPos / (expNeg + expPos);
};

const sigmoid = (value) => {
  const x = Number(value);
  if (!Number.isFinite(x)) return 0;
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
};

const toResult = (scamProb, topicProb, thresholds) => {
  const scam = clampProb(scamProb);
  const topic = clampProb(topicProb);
  const clean = clampProb(1 - Math.max(scam, topic));

  let label = "clean";
  if (scam >= thresholds.scam) {
    label = "scam";
  } else if (topic >= thresholds.topic_crypto) {
    label = "topic_crypto";
  }

  return {
    isFlagged: label === "scam",
    probability: scam,
    threshold: thresholds.scam,
    thresholds,
    label,
    labels: [label],
    scores: {
      clean,
      topic_crypto: topic,
      scam,
    },
    mode: ENGINE,
    classes: CLASSES,
  };
};

export const predictTransformerBatch = async (
  texts,
  { thresholds, threshold } = {},
) => {
  const safeTexts = Array.isArray(texts) ? texts : [];
  if (safeTexts.length === 0) return [];

  const [model, loadedThresholds] = await Promise.all([
    loadTransformerModel(),
    thresholds ? Promise.resolve(thresholds) : loadTransformerThresholds(),
  ]);
  const appliedThresholds = applyThresholds(loadedThresholds, threshold);

  const batchSize = safeTexts.length;
  const maxLength = model.tokenizer.maxLength;
  const encoded = encodeBatch(safeTexts, model.tokenizer);

  const feeds = {
    input_ids: new model.ort.Tensor("int64", encoded.inputIds, [
      batchSize,
      maxLength,
    ]),
    attention_mask: new model.ort.Tensor("int64", encoded.attentionMask, [
      batchSize,
      maxLength,
    ]),
  };

  const outputs = await model.session.run(feeds, [
    "scam_logits",
    "topic_logits",
  ]);
  const scamLogits = outputs?.scam_logits?.data;
  const topicLogits = outputs?.topic_logits?.data;

  if (!scamLogits || !topicLogits) {
    throw new Error("Transformer inference did not return scam/topic logits.");
  }

  const results = [];
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

export const predictTransformer = async (text, options = {}) => {
  const [result] = await predictTransformerBatch([text], options);
  return result;
};

export const resetTransformerModel = () => {
  ortPromise = null;
  modelPromise = null;
  configPromise = null;
  thresholdsPromise = null;
  tokenizerPromise = null;
};
