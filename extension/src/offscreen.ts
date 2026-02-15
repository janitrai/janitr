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
  transformerSourceKey,
} from "./transformer/model-repo.js";
import type { ClassifierResult, Engine } from "./types.js";

const ENGINE_FASTTEXT: Engine = "fasttext";
const ENGINE_TRANSFORMER: Engine = "transformer";
const ENGINE_AUTO: Engine = "auto";

const normalizeEngine = (value: unknown): Engine => {
  const candidate = String(value ?? "")
    .trim()
    .toLowerCase();
  if (candidate === ENGINE_FASTTEXT) return ENGINE_FASTTEXT;
  if (candidate === ENGINE_TRANSFORMER) return ENGINE_TRANSFORMER;
  if (candidate === ENGINE_AUTO) return ENGINE_TRANSFORMER;
  return ENGINE_TRANSFORMER;
};

let queue: Promise<void> = Promise.resolve();
let activeTransformerSourceKey: string | null = null;
let activeTransformerAssets: Awaited<
  ReturnType<typeof resolveTransformerAssetsForSource>
> | null = null;

const enqueue = <T>(task: () => Promise<T>): Promise<T> => {
  const next = queue.then(task, task);
  queue = next.then(
    () => undefined,
    () => undefined,
  );
  return next;
};

const classifyTextsFasttext = async (
  texts: string[],
): Promise<{ results: ClassifierResult[]; engine: Engine }> => {
  await loadClassifierModel();
  const thresholds = await loadClassifierThresholds();
  const results: ClassifierResult[] = [];
  for (const text of texts) {
    results.push(
      await predictClassifier(text, { thresholds, allowEmpty: false }),
    );
  }
  return { results, engine: ENGINE_FASTTEXT };
};

const classifyTextsTransformer = async (
  texts: string[],
): Promise<{ results: ClassifierResult[]; engine: Engine }> => {
  const source = await getActiveTransformerSource();
  const sourceKey = transformerSourceKey(source);

  if (!activeTransformerAssets || activeTransformerSourceKey !== sourceKey) {
    activeTransformerAssets = await resolveTransformerAssetsForSource(source);
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

type ClassifiedResponse = {
  results: ClassifierResult[];
  engine: Engine;
};

const resetTransformerRuntimeState = (): void => {
  resetTransformerModel();
  activeTransformerSourceKey = null;
  activeTransformerAssets = null;
};

const classifyTexts = async (
  texts: string[],
  engine: unknown,
): Promise<ClassifiedResponse> => {
  const requested = normalizeEngine(engine);
  if (requested === ENGINE_FASTTEXT) {
    return classifyTextsFasttext(texts);
  }
  try {
    return await classifyTextsTransformer(texts);
  } catch (err: any) {
    resetTransformerRuntimeState();
    const reason = String(err && err.message ? err.message : err);
    throw new Error(`Transformer inference failed: ${reason}`);
  }
};

chrome.runtime.onMessage.addListener(
  (message: any, _sender: any, sendResponse: (response: unknown) => void) => {
    if (!message || message.type !== "ic-infer-offscreen") {
      return undefined;
    }

    const texts = Array.isArray(message.texts)
      ? message.texts.map((text: unknown) => String(text ?? ""))
      : [];
    const engine = normalizeEngine(message.engine);
    void enqueue(() => classifyTexts(texts, engine))
      .then((response) => {
        sendResponse({
          ok: true,
          ...response,
        });
      })
      .catch((err: any) => {
        resetClassifierModel();
        resetTransformerRuntimeState();
        sendResponse({
          ok: false,
          error: String(err && err.stack ? err.stack : err),
        });
      });

    return true;
  },
);
