import type { ClassifierResult, InferenceResponse, ScoreMap } from "./types.js";

const MIN_CHARS = 20;
const MAX_CHARS = 800;
const BATCH_SIZE = 4;
const LOG_INFERENCE = true;

const HOSTNAME = String(globalThis.location?.hostname || "").toLowerCase();
const IS_X = HOSTNAME === "x.com" || HOSTNAME.endsWith(".x.com");
const X_TWEET_SELECTOR = 'div[data-testid="tweetText"]';

const queue: Element[] = [];
const queued = new Set<Element>();
const lastText = new WeakMap<Element, string>();
let processing = false;
let stylesInjected = false;

const SKIP_TAGS = new Set([
  "SCRIPT",
  "STYLE",
  "NOSCRIPT",
  "INPUT",
  "TEXTAREA",
  "SELECT",
  "OPTION",
  "CODE",
  "PRE",
  "SVG",
  "CANVAS",
]);

type InferenceBatch = {
  results: ClassifierResult[];
  engine: string;
  fallbackFrom: string | null;
};

const idle = (): Promise<void> =>
  new Promise<void>((resolve) => {
    if (typeof requestIdleCallback === "function") {
      requestIdleCallback(() => resolve(), { timeout: 500 });
    } else {
      setTimeout(resolve, 16);
    }
  });

const normalizeText = (text: unknown): string =>
  String(text || "")
    .replace(/\s+/g, " ")
    .trim();

const previewText = (text: unknown, limit = 200): string => {
  const cleaned = normalizeText(text);
  if (cleaned.length <= limit) return cleaned;
  return `${cleaned.slice(0, limit)}...`;
};

const isSkippableElement = (el: Element | null | undefined): boolean => {
  if (!el || el.nodeType !== Node.ELEMENT_NODE) return true;
  if (SKIP_TAGS.has(el.tagName)) return true;
  if ((el as HTMLElement).isContentEditable) return true;
  if (el.closest('[contenteditable="true"]')) return true;
  return false;
};

const extractText = (el: Element | null | undefined): string => {
  if (!el) return "";
  const htmlElement = el as HTMLElement;
  const raw = htmlElement.innerText || el.textContent || "";
  let text = normalizeText(raw);
  if (text.length > MAX_CHARS) {
    text = text.slice(0, MAX_CHARS);
  }
  return text;
};

const getRuntime = (): any | null => {
  if (
    typeof chrome !== "undefined" &&
    chrome.runtime &&
    chrome.runtime.sendMessage
  ) {
    return chrome.runtime;
  }
  if (
    typeof browser !== "undefined" &&
    browser.runtime &&
    browser.runtime.sendMessage
  ) {
    return browser.runtime;
  }
  return null;
};

const sendMessage = (message: unknown): Promise<InferenceResponse> => {
  const runtime = getRuntime();
  if (!runtime) {
    return Promise.reject(new Error("Extension runtime is unavailable."));
  }
  if (
    typeof runtime.sendMessage === "function" &&
    runtime.sendMessage.length >= 2
  ) {
    return new Promise<InferenceResponse>((resolve, reject) => {
      runtime.sendMessage(message, (response: InferenceResponse) => {
        const err = runtime.lastError;
        if (err) {
          reject(err);
          return;
        }
        resolve(response);
      });
    });
  }
  return runtime.sendMessage(message) as Promise<InferenceResponse>;
};

const inferBatch = async (texts: string[]): Promise<InferenceBatch> => {
  const response = await sendMessage({ type: "ic-infer", texts });
  if (!response || !response.ok) {
    throw new Error(response?.error || "Inference failed");
  }
  return {
    results: Array.isArray(response.results) ? response.results : [],
    engine: typeof response.engine === "string" ? response.engine : "unknown",
    fallbackFrom:
      typeof response.fallbackFrom === "string" ? response.fallbackFrom : null,
  };
};

const formatScores = (scores: ScoreMap, limit = 4): string => {
  const entries = (Object.entries(scores) as Array<[string, number]>)
    .filter(([, score]) => Number.isFinite(score))
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([label, score]) => `${label}=${score.toFixed(3)}`);
  return entries.join(" ");
};

const clearHighlight = (el: Element | null | undefined): void => {
  if (!el) return;
  el.classList.remove("ic-flagged", "ic-scam", "ic-crypto", "ic-promo");
  el.removeAttribute("data-ic-label");
  el.removeAttribute("data-ic-labels");
  el.removeAttribute("data-ic-score");
  el.removeAttribute("data-ic-confidence");
  el.removeAttribute("title");
};

const applyHighlight = (
  el: Element | null | undefined,
  label: string,
  score: number,
  confidence: number,
  labels: string[] = [],
  scores: ScoreMap = {},
): void => {
  if (!el) return;
  const htmlElement = el as HTMLElement;
  htmlElement.classList.add("ic-flagged");
  htmlElement.classList.toggle("ic-scam", label === "scam");
  htmlElement.classList.toggle("ic-crypto", label === "topic_crypto");
  htmlElement.classList.toggle("ic-promo", label === "promo");
  htmlElement.dataset.icLabel = label;
  if (labels.length > 0) {
    htmlElement.dataset.icLabels = labels.join(" + ");
  }
  if (Number.isFinite(score)) {
    htmlElement.dataset.icScore = score.toFixed(3);
  }
  if (Number.isFinite(confidence)) {
    htmlElement.dataset.icConfidence = confidence.toFixed(3);
  }
  const labelText = labels.length > 0 ? labels.join(" + ") : `${label}`;
  const scoreText = Number.isFinite(score) ? `, score=${score.toFixed(3)}` : "";
  const confidenceText = Number.isFinite(confidence)
    ? `, confidence=${confidence.toFixed(3)}`
    : "";
  const scoreList = formatScores(scores);
  const scoreListText = scoreList ? `, scores: ${scoreList}` : "";
  htmlElement.title = `Classifier: ${labelText}${scoreText}${confidenceText}${scoreListText}`;
};

const buildItem = (el: Element): { el: Element; text: string } | null => {
  if (!document.contains(el) || isSkippableElement(el)) return null;

  const text = extractText(el);
  if (text.length < MIN_CHARS) {
    clearHighlight(el);
    return null;
  }

  if (lastText.get(el) === text) return null;
  lastText.set(el, text);

  return { el, text };
};

const processQueue = async (): Promise<void> => {
  if (processing) return;
  processing = true;
  while (queue.length > 0) {
    const batch = queue.splice(0, BATCH_SIZE);
    const items: Array<{ el: Element; text: string }> = [];
    for (const el of batch) {
      queued.delete(el);
      const item = buildItem(el);
      if (item) items.push(item);
    }

    if (items.length > 0) {
      try {
        const { results, engine, fallbackFrom } = await inferBatch(
          items.map((item) => item.text),
        );
        for (let i = 0; i < items.length; i += 1) {
          const { el } = items[i];
          const result = results[i];
          if (!result) continue;

          const scores =
            result.scores && typeof result.scores === "object"
              ? (result.scores as ScoreMap)
              : {};
          const label =
            typeof result.label === "string" ? result.label : "clean";
          const score =
            typeof scores[label] === "number" ? scores[label] : Number.NaN;
          const labelList = Array.isArray(result.labels)
            ? result.labels.filter(
                (entry): entry is string => typeof entry === "string",
              )
            : [];
          const fallbackConfidence =
            typeof result.probability === "number" ? result.probability : 0;
          const confidence =
            typeof scores.scam === "number" ? scores.scam : fallbackConfidence;

          if (LOG_INFERENCE) {
            console.log("[IC] inference", {
              engine,
              fallbackFrom,
              label,
              score: Number.isFinite(score) ? Number(score.toFixed(3)) : score,
              confidence: Number.isFinite(confidence)
                ? Number(confidence.toFixed(3))
                : confidence,
              length: items[i].text.length,
              text: previewText(items[i].text),
            });
          }

          if (label !== "clean") {
            applyHighlight(el, label, score, confidence, labelList, scores);
          } else {
            clearHighlight(el);
          }
        }
      } catch (err) {
        console.warn("Classifier inference failed", err);
      }
    }
    await idle();
  }
  processing = false;
};

const enqueueElement = (el: Element | null | undefined): void => {
  if (!el || isSkippableElement(el)) return;
  if (queued.has(el)) return;
  queued.add(el);
  queue.push(el);
  void processQueue();
};

const scanTree = (root: ParentNode | null): void => {
  if (!root) return;
  if (IS_X) {
    root.querySelectorAll(X_TWEET_SELECTOR).forEach((el) => enqueueElement(el));
    return;
  }
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => {
      if (!node || !node.parentElement) return NodeFilter.FILTER_REJECT;
      if (isSkippableElement(node.parentElement)) {
        return NodeFilter.FILTER_REJECT;
      }
      const text = normalizeText(node.textContent || "");
      if (text.length < MIN_CHARS) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  let node = walker.nextNode();
  while (node) {
    if (node.parentElement) enqueueElement(node.parentElement);
    node = walker.nextNode();
  }
};

const observeMutations = (): void => {
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type === "characterData") {
        const el = mutation.target.parentElement;
        if (!el) continue;
        if (IS_X) {
          const tweet = el.closest(X_TWEET_SELECTOR);
          if (tweet) enqueueElement(tweet);
          continue;
        }
        enqueueElement(el);
      } else if (mutation.type === "childList") {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === Node.TEXT_NODE) {
            if (!node.parentElement) return;
            if (IS_X) {
              const tweet = node.parentElement.closest(X_TWEET_SELECTOR);
              if (tweet) enqueueElement(tweet);
              return;
            }
            enqueueElement(node.parentElement);
          } else if (node.nodeType === Node.ELEMENT_NODE) {
            const el = node as Element;
            if (IS_X) {
              if (el.matches(X_TWEET_SELECTOR)) {
                enqueueElement(el);
              }
              el.querySelectorAll(X_TWEET_SELECTOR).forEach((tweet) =>
                enqueueElement(tweet),
              );
            } else {
              scanTree(el);
            }
          }
        });
      }
    }
  });

  observer.observe(document.documentElement, {
    subtree: true,
    childList: true,
    characterData: true,
  });
};

const injectStyles = (): void => {
  if (stylesInjected) return;
  stylesInjected = true;
  const style = document.createElement("style");
  style.textContent = `
    .ic-flagged {
      outline: 2px solid #ff4d4f;
      background: rgba(255, 77, 79, 0.12);
      border-radius: 4px;
      position: relative;
    }
    .ic-flagged.ic-crypto {
      outline-color: #faad14;
      background: rgba(250, 173, 20, 0.12);
    }
    .ic-flagged.ic-scam {
      outline-color: #ff4d4f;
      background: rgba(255, 77, 79, 0.14);
    }
    .ic-flagged.ic-promo {
      outline-color: #1677ff;
      background: rgba(22, 119, 255, 0.14);
    }
    .ic-flagged::after {
      content: attr(data-ic-labels);
      position: absolute;
      top: -6px;
      left: -6px;
      background: rgba(0, 0, 0, 0.75);
      color: #fff;
      padding: 2px 6px;
      border-radius: 10px;
      font-size: 10px;
      line-height: 1;
      white-space: nowrap;
      pointer-events: none;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }
  `;
  document.documentElement.appendChild(style);
};

const init = (): void => {
  injectStyles();
  scanTree(document.body || document.documentElement);
  observeMutations();
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init, { once: true });
} else {
  init();
}
