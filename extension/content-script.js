const MIN_CHARS = 20;
const MAX_CHARS = 800;
const BATCH_SIZE = 4;
const LOG_INFERENCE = true;

const HOSTNAME = String(globalThis.location?.hostname || '').toLowerCase();
const IS_X = HOSTNAME === 'x.com' || HOSTNAME.endsWith('.x.com');
const X_TWEET_SELECTOR = 'div[data-testid="tweetText"]';

const queue = [];
const queued = new Set();
const lastText = new WeakMap();
let processing = false;
let stylesInjected = false;

const SKIP_TAGS = new Set([
  'SCRIPT',
  'STYLE',
  'NOSCRIPT',
  'INPUT',
  'TEXTAREA',
  'SELECT',
  'OPTION',
  'CODE',
  'PRE',
  'SVG',
  'CANVAS',
]);

const idle = () =>
  new Promise((resolve) => {
    if (typeof requestIdleCallback === 'function') {
      requestIdleCallback(() => resolve(), { timeout: 500 });
    } else {
      setTimeout(resolve, 16);
    }
  });

const normalizeText = (text) => String(text || '').replace(/\s+/g, ' ').trim();

const previewText = (text, limit = 200) => {
  const cleaned = normalizeText(text);
  if (cleaned.length <= limit) return cleaned;
  return `${cleaned.slice(0, limit)}…`;
};

const isSkippableElement = (el) => {
  if (!el || el.nodeType !== Node.ELEMENT_NODE) return true;
  if (SKIP_TAGS.has(el.tagName)) return true;
  if (el.isContentEditable) return true;
  if (el.closest('[contenteditable="true"]')) return true;
  return false;
};

const extractText = (el) => {
  if (!el) return '';
  const raw = el.innerText || el.textContent || '';
  let text = normalizeText(raw);
  if (text.length > MAX_CHARS) {
    text = text.slice(0, MAX_CHARS);
  }
  return text;
};

const getRuntime = () => {
  if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.sendMessage) {
    return chrome.runtime;
  }
  if (typeof browser !== 'undefined' && browser.runtime && browser.runtime.sendMessage) {
    return browser.runtime;
  }
  return null;
};

const sendMessage = (message) => {
  const runtime = getRuntime();
  if (!runtime) {
    return Promise.reject(new Error('Extension runtime is unavailable.'));
  }
  if (typeof runtime.sendMessage === 'function' && runtime.sendMessage.length >= 2) {
    return new Promise((resolve, reject) => {
      runtime.sendMessage(message, (response) => {
        const err = runtime.lastError;
        if (err) {
          reject(err);
          return;
        }
        resolve(response);
      });
    });
  }
  return runtime.sendMessage(message);
};

const inferBatch = async (texts) => {
  const response = await sendMessage({ type: 'ic-infer', texts });
  if (!response || !response.ok) {
    throw new Error(response?.error || 'Inference failed');
  }
  return response.results || [];
};

const pickTopLabel = (scores) => {
  let bestLabel = 'clean';
  let bestScore = -1;
  if (scores && typeof scores === 'object') {
    for (const [label, score] of Object.entries(scores)) {
      if (typeof score === 'number' && score > bestScore) {
        bestScore = score;
        bestLabel = label;
      }
    }
  }
  return { label: bestLabel, score: bestScore };
};

const clearHighlight = (el) => {
  if (!el) return;
  el.classList.remove('ic-flagged', 'ic-scam', 'ic-crypto');
  el.removeAttribute('data-ic-label');
  el.removeAttribute('data-ic-score');
  el.removeAttribute('data-ic-pscam');
  el.removeAttribute('title');
};

const applyHighlight = (el, label, score, pScam) => {
  if (!el) return;
  el.classList.add('ic-flagged');
  el.classList.toggle('ic-scam', label === 'scam');
  el.classList.toggle('ic-crypto', label === 'crypto');
  el.dataset.icLabel = label;
  if (Number.isFinite(score)) {
    el.dataset.icScore = score.toFixed(3);
  }
  if (Number.isFinite(pScam)) {
    el.dataset.icPscam = pScam.toFixed(3);
  }
  const labelText = `${label}`;
  const scoreText = Number.isFinite(score) ? `, score=${score.toFixed(3)}` : '';
  const pScamText = Number.isFinite(pScam) ? `, pScam=${pScam.toFixed(3)}` : '';
  el.title = `Scam model: ${labelText}${scoreText}${pScamText}`;
};

const buildItem = (el) => {
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

const processQueue = async () => {
  if (processing) return;
  processing = true;
  while (queue.length > 0) {
    const batch = queue.splice(0, BATCH_SIZE);
    const items = [];
    for (const el of batch) {
      queued.delete(el);
      const item = buildItem(el);
      if (item) items.push(item);
    }

    if (items.length > 0) {
      try {
        const results = await inferBatch(items.map((item) => item.text));
        for (let i = 0; i < items.length; i += 1) {
          const { el } = items[i];
          const result = results[i];
          if (!result) continue;
          const scores = result?.scores || {};
          const { label, score } = pickTopLabel(scores);
          const pScam =
            typeof scores.scam === 'number' ? scores.scam : result?.probability || 0;

          if (LOG_INFERENCE) {
            console.log('[IC] inference', {
              label,
              score: Number.isFinite(score) ? Number(score.toFixed(3)) : score,
              pScam: Number.isFinite(pScam) ? Number(pScam.toFixed(3)) : pScam,
              length: items[i].text.length,
              text: previewText(items[i].text),
            });
          }

          if (label !== 'clean') {
            applyHighlight(el, label, score, pScam);
          } else {
            clearHighlight(el);
          }
        }
      } catch (err) {
        console.warn('Scam detector inference failed', err);
      }
    }
    await idle();
  }
  processing = false;
};

const enqueueElement = (el) => {
  if (!el || isSkippableElement(el)) return;
  if (queued.has(el)) return;
  queued.add(el);
  queue.push(el);
  processQueue();
};

const scanTree = (root) => {
  if (!root) return;
  if (IS_X) {
    root.querySelectorAll(X_TWEET_SELECTOR).forEach((el) => enqueueElement(el));
    return;
  }
  const walker = document.createTreeWalker(
    root,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode: (node) => {
        if (!node || !node.parentElement) return NodeFilter.FILTER_REJECT;
        if (isSkippableElement(node.parentElement)) return NodeFilter.FILTER_REJECT;
        const text = normalizeText(node.textContent || '');
        if (text.length < MIN_CHARS) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    },
  );

  let node = walker.nextNode();
  while (node) {
    if (node.parentElement) enqueueElement(node.parentElement);
    node = walker.nextNode();
  }
};

const observeMutations = () => {
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type === 'characterData') {
        const el = mutation.target.parentElement;
        if (!el) continue;
        if (IS_X) {
          const tweet = el.closest(X_TWEET_SELECTOR);
          if (tweet) enqueueElement(tweet);
          continue;
        }
        enqueueElement(el);
      } else if (mutation.type === 'childList') {
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
            if (IS_X) {
              const el = node;
              if (el.matches && el.matches(X_TWEET_SELECTOR)) {
                enqueueElement(el);
              }
              el.querySelectorAll?.(X_TWEET_SELECTOR).forEach((tweet) => enqueueElement(tweet));
            } else {
              scanTree(node);
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

const injectStyles = () => {
  if (stylesInjected) return;
  stylesInjected = true;
  const style = document.createElement('style');
  style.textContent = `
    .ic-flagged {
      outline: 2px solid #ff4d4f;
      background: rgba(255, 77, 79, 0.12);
      border-radius: 4px;
    }
    .ic-flagged.ic-crypto {
      outline-color: #faad14;
      background: rgba(250, 173, 20, 0.12);
    }
    .ic-flagged.ic-scam {
      outline-color: #ff4d4f;
      background: rgba(255, 77, 79, 0.14);
    }
  `;
  document.documentElement.appendChild(style);
};

const init = () => {
  injectStyles();
  scanTree(document.body || document.documentElement);
  observeMutations();
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init, { once: true });
} else {
  init();
}
