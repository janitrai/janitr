import { loadScamModel, predictScam, resetScamModel, PRODUCTION_THRESHOLD } from './fasttext/scam-detector.js';

let queue = Promise.resolve();

const enqueue = (task) => {
  const next = queue.then(task, task);
  queue = next.then(
    () => undefined,
    () => undefined,
  );
  return next;
};

const classifyTexts = async (texts) => {
  await loadScamModel();
  const results = [];
  for (const text of texts) {
    results.push(await predictScam(text, { threshold: PRODUCTION_THRESHOLD, k: 3 }));
  }
  return results;
};

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || message.type !== 'ic-infer-offscreen') {
    return undefined;
  }

  const texts = Array.isArray(message.texts) ? message.texts : [];
  enqueue(() => classifyTexts(texts))
    .then((results) => {
      sendResponse({ ok: true, results });
    })
    .catch((err) => {
      resetScamModel();
      sendResponse({ ok: false, error: String(err && err.stack ? err.stack : err) });
    });

  return true;
});
