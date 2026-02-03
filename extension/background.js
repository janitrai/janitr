const OFFSCREEN_URL = 'offscreen.html';

const ensureOffscreen = async () => {
  if (!chrome?.offscreen?.createDocument) {
    throw new Error('chrome.offscreen API is unavailable.');
  }

  if (chrome.offscreen.hasDocument) {
    const hasDoc = await chrome.offscreen.hasDocument();
    if (hasDoc) return;
  }

  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ['WORKERS'],
    justification: 'Run fastText WASM inference without page CSP limitations.',
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
  if (!message || message.type !== 'ic-infer') {
    return undefined;
  }

  const texts = Array.isArray(message.texts) ? message.texts : [];
  (async () => {
    try {
      await ensureOffscreen();
      const response = await sendToOffscreen({ type: 'ic-infer-offscreen', texts });
      sendResponse(response || { ok: false, error: 'No response from offscreen' });
    } catch (err) {
      sendResponse({
        ok: false,
        error: String(err && err.stack ? err.stack : err),
      });
    }
  })();

  return true;
});
