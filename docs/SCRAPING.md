# X Home-Feed Scraping (OpenClaw)

## Purpose and scope
Collect posts from the X home feed via the OpenClaw-managed browser and export
JSONL records for downstream labeling. This doc covers the manual, in-browser
collection flow only.

## Prerequisites
- OpenClaw-managed browser is running.
- You are logged in to X in that browser session.

## Step-by-step flow
1. Open `https://x.com/home`.
2. Inject the JS collector snippet (below) in the devtools console.
3. Scroll slowly and wait for more posts to load.
4. Let the collector run; it will dedupe by post id.
5. Export the collected records and append them to `data/sample.jsonl`.

## Expected fields
Each JSONL line must include:
- `id`: local unique id for the dataset (e.g., `x_home_000001`).
- `text`: post text as shown in the UI.
- `source`: string tag for provenance (e.g., `x_home`).
- `url`: canonical post URL.
- `collected_at`: ISO timestamp.
- `labels`: array of labels (use `[]` at collection time).

## Example JS: collect + dedupe
```js
// In the browser console on https://x.com/home
(() => {
  const state = window.__xHomeCollector || {
    seen: new Set(),
    rows: []
  };

  function nowIso() {
    return new Date().toISOString();
  }

  function canonicalUrl(el) {
    const link = el.querySelector('a[href*="/status/"]');
    return link ? new URL(link.getAttribute('href'), location.origin).toString() : null;
  }

  function extractText(el) {
    const textEl = el.querySelector('div[data-testid="tweetText"]');
    return textEl ? textEl.innerText.trim() : "";
  }

  function collectOnce() {
    const cards = document.querySelectorAll('article');
    cards.forEach((card) => {
      const url = canonicalUrl(card);
      if (!url) return;
      const id = url.split('/status/')[1]?.split('?')[0];
      if (!id || state.seen.has(id)) return;

      const text = extractText(card);
      state.seen.add(id);
      state.rows.push({
        id,
        text,
        source: 'x_home',
        url,
        collected_at: nowIso(),
        labels: []
      });
    });

    window.__xHomeCollector = state;
    return { total: state.rows.length, newSeen: state.seen.size };
  }

  // Run once now, then you can call window.__xHomeCollect() after more scrolling.
  window.__xHomeCollect = collectOnce;
  return collectOnce();
})();
```

## Example JS: export JSONL
```js
(() => {
  const state = window.__xHomeCollector;
  if (!state || !state.rows.length) return "No rows collected.";

  const jsonl = state.rows.map((r, i) => {
    const padded = String(i + 1).padStart(6, '0');
    const localId = `x_home_${padded}`;
    return JSON.stringify({
      id: localId,
      text: r.text,
      source: r.source,
      url: r.url,
      collected_at: r.collected_at,
      labels: r.labels
    });
  }).join('\n');

  // Copy JSONL to clipboard for appending to data/sample.jsonl
  navigator.clipboard.writeText(jsonl);
  return `Copied ${state.rows.length} lines to clipboard.`;
})();
```

## Output location
Append the exported JSONL lines to `data/sample.jsonl`.
