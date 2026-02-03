import { predictScam, PRODUCTION_THRESHOLD } from '../fasttext/scam-detector.js';

const output = document.getElementById('output');
const samples = [
  {
    text: '🚀 FREE AIRDROP! Connect wallet now to claim your reward.',
    note: 'scam-ish',
  },
  {
    text: 'Just finished my morning run and made breakfast. Feeling great today!',
    note: 'clean-ish',
  },
];

const run = async () => {
  try {
    const results = [];
    for (const sample of samples) {
      const result = await predictScam(sample.text, { threshold: PRODUCTION_THRESHOLD });
      results.push({ ...sample, ...result });
    }
    window.__wasmTestResults = results;
    window.__wasmTestDone = true;
    output.textContent = JSON.stringify(results, null, 2);
  } catch (err) {
    window.__wasmTestError = String(err && err.stack ? err.stack : err);
    window.__wasmTestDone = true;
    output.textContent = window.__wasmTestError;
  }
};

run();
