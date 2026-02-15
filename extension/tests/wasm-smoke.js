import {
  loadClassifierThresholds,
  predictClassifier,
} from "../fasttext/classifier.js";

const output = document.getElementById("output");
const samples = [
  {
    text: "🚀 FREE AIRDROP! Connect wallet now to claim your reward.",
    note: "scam-ish",
  },
  {
    text: "Just finished my morning run and made breakfast. Feeling great today!",
    note: "clean-ish",
  },
];

const run = async () => {
  try {
    const thresholds = await loadClassifierThresholds();
    const results = [];
    for (const sample of samples) {
      const result = await predictClassifier(sample.text, { thresholds });
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
