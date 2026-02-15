// Generated from extension/src/*.ts by `npm run extension:build`.
const DEFAULT_REPO = "janitr/experiments";
const SCAM_SAMPLE =
  "URGENT AIRDROP: Connect wallet now and enter your seed phrase to claim rewards. Limited time, DM for instant whitelist access.";
const output = document.getElementById("output");
const globalRef = window;
const sendMessage = (message) =>
  new Promise((resolve, reject) => {
    if (!chrome?.runtime?.sendMessage) {
      reject(new Error("chrome.runtime.sendMessage is unavailable."));
      return;
    }
    chrome.runtime.sendMessage(message, (response) => {
      const err = chrome.runtime.lastError;
      if (err) {
        reject(new Error(err.message || String(err)));
        return;
      }
      resolve(response);
    });
  });
const pickRunId = (runs) => {
  const candidates = (Array.isArray(runs) ? runs : [])
    .filter((run2) => run2 && run2.is_transformer_candidate !== false)
    .map((run2) => ({
      runId: String(run2?.run_id || "").trim(),
      totalBytes: Number(run2?.total_bytes || Number.POSITIVE_INFINITY),
    }))
    .filter((run2) => run2.runId.length > 0);
  if (candidates.length === 0) return null;
  candidates.sort((left, right) => left.totalBytes - right.totalBytes);
  return candidates[0].runId;
};
const fail = (message) => {
  throw new Error(message);
};
const run = async () => {
  try {
    const listResponse = await sendMessage({
      type: "ic-list-remote-model-runs",
      repo: DEFAULT_REPO,
    });
    if (!listResponse?.ok) {
      fail(listResponse?.error || "Failed to list remote transformer runs.");
    }
    const runId = pickRunId(listResponse.runs);
    if (!runId) {
      fail(`No transformer-compatible runs found in ${DEFAULT_REPO}.`);
    }
    const activateResponse = await sendMessage({
      type: "ic-download-and-activate-transformer-run",
      repo: DEFAULT_REPO,
      runId,
      setBackendToTransformer: true,
    });
    if (!activateResponse?.ok) {
      fail(
        activateResponse?.error ||
          `Failed to download+activate run ${DEFAULT_REPO}/${runId}.`,
      );
    }
    const inferResponse = await sendMessage({
      type: "ic-infer",
      texts: [SCAM_SAMPLE],
      engine: "transformer",
    });
    if (!inferResponse?.ok) {
      fail(inferResponse?.error || "Inference failed.");
    }
    if (inferResponse.engine !== "transformer") {
      fail(
        `Expected transformer engine, got "${String(inferResponse.engine)}".`,
      );
    }
    const result = Array.isArray(inferResponse.results)
      ? inferResponse.results[0]
      : null;
    if (!result) {
      fail("No inference result returned.");
    }
    const scamScore = Number(
      (result.scores && result.scores.scam) ?? result.probability,
    );
    const scamThreshold = Number(
      (result.thresholds && result.thresholds.scam) ?? result.threshold,
    );
    if (result.label !== "scam") {
      fail(
        `Expected label "scam" for obvious scam sample, got "${String(result.label)}" (score=${String(scamScore)}, threshold=${String(scamThreshold)}).`,
      );
    }
    if (Number.isFinite(scamScore) && Number.isFinite(scamThreshold)) {
      if (scamScore < scamThreshold) {
        fail(
          `Scam score ${scamScore.toFixed(4)} is below threshold ${scamThreshold.toFixed(4)}.`,
        );
      }
    }
    const payload = {
      repo: DEFAULT_REPO,
      runId,
      engine: inferResponse.engine,
      label: result.label,
      scamScore,
      scamThreshold,
      sample: SCAM_SAMPLE,
    };
    globalRef.__transformerTestResults = payload;
    globalRef.__transformerTestDone = true;
    if (output) {
      output.textContent = JSON.stringify(payload, null, 2);
    }
  } catch (error) {
    const message = String(error && error.stack ? error.stack : error);
    globalRef.__transformerTestError = message;
    globalRef.__transformerTestDone = true;
    if (output) {
      output.textContent = message;
    }
  }
};
run();
