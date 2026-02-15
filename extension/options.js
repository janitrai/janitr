// Generated from extension/src/*.ts by `npm run extension:build`.
const DEFAULT_STATUS = "Ready.";
const bytesToMiB = (bytes) => {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) return "0 MiB";
  return `${(value / (1024 * 1024)).toFixed(2)} MiB`;
};
const scoreToPercent = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "n/a";
  return `${(numeric * 100).toFixed(2)}%`;
};
const timeAgo = (iso) => {
  const ts = Date.parse(String(iso || ""));
  if (!Number.isFinite(ts)) return "unknown";
  const diffMs = Date.now() - ts;
  const diffMinutes = Math.max(0, Math.floor(diffMs / 6e4));
  if (diffMinutes < 1) return "just now";
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
};
const getRuntime = () => {
  if (typeof chrome !== "undefined" && chrome.runtime?.sendMessage) {
    return chrome.runtime;
  }
  if (typeof browser !== "undefined" && browser.runtime?.sendMessage) {
    return browser.runtime;
  }
  return null;
};
const sendMessage = async (payload) => {
  const runtime = getRuntime();
  if (!runtime) {
    throw new Error("Extension runtime is unavailable.");
  }
  if (
    typeof runtime.sendMessage === "function" &&
    runtime.sendMessage.length >= 2
  ) {
    return new Promise((resolve, reject) => {
      runtime.sendMessage(payload, (response) => {
        const err = runtime.lastError;
        if (err) {
          reject(new Error(err.message || String(err)));
          return;
        }
        resolve(response);
      });
    });
  }
  return runtime.sendMessage(payload);
};
const byId = (id) => {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing required element: #${id}`);
  }
  return element;
};
const statusEl = byId("status");
const backendSelect = byId("backend-select");
const sourceEl = byId("active-source");
const sourceKeyEl = byId("active-source-key");
const repoInput = byId("repo-input");
const refreshRunsButton = byId("refresh-runs");
const runsSelect = byId("runs-select");
const runMetaEl = byId("run-meta");
const runEvalEl = byId("run-eval");
const inspectRunButton = byId("inspect-run");
const downloadRunButton = byId("download-run");
const useBuiltinButton = byId("use-builtin");
const cachedRunsList = byId("cached-runs");
let remoteRuns = [];
let currentState = null;
let busy = false;
const setBusy = (value) => {
  busy = value;
  const disabled = value;
  backendSelect.disabled = disabled;
  refreshRunsButton.disabled = disabled;
  runsSelect.disabled = disabled;
  inspectRunButton.disabled = disabled;
  downloadRunButton.disabled = disabled;
  useBuiltinButton.disabled = disabled;
};
const setStatus = (text, tone = "normal") => {
  statusEl.textContent = text;
  statusEl.dataset.tone = tone;
};
const currentRepo = () =>
  String(
    repoInput.value || currentState?.defaultRepo || "janitr/experiments",
  ).trim();
const selectedRunId = () => String(runsSelect.value || "").trim();
const renderCachedRuns = (rows) => {
  cachedRunsList.innerHTML = "";
  if (!rows || rows.length === 0) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = "No cached transformer runs.";
    cachedRunsList.appendChild(empty);
    return;
  }
  for (const row of rows) {
    const item = document.createElement("div");
    item.className = "cached-row";
    const label = document.createElement("div");
    label.className = "cached-label";
    label.textContent = `${row.runId} · ${bytesToMiB(row.totalBytes)} · ${timeAgo(
      row.downloadedAtUtc,
    )}`;
    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.className = "danger ghost";
    removeButton.textContent = "Remove";
    removeButton.addEventListener("click", async () => {
      if (busy) return;
      setBusy(true);
      setStatus(`Removing cached run ${row.runId}...`);
      try {
        const response = await sendMessage({
          type: "ic-remove-cached-transformer-run",
          repo: row.repo,
          runId: row.runId,
        });
        if (!response?.ok) {
          throw new Error(response?.error || "Failed to remove cached run.");
        }
        await refreshState();
        setStatus(`Removed cached run ${row.runId}.`, "success");
      } catch (error) {
        setStatus(
          `Remove failed: ${String(error && error.message ? error.message : error)}`,
          "error",
        );
      } finally {
        setBusy(false);
      }
    });
    item.appendChild(label);
    item.appendChild(removeButton);
    cachedRunsList.appendChild(item);
  }
};
const formatSource = (state) => {
  if (state.transformerSource?.type === "hf_run") {
    const repo = state.transformerSource.repo || "unknown-repo";
    const runId = state.transformerSource.runId || "unknown-run";
    return `Hugging Face run: ${repo}/${runId}`;
  }
  return "Bundled transformer (extension package)";
};
const renderState = (state) => {
  currentState = state;
  backendSelect.value = String(state.engine || "transformer");
  sourceEl.textContent = formatSource(state);
  sourceKeyEl.textContent = state.transformerSourceKey || "builtin";
  if (!repoInput.value) {
    repoInput.value = String(state.defaultRepo || "janitr/experiments");
  }
  renderCachedRuns(Array.isArray(state.cachedRuns) ? state.cachedRuns : []);
};
const renderRemoteRuns = (runs) => {
  const previous = selectedRunId();
  runsSelect.innerHTML = "";
  const candidates = runs.filter(
    (run) => run.is_transformer_candidate !== false,
  );
  for (const run of candidates) {
    const option = document.createElement("option");
    option.value = run.run_id;
    option.textContent = `${run.run_id} · ${bytesToMiB(run.total_bytes)}`;
    runsSelect.appendChild(option);
  }
  if (candidates.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No transformer-compatible runs found";
    runsSelect.appendChild(option);
    runsSelect.value = "";
    return;
  }
  const hasPrevious = candidates.some((item) => item.run_id === previous);
  runsSelect.value = hasPrevious ? previous : candidates[0].run_id;
};
const renderEvalSummary = (evalSummary) => {
  if (!evalSummary) {
    runEvalEl.textContent = "No eval summary found for this run.";
    return;
  }
  runEvalEl.textContent = `scam P=${scoreToPercent(evalSummary.scam_precision)}  R=${scoreToPercent(evalSummary.scam_recall)}  F1=${scoreToPercent(evalSummary.scam_f1)}  FPR=${scoreToPercent(evalSummary.scam_fpr)}  macro F1=${scoreToPercent(evalSummary.macro_f1)}  exact=${scoreToPercent(evalSummary.exact_match)}  PR-AUC=${scoreToPercent(evalSummary.scam_pr_auc)}`;
};
const inspectRun = async () => {
  const repo = currentRepo();
  const runId = selectedRunId();
  if (!runId) {
    setStatus("Select a run first.", "error");
    return;
  }
  setStatus(`Loading run details for ${runId}...`);
  const response = await sendMessage({
    type: "ic-get-remote-model-run",
    repo,
    runId,
  });
  if (!response?.ok) {
    throw new Error(response?.error || "Unable to fetch run details.");
  }
  const runInfo = response.runInfo || {};
  const files = Array.isArray(runInfo.files) ? runInfo.files : [];
  runMetaEl.textContent = `run_id=${String(runInfo.run_id || runId)}  kind=${String(runInfo.kind || "unknown")}  files=${files.length}  total=${bytesToMiB(Number(runInfo.total_bytes || 0))}`;
  const evalSummary = response.evalSummary || null;
  renderEvalSummary(evalSummary);
  setStatus(`Loaded run metadata for ${runId}.`, "success");
};
const refreshRemoteRuns = async () => {
  const repo = currentRepo();
  setStatus(`Loading runs from ${repo}...`);
  const response = await sendMessage({
    type: "ic-list-remote-model-runs",
    repo,
  });
  if (!response?.ok) {
    throw new Error(response?.error || "Unable to list remote runs.");
  }
  remoteRuns = Array.isArray(response.runs) ? response.runs : [];
  renderRemoteRuns(remoteRuns);
  runMetaEl.textContent = "Select a run and click Inspect.";
  runEvalEl.textContent = "";
  setStatus(`Loaded ${remoteRuns.length} runs from ${repo}.`, "success");
};
const refreshState = async () => {
  const response = await sendMessage({
    type: "ic-get-model-state",
  });
  if (!response?.ok) {
    throw new Error(response?.error || "Failed to read model state.");
  }
  renderState(response);
};
const updateBackend = async () => {
  const engine = String(backendSelect.value || "transformer");
  setStatus(`Switching backend to ${engine}...`);
  const response = await sendMessage({
    type: "ic-set-model-backend",
    engine,
  });
  if (!response?.ok) {
    throw new Error(response?.error || "Failed to set backend.");
  }
  await refreshState();
  setStatus(`Backend set to ${engine}.`, "success");
};
const activateSelectedRun = async () => {
  const repo = currentRepo();
  const runId = selectedRunId();
  if (!runId) {
    setStatus("Select a run first.", "error");
    return;
  }
  setStatus(`Downloading and activating ${runId}...`);
  const response = await sendMessage({
    type: "ic-download-and-activate-transformer-run",
    repo,
    runId,
    setBackendToTransformer: true,
  });
  if (!response?.ok) {
    throw new Error(response?.error || "Failed to download and activate run.");
  }
  const download = response.download || {};
  await refreshState();
  setStatus(
    `Activated ${runId}. Downloaded ${bytesToMiB(
      Number(download.downloadedBytes || 0),
    )}, reused ${Number(download.reusedCachedFiles || 0)} files.`,
    "success",
  );
};
const useBuiltinTransformer = async () => {
  setStatus("Switching to bundled transformer...");
  const response = await sendMessage({
    type: "ic-use-builtin-transformer",
    setBackendToTransformer: true,
  });
  if (!response?.ok) {
    throw new Error(
      response?.error || "Failed to switch to built-in transformer.",
    );
  }
  await refreshState();
  setStatus("Bundled transformer is active.", "success");
};
const bindEvents = () => {
  backendSelect.addEventListener("change", () => {
    if (busy) return;
    setBusy(true);
    updateBackend()
      .catch((error) => {
        setStatus(
          `Backend change failed: ${String(error && error.message ? error.message : error)}`,
          "error",
        );
      })
      .finally(() => setBusy(false));
  });
  refreshRunsButton.addEventListener("click", () => {
    if (busy) return;
    setBusy(true);
    refreshRemoteRuns()
      .catch((error) => {
        setStatus(
          `Run refresh failed: ${String(error && error.message ? error.message : error)}`,
          "error",
        );
      })
      .finally(() => setBusy(false));
  });
  inspectRunButton.addEventListener("click", () => {
    if (busy) return;
    setBusy(true);
    inspectRun()
      .catch((error) => {
        setStatus(
          `Inspect failed: ${String(error && error.message ? error.message : error)}`,
          "error",
        );
      })
      .finally(() => setBusy(false));
  });
  downloadRunButton.addEventListener("click", () => {
    if (busy) return;
    setBusy(true);
    activateSelectedRun()
      .catch((error) => {
        setStatus(
          `Activation failed: ${String(error && error.message ? error.message : error)}`,
          "error",
        );
      })
      .finally(() => setBusy(false));
  });
  useBuiltinButton.addEventListener("click", () => {
    if (busy) return;
    setBusy(true);
    useBuiltinTransformer()
      .catch((error) => {
        setStatus(
          `Switch failed: ${String(error && error.message ? error.message : error)}`,
          "error",
        );
      })
      .finally(() => setBusy(false));
  });
};
const initialize = async () => {
  setStatus("Loading model settings...");
  setBusy(true);
  try {
    await refreshState();
    await refreshRemoteRuns();
    setStatus(DEFAULT_STATUS, "normal");
  } finally {
    setBusy(false);
  }
};
bindEvents();
initialize().catch((error) => {
  setStatus(
    `Initialization failed: ${String(error && error.message ? error.message : error)}`,
    "error",
  );
});
