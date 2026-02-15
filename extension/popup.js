// Generated from extension/src/*.ts by `npm run extension:build`.
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
const activeSource = byId("active-source");
const repoInput = byId("repo-input");
const runsSelect = byId("runs-select");
const refreshRunsButton = byId("refresh-runs");
const activateRunButton = byId("activate-run");
const useBuiltinButton = byId("use-builtin");
const openOptionsButton = byId("open-options");
let busy = false;
const setBusy = (value) => {
  busy = value;
  backendSelect.disabled = value;
  repoInput.disabled = value;
  runsSelect.disabled = value;
  refreshRunsButton.disabled = value;
  activateRunButton.disabled = value;
  useBuiltinButton.disabled = value;
  openOptionsButton.disabled = value;
};
const setStatus = (text, tone = "normal") => {
  statusEl.textContent = text;
  statusEl.dataset.tone = tone;
};
const bytesToMiB = (bytes) =>
  `${(Number(bytes || 0) / (1024 * 1024)).toFixed(2)} MiB`;
const formatSource = (state) => {
  const source = state.transformerSource;
  if (source?.type === "hf_run") {
    return `${source.repo || "repo"}/${source.runId || "run"}`;
  }
  return "bundled";
};
const refreshState = async () => {
  const response = await sendMessage({
    type: "ic-get-model-state",
  });
  if (!response?.ok) {
    throw new Error(response?.error || "Failed to read model state.");
  }
  const state = response;
  backendSelect.value = String(state.engine || "transformer");
  activeSource.textContent = formatSource(state);
  if (!repoInput.value) {
    repoInput.value = String(state.defaultRepo || "janitr/experiments");
  }
  return state;
};
const refreshRuns = async () => {
  const repo = String(repoInput.value || "janitr/experiments").trim();
  setStatus(`Loading runs from ${repo}...`);
  const response = await sendMessage({
    type: "ic-list-remote-model-runs",
    repo,
  });
  if (!response?.ok) {
    throw new Error(response?.error || "Failed to list runs.");
  }
  const runs = Array.isArray(response.runs) ? response.runs : [];
  const candidates = runs.filter(
    (run) => run.is_transformer_candidate !== false,
  );
  runsSelect.innerHTML = "";
  if (candidates.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No compatible runs";
    runsSelect.appendChild(option);
    setStatus("No compatible runs found.");
    return;
  }
  for (const run of candidates) {
    const option = document.createElement("option");
    option.value = run.run_id;
    option.textContent = `${run.run_id} · ${bytesToMiB(run.total_bytes)}`;
    runsSelect.appendChild(option);
  }
  setStatus(`Loaded ${candidates.length} runs.`, "success");
};
const updateBackend = async () => {
  const engine = String(backendSelect.value || "transformer");
  setStatus(`Switching backend to ${engine}...`);
  const response = await sendMessage({
    type: "ic-set-model-backend",
    engine,
  });
  if (!response?.ok) {
    throw new Error(response?.error || "Failed to update backend.");
  }
  await refreshState();
  setStatus(`Backend set to ${engine}.`, "success");
};
const activateRun = async () => {
  const repo = String(repoInput.value || "janitr/experiments").trim();
  const runId = String(runsSelect.value || "").trim();
  if (!runId) {
    setStatus("Select a run first.", "error");
    return;
  }
  setStatus(`Downloading ${runId}...`);
  const response = await sendMessage({
    type: "ic-download-and-activate-transformer-run",
    repo,
    runId,
    setBackendToTransformer: true,
  });
  if (!response?.ok) {
    throw new Error(response?.error || "Failed to activate run.");
  }
  await refreshState();
  setStatus(`Activated ${runId}.`, "success");
};
const useBundled = async () => {
  setStatus("Switching to bundled transformer...");
  const response = await sendMessage({
    type: "ic-use-builtin-transformer",
    setBackendToTransformer: true,
  });
  if (!response?.ok) {
    throw new Error(
      response?.error || "Failed to switch to bundled transformer.",
    );
  }
  await refreshState();
  setStatus("Bundled transformer active.", "success");
};
const openOptions = async () => {
  if (typeof chrome !== "undefined" && chrome.runtime?.openOptionsPage) {
    await chrome.runtime.openOptionsPage();
    return;
  }
  if (typeof browser !== "undefined" && browser.runtime?.openOptionsPage) {
    await browser.runtime.openOptionsPage();
    return;
  }
  window.open("options.html", "_blank");
};
const withBusy = (action) => {
  if (busy) return;
  setBusy(true);
  action()
    .catch((error) => {
      setStatus(
        String(error && error.message ? error.message : error),
        "error",
      );
    })
    .finally(() => setBusy(false));
};
backendSelect.addEventListener("change", () => withBusy(updateBackend));
refreshRunsButton.addEventListener("click", () => withBusy(refreshRuns));
activateRunButton.addEventListener("click", () => withBusy(activateRun));
useBuiltinButton.addEventListener("click", () => withBusy(useBundled));
openOptionsButton.addEventListener("click", () => withBusy(openOptions));
const initialize = async () => {
  setBusy(true);
  setStatus("Loading...");
  try {
    await refreshState();
    await refreshRuns();
    setStatus("Ready.");
  } finally {
    setBusy(false);
  }
};
initialize().catch((error) => {
  setStatus(String(error && error.message ? error.message : error), "error");
});
