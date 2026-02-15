export {};

interface RuntimeMessageResponse {
  ok: boolean;
  error?: string;
  [key: string]: unknown;
}

interface ModelState {
  engine: string;
  transformerSource: {
    type: string;
    repo?: string;
    runId?: string;
  };
  defaultRepo: string;
}

interface RemoteRunItem {
  run_id: string;
  total_bytes: number;
  is_transformer_candidate?: boolean;
}

const getRuntime = (): any | null => {
  if (typeof chrome !== "undefined" && chrome.runtime?.sendMessage) {
    return chrome.runtime;
  }
  if (typeof browser !== "undefined" && browser.runtime?.sendMessage) {
    return browser.runtime;
  }
  return null;
};

const sendMessage = async <T extends RuntimeMessageResponse>(
  payload: Record<string, unknown>,
): Promise<T> => {
  const runtime = getRuntime();
  if (!runtime) {
    throw new Error("Extension runtime is unavailable.");
  }

  if (
    typeof runtime.sendMessage === "function" &&
    runtime.sendMessage.length >= 2
  ) {
    return new Promise<T>((resolve, reject) => {
      runtime.sendMessage(payload, (response: T) => {
        const err = runtime.lastError;
        if (err) {
          reject(new Error(err.message || String(err)));
          return;
        }
        resolve(response);
      });
    });
  }

  return runtime.sendMessage(payload) as Promise<T>;
};

const byId = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing required element: #${id}`);
  }
  return element as T;
};

const statusEl = byId<HTMLDivElement>("status");
const backendSelect = byId<HTMLSelectElement>("backend-select");
const activeSource = byId<HTMLSpanElement>("active-source");
const repoInput = byId<HTMLInputElement>("repo-input");
const runsSelect = byId<HTMLSelectElement>("runs-select");
const refreshRunsButton = byId<HTMLButtonElement>("refresh-runs");
const activateRunButton = byId<HTMLButtonElement>("activate-run");
const useBuiltinButton = byId<HTMLButtonElement>("use-builtin");
const openOptionsButton = byId<HTMLButtonElement>("open-options");

let busy = false;

const setBusy = (value: boolean): void => {
  busy = value;
  backendSelect.disabled = value;
  repoInput.disabled = value;
  runsSelect.disabled = value;
  refreshRunsButton.disabled = value;
  activateRunButton.disabled = value;
  useBuiltinButton.disabled = value;
  openOptionsButton.disabled = value;
};

const setStatus = (
  text: string,
  tone: "normal" | "success" | "error" = "normal",
): void => {
  statusEl.textContent = text;
  statusEl.dataset.tone = tone;
};

const bytesToMiB = (bytes: number): string =>
  `${(Number(bytes || 0) / (1024 * 1024)).toFixed(2)} MiB`;

const formatSource = (state: ModelState): string => {
  const source = state.transformerSource;
  if (source?.type === "hf_run") {
    return `${source.repo || "repo"}/${source.runId || "run"}`;
  }
  return "bundled";
};

const refreshState = async (): Promise<ModelState> => {
  const response = await sendMessage<RuntimeMessageResponse>({
    type: "ic-get-model-state",
  });
  if (!response?.ok) {
    throw new Error(response?.error || "Failed to read model state.");
  }

  const state = response as unknown as ModelState;
  backendSelect.value = String(state.engine || "transformer");
  activeSource.textContent = formatSource(state);

  if (!repoInput.value) {
    repoInput.value = String(state.defaultRepo || "janitr/experiments");
  }

  return state;
};

const refreshRuns = async (): Promise<void> => {
  const repo = String(repoInput.value || "janitr/experiments").trim();
  setStatus(`Loading runs from ${repo}...`);

  const response = await sendMessage<RuntimeMessageResponse>({
    type: "ic-list-remote-model-runs",
    repo,
  });
  if (!response?.ok) {
    throw new Error(response?.error || "Failed to list runs.");
  }

  const runs = Array.isArray(response.runs)
    ? (response.runs as RemoteRunItem[])
    : [];
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

const updateBackend = async (): Promise<void> => {
  const engine = String(backendSelect.value || "transformer");
  setStatus(`Switching backend to ${engine}...`);

  const response = await sendMessage<RuntimeMessageResponse>({
    type: "ic-set-model-backend",
    engine,
  });
  if (!response?.ok) {
    throw new Error(response?.error || "Failed to update backend.");
  }

  await refreshState();
  setStatus(`Backend set to ${engine}.`, "success");
};

const activateRun = async (): Promise<void> => {
  const repo = String(repoInput.value || "janitr/experiments").trim();
  const runId = String(runsSelect.value || "").trim();
  if (!runId) {
    setStatus("Select a run first.", "error");
    return;
  }

  setStatus(`Downloading ${runId}...`);
  const response = await sendMessage<RuntimeMessageResponse>({
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

const useBundled = async (): Promise<void> => {
  setStatus("Switching to bundled transformer...");
  const response = await sendMessage<RuntimeMessageResponse>({
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

const openOptions = async (): Promise<void> => {
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

const withBusy = (action: () => Promise<void>): void => {
  if (busy) return;
  setBusy(true);
  action()
    .catch((error: any) => {
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

const initialize = async (): Promise<void> => {
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

initialize().catch((error: any) => {
  setStatus(String(error && error.message ? error.message : error), "error");
});
