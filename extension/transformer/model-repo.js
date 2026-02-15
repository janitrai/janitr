// Generated from extension/src/*.ts by `npm run extension:build`.
const DEFAULT_SOURCE_TYPE = "builtin";
const HF_SOURCE_TYPE = "hf_run";
const DEFAULT_HF_EXPERIMENTS_REPO = "janitr/experiments";
const STORAGE_TRANSFORMER_SOURCE_KEY = "ic_transformer_source";
const REQUIRED_TRANSFORMER_DESTINATIONS = [
  "model/model.int8.onnx",
  "model/student_config.json",
  "model/tokenizer/vocab.txt",
  "model/thresholds.json",
];
const DB_NAME = "janitr_transformer_models";
const DB_VERSION = 1;
const RUN_STORE = "runs";
const ASSET_STORE = "assets";
const getRuntime = () => {
  if (typeof chrome !== "undefined") {
    return chrome;
  }
  if (typeof browser !== "undefined") {
    return browser;
  }
  return null;
};
const getStorageArea = () => {
  const runtime = getRuntime();
  if (runtime?.storage?.local) {
    return runtime.storage.local;
  }
  return null;
};
const storageGet = async (area, key) => {
  if (!area) return {};
  if (typeof area.get === "function" && area.get.length < 2) {
    const payload = await area.get(key);
    return payload || {};
  }
  return new Promise((resolve, reject) => {
    area.get(key, (value) => {
      const err = chrome?.runtime?.lastError || browser?.runtime?.lastError;
      if (err) {
        reject(new Error(err.message || String(err)));
        return;
      }
      resolve(value || {});
    });
  });
};
const storageSet = async (area, payload) => {
  if (!area) return;
  if (typeof area.set === "function" && area.set.length < 2) {
    await area.set(payload);
    return;
  }
  await new Promise((resolve, reject) => {
    area.set(payload, () => {
      const err = chrome?.runtime?.lastError || browser?.runtime?.lastError;
      if (err) {
        reject(new Error(err.message || String(err)));
        return;
      }
      resolve();
    });
  });
};
const toPositiveInt = (value, fallback = 0) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) return fallback;
  return Math.floor(numeric);
};
const normalizeRepo = (repo) => {
  const raw = String(repo || DEFAULT_HF_EXPERIMENTS_REPO)
    .trim()
    .replace(/^\/+/, "")
    .replace(/\/+$/, "");
  return raw || DEFAULT_HF_EXPERIMENTS_REPO;
};
const normalizeRunId = (runId) => String(runId || "").trim();
const normalizeHash = (value) =>
  String(value || "")
    .trim()
    .toLowerCase();
const normalizeDestination = (value) =>
  String(value || "")
    .trim()
    .replace(/^\/+/, "");
const ensureRecord = (value) =>
  value && typeof value === "object" ? value : {};
const buildRunCacheKey = (repo, runId) =>
  `${normalizeRepo(repo)}::${normalizeRunId(runId)}`;
const buildAssetCacheKey = (repo, runId, destination) =>
  `${buildRunCacheKey(repo, runId)}::${normalizeDestination(destination)}`;
const buildResolveUrl = (repo, path) => {
  const safeRepo = normalizeRepo(repo);
  const safePath = String(path || "").replace(/^\/+/, "");
  return `https://huggingface.co/${encodeURI(safeRepo)}/resolve/main/${safePath}`;
};
const runInfoUrl = (repo, runId) =>
  buildResolveUrl(repo, `runs/${runId}/RUN_INFO.json`);
const runFileUrl = (repo, runId, destination) =>
  buildResolveUrl(repo, `runs/${runId}/${normalizeDestination(destination)}`);
const fetchText = async (url, label) => {
  const response = await fetch(url, {
    method: "GET",
    cache: "no-store",
    redirect: "follow",
  });
  if (!response.ok) {
    throw new Error(
      `Failed to fetch ${label}: ${response.status} ${response.statusText}`,
    );
  }
  return response.text();
};
const fetchJson = async (url, label) => {
  const text = await fetchText(url, label);
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(
      `Invalid JSON in ${label}: ${String(error && error.message ? error.message : error)}`,
    );
  }
};
const parseRunFile = (value, index) => {
  const record = ensureRecord(value);
  const destination = normalizeDestination(record.destination);
  const sha256 = normalizeHash(record.sha256);
  const sizeBytes = toPositiveInt(record.size_bytes, -1);
  if (!destination || !sha256 || sizeBytes < 0) {
    return null;
  }
  return {
    source: String(record.source || ""),
    destination,
    size_bytes: sizeBytes,
    sha256,
  };
};
const parseRunInfo = (payload) => {
  const record = ensureRecord(payload);
  const runId = normalizeRunId(record.run_id);
  if (!runId) {
    throw new Error("RUN_INFO.json is missing run_id.");
  }
  const rawFiles = Array.isArray(record.files) ? record.files : [];
  const files = [];
  for (let i = 0; i < rawFiles.length; i += 1) {
    const parsed = parseRunFile(rawFiles[i], i);
    if (parsed) files.push(parsed);
  }
  if (files.length === 0) {
    throw new Error(`RUN_INFO for ${runId} has no valid files list.`);
  }
  return {
    schema_version: toPositiveInt(record.schema_version, 0),
    run_id: runId,
    kind: String(record.kind || ""),
    total_bytes: toPositiveInt(record.total_bytes, 0),
    source_run_dir: String(record.source_run_dir || ""),
    created_at_utc: String(record.created_at_utc || ""),
    files,
  };
};
const parseRunsIndex = (payload) => {
  const record = ensureRecord(payload);
  const rawRuns = Array.isArray(record.runs) ? record.runs : [];
  const runs = [];
  for (const raw of rawRuns) {
    const item = ensureRecord(raw);
    const runId = normalizeRunId(item.run_id);
    if (!runId) continue;
    runs.push({
      run_id: runId,
      file_count: toPositiveInt(item.file_count, 0),
      total_bytes: toPositiveInt(item.total_bytes, 0),
      git_date: String(item.git_date || ""),
      source_run_dir: String(item.source_run_dir || ""),
    });
  }
  return {
    schema_version: toPositiveInt(record.schema_version, 0),
    generated_at_utc: String(record.generated_at_utc || ""),
    run_count: toPositiveInt(record.run_count, runs.length),
    runs,
  };
};
const sourceToRecord = (source) => {
  if (source.type === HF_SOURCE_TYPE) {
    return {
      type: HF_SOURCE_TYPE,
      repo: source.repo,
      runId: source.runId,
    };
  }
  return { type: DEFAULT_SOURCE_TYPE };
};
const sourceFromRecord = (value) => {
  const record = ensureRecord(value);
  const type = String(record.type || "")
    .trim()
    .toLowerCase();
  if (type === HF_SOURCE_TYPE) {
    const repo = normalizeRepo(record.repo);
    const runId = normalizeRunId(record.runId);
    if (runId) {
      return {
        type: HF_SOURCE_TYPE,
        repo,
        runId,
      };
    }
  }
  return { type: DEFAULT_SOURCE_TYPE };
};
const openDb = async () =>
  new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(RUN_STORE)) {
        db.createObjectStore(RUN_STORE, { keyPath: "cache_key" });
      }
      if (!db.objectStoreNames.contains(ASSET_STORE)) {
        const store = db.createObjectStore(ASSET_STORE, {
          keyPath: "asset_key",
        });
        store.createIndex("by_cache_key", "cache_key", { unique: false });
      }
    };
    request.onerror = () =>
      reject(request.error || new Error("Failed to open model cache DB."));
    request.onsuccess = () => resolve(request.result);
  });
const txComplete = async (tx) =>
  new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () =>
      reject(tx.error || new Error("IndexedDB transaction failed."));
    tx.onabort = () =>
      reject(tx.error || new Error("IndexedDB transaction aborted."));
  });
const requestToPromise = async (request) =>
  new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error || new Error("IndexedDB request failed."));
  });
const listAssetsForRun = async (db, cacheKey) => {
  const tx = db.transaction(ASSET_STORE, "readonly");
  const store = tx.objectStore(ASSET_STORE);
  const index = store.index("by_cache_key");
  const rows = await requestToPromise(index.getAll(cacheKey));
  await txComplete(tx);
  return Array.isArray(rows) ? rows : [];
};
const getRunRecord = async (db, repo, runId) => {
  const tx = db.transaction(RUN_STORE, "readonly");
  const store = tx.objectStore(RUN_STORE);
  const cacheKey = buildRunCacheKey(repo, runId);
  const row = await requestToPromise(store.get(cacheKey));
  await txComplete(tx);
  return row || null;
};
const sha256Hex = async (bytes) => {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const view = new Uint8Array(digest);
  let output = "";
  for (const byte of view) {
    output += byte.toString(16).padStart(2, "0");
  }
  return output;
};
const isTransformerCandidate = (summary) => summary.file_count >= 4;
const requiredFilesFromRunInfo = (runInfo) => {
  const byDestination = /* @__PURE__ */ new Map();
  for (const entry of runInfo.files) {
    byDestination.set(entry.destination, entry);
  }
  const missing = [];
  const required = [];
  for (const destination of REQUIRED_TRANSFORMER_DESTINATIONS) {
    const found = byDestination.get(destination);
    if (!found) {
      missing.push(destination);
      continue;
    }
    required.push(found);
  }
  if (missing.length > 0) {
    throw new Error(
      `Run ${runInfo.run_id} is missing required transformer files: ${missing.join(", ")}.`,
    );
  }
  return required;
};
const fileLooksCached = (cached, expected) => {
  if (!cached) return false;
  if (normalizeHash(cached.sha256) !== normalizeHash(expected.sha256))
    return false;
  if (
    toPositiveInt(cached.size_bytes, -1) !==
    toPositiveInt(expected.size_bytes, -2)
  ) {
    return false;
  }
  return true;
};
const readTextBlob = async (blob, label) => {
  try {
    return await blob.text();
  } catch (error) {
    throw new Error(
      `Unable to read cached ${label}: ${String(error && error.message ? error.message : error)}`,
    );
  }
};
const readJsonBlob = async (blob, label) => {
  const text = await readTextBlob(blob, label);
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(
      `Invalid cached JSON for ${label}: ${String(error && error.message ? error.message : error)}`,
    );
  }
};
const createBuiltinAssets = () => ({
  source: { type: DEFAULT_SOURCE_TYPE },
  sourceKey: DEFAULT_SOURCE_TYPE,
  modelLoadOptions: {
    cacheKey: DEFAULT_SOURCE_TYPE,
  },
  thresholdLoadOptions: {
    cacheKey: DEFAULT_SOURCE_TYPE,
  },
});
const normalizeEvalPayload = (payload) => {
  const metrics = ensureRecord(payload.metrics);
  const summary = ensureRecord(metrics.summary);
  const scam = ensureRecord(summary.scam);
  const macro = ensureRecord(summary.macro);
  const pick = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return void 0;
    return numeric;
  };
  return {
    scam_precision: pick(scam.precision),
    scam_recall: pick(scam.recall),
    scam_f1: pick(scam.f1),
    scam_fpr: pick(summary.scam_fpr),
    macro_f1: pick(macro.f1),
    exact_match: pick(summary.exact_match),
    scam_pr_auc: pick(summary.scam_pr_auc),
  };
};
const normalizeTransformerSource = (value) => sourceFromRecord(value);
const transformerSourceKey = (source) => {
  if (source.type === HF_SOURCE_TYPE) {
    return `${HF_SOURCE_TYPE}:${normalizeRepo(source.repo)}:${normalizeRunId(source.runId)}`;
  }
  return DEFAULT_SOURCE_TYPE;
};
const getActiveTransformerSource = async () => {
  const area = getStorageArea();
  if (!area) return { type: DEFAULT_SOURCE_TYPE };
  const payload = await storageGet(area, STORAGE_TRANSFORMER_SOURCE_KEY);
  return sourceFromRecord(payload?.[STORAGE_TRANSFORMER_SOURCE_KEY]);
};
const setActiveTransformerSource = async (source) => {
  const normalized = sourceFromRecord(source);
  const area = getStorageArea();
  if (!area) return normalized;
  await storageSet(area, {
    [STORAGE_TRANSFORMER_SOURCE_KEY]: sourceToRecord(normalized),
  });
  return normalized;
};
const fetchRemoteRuns = async (repo = DEFAULT_HF_EXPERIMENTS_REPO) => {
  const normalizedRepo = normalizeRepo(repo);
  const index = parseRunsIndex(
    await fetchJson(
      buildResolveUrl(normalizedRepo, "runs/INDEX.json"),
      `${normalizedRepo} runs index`,
    ),
  );
  const runs = index.runs
    .slice()
    .sort((left, right) => right.run_id.localeCompare(left.run_id))
    .map((run) => ({
      ...run,
      repo: normalizedRepo,
      is_transformer_candidate: isTransformerCandidate(run),
    }));
  return runs;
};
const fetchRemoteRunInfo = async (repo, runId) => {
  const normalizedRepo = normalizeRepo(repo);
  const normalizedRunId = normalizeRunId(runId);
  if (!normalizedRunId) {
    throw new Error("runId is required.");
  }
  const payload = await fetchJson(
    runInfoUrl(normalizedRepo, normalizedRunId),
    `${normalizedRepo}/${normalizedRunId} run info`,
  );
  const runInfo = parseRunInfo(payload);
  if (runInfo.run_id !== normalizedRunId) {
    throw new Error(
      `Run ID mismatch in RUN_INFO (${runInfo.run_id} !== ${normalizedRunId}).`,
    );
  }
  return runInfo;
};
const fetchRemoteRunEval = async (repo, runId) => {
  const normalizedRepo = normalizeRepo(repo);
  const normalizedRunId = normalizeRunId(runId);
  if (!normalizedRunId) return null;
  const url = buildResolveUrl(
    normalizedRepo,
    `runs/${normalizedRunId}/eval/eval.json`,
  );
  try {
    const payload = await fetchJson(
      url,
      `${normalizedRepo}/${normalizedRunId} eval`,
    );
    return normalizeEvalPayload(payload);
  } catch {
    return null;
  }
};
const listCachedTransformerRuns = async () => {
  const db = await openDb();
  try {
    const tx = db.transaction(RUN_STORE, "readonly");
    const rows = await requestToPromise(tx.objectStore(RUN_STORE).getAll());
    await txComplete(tx);
    const records = Array.isArray(rows) ? rows : [];
    return records
      .map((record) => ({
        repo: normalizeRepo(record.repo),
        runId: normalizeRunId(record.run_id),
        downloadedAtUtc: String(record.downloaded_at_utc || ""),
        totalBytes: toPositiveInt(record.total_bytes, 0),
        kind: String(record.run_info?.kind || ""),
        sourceRunDir: String(record.run_info?.source_run_dir || ""),
      }))
      .filter((record) => Boolean(record.runId))
      .sort((left, right) =>
        right.downloadedAtUtc.localeCompare(left.downloadedAtUtc),
      );
  } finally {
    db.close();
  }
};
const removeCachedTransformerRun = async (repo, runId) => {
  const normalizedRepo = normalizeRepo(repo);
  const normalizedRunId = normalizeRunId(runId);
  if (!normalizedRunId) return;
  const db = await openDb();
  try {
    const cacheKey = buildRunCacheKey(normalizedRepo, normalizedRunId);
    const assets = await listAssetsForRun(db, cacheKey);
    const tx = db.transaction([RUN_STORE, ASSET_STORE], "readwrite");
    const runStore = tx.objectStore(RUN_STORE);
    const assetStore = tx.objectStore(ASSET_STORE);
    runStore.delete(cacheKey);
    for (const asset of assets) {
      assetStore.delete(asset.asset_key);
    }
    await txComplete(tx);
    const active = await getActiveTransformerSource();
    if (
      active.type === HF_SOURCE_TYPE &&
      normalizeRepo(active.repo) === normalizedRepo &&
      normalizeRunId(active.runId) === normalizedRunId
    ) {
      await setActiveTransformerSource({ type: DEFAULT_SOURCE_TYPE });
    }
  } finally {
    db.close();
  }
};
const downloadAndActivateTransformerRun = async (repo, runId) => {
  const normalizedRepo = normalizeRepo(repo);
  const normalizedRunId = normalizeRunId(runId);
  if (!normalizedRunId) {
    throw new Error("runId is required.");
  }
  const runInfo = await fetchRemoteRunInfo(normalizedRepo, normalizedRunId);
  const requiredFiles = requiredFilesFromRunInfo(runInfo);
  const runCacheKey = buildRunCacheKey(normalizedRepo, normalizedRunId);
  const db = await openDb();
  try {
    const existingAssets = await listAssetsForRun(db, runCacheKey);
    const byDestination = /* @__PURE__ */ new Map();
    for (const entry of existingAssets) {
      byDestination.set(normalizeDestination(entry.destination), entry);
    }
    const now = /* @__PURE__ */ new Date().toISOString();
    const updates = [];
    let downloadedBytes = 0;
    let reusedCachedFiles = 0;
    for (const file of requiredFiles) {
      const cached = byDestination.get(file.destination);
      if (fileLooksCached(cached, file)) {
        reusedCachedFiles += 1;
        continue;
      }
      const url = runFileUrl(normalizedRepo, normalizedRunId, file.destination);
      const response = await fetch(url, {
        method: "GET",
        cache: "no-store",
        redirect: "follow",
      });
      if (!response.ok) {
        throw new Error(
          `Failed to download ${file.destination}: ${response.status} ${response.statusText}`,
        );
      }
      const blob = await response.blob();
      const buffer = await blob.arrayBuffer();
      const actualSize = buffer.byteLength;
      const actualSha = await sha256Hex(buffer);
      if (actualSize !== file.size_bytes) {
        throw new Error(
          `Size mismatch for ${file.destination}: expected ${file.size_bytes}, got ${actualSize}.`,
        );
      }
      if (actualSha !== normalizeHash(file.sha256)) {
        throw new Error(
          `Checksum mismatch for ${file.destination}: expected ${file.sha256}, got ${actualSha}.`,
        );
      }
      downloadedBytes += actualSize;
      updates.push({
        asset_key: buildAssetCacheKey(
          normalizedRepo,
          normalizedRunId,
          file.destination,
        ),
        cache_key: runCacheKey,
        repo: normalizedRepo,
        run_id: normalizedRunId,
        destination: file.destination,
        sha256: actualSha,
        size_bytes: actualSize,
        content_type: blob.type || "application/octet-stream",
        downloaded_at_utc: now,
        blob: blob.type
          ? blob
          : new Blob([buffer], { type: "application/octet-stream" }),
      });
    }
    const runRecord = {
      cache_key: runCacheKey,
      repo: normalizedRepo,
      run_id: normalizedRunId,
      run_info: runInfo,
      downloaded_at_utc: now,
      total_bytes: requiredFiles.reduce(
        (sum, file) => sum + file.size_bytes,
        0,
      ),
      required_destinations: REQUIRED_TRANSFORMER_DESTINATIONS.map((item) =>
        String(item),
      ),
    };
    const tx = db.transaction([RUN_STORE, ASSET_STORE], "readwrite");
    const runStore = tx.objectStore(RUN_STORE);
    const assetStore = tx.objectStore(ASSET_STORE);
    runStore.put(runRecord);
    for (const update of updates) {
      assetStore.put(update);
    }
    await txComplete(tx);
    await setActiveTransformerSource({
      type: HF_SOURCE_TYPE,
      repo: normalizedRepo,
      runId: normalizedRunId,
    });
    return {
      repo: normalizedRepo,
      runId: normalizedRunId,
      downloadedBytes,
      totalBytes: runRecord.total_bytes,
      reusedCachedFiles,
    };
  } finally {
    db.close();
  }
};
const resolveTransformerAssetsForSource = async (source) => {
  const normalizedSource = sourceFromRecord(source);
  const sourceKey = transformerSourceKey(normalizedSource);
  if (normalizedSource.type !== HF_SOURCE_TYPE) {
    return createBuiltinAssets();
  }
  const db = await openDb();
  try {
    const runRecord = await getRunRecord(
      db,
      normalizedSource.repo,
      normalizedSource.runId,
    );
    if (!runRecord) {
      throw new Error(
        `Transformer run ${normalizedSource.repo}/${normalizedSource.runId} is not cached locally.`,
      );
    }
    const assets = await listAssetsForRun(db, runRecord.cache_key);
    const byDestination = /* @__PURE__ */ new Map();
    for (const entry of assets) {
      byDestination.set(normalizeDestination(entry.destination), entry);
    }
    const missing = [];
    for (const destination of REQUIRED_TRANSFORMER_DESTINATIONS) {
      if (!byDestination.has(destination)) {
        missing.push(destination);
      }
    }
    if (missing.length > 0) {
      throw new Error(
        `Cached run ${normalizedSource.runId} is incomplete. Missing: ${missing.join(
          ", ",
        )}.`,
      );
    }
    const modelAsset = byDestination.get("model/model.int8.onnx");
    const configAsset = byDestination.get("model/student_config.json");
    const vocabAsset = byDestination.get("model/tokenizer/vocab.txt");
    const thresholdsAsset = byDestination.get("model/thresholds.json");
    const modelBytes = new Uint8Array(await modelAsset.blob.arrayBuffer());
    const studentConfig = await readJsonBlob(
      configAsset.blob,
      "student config",
    );
    const vocabText = await readTextBlob(vocabAsset.blob, "vocab");
    const thresholdsPayload = await readJsonBlob(
      thresholdsAsset.blob,
      "thresholds",
    );
    return {
      source: normalizedSource,
      sourceKey,
      modelLoadOptions: {
        modelData: modelBytes,
        studentConfig,
        vocabText,
        cacheKey: sourceKey,
      },
      thresholdLoadOptions: {
        thresholdsPayload,
        cacheKey: sourceKey,
      },
    };
  } finally {
    db.close();
  }
};
const resolveActiveTransformerAssets = async () => {
  const source = await getActiveTransformerSource();
  try {
    return await resolveTransformerAssetsForSource(source);
  } catch (error) {
    if (source.type === HF_SOURCE_TYPE) {
      if (typeof console !== "undefined" && console.warn) {
        console.warn(
          "Active Hugging Face transformer run could not be loaded, falling back to bundled transformer.",
          error,
        );
      }
      await setActiveTransformerSource({ type: DEFAULT_SOURCE_TYPE });
    }
    return createBuiltinAssets();
  }
};
const summarizeRemoteRun = async (repo, runId) => {
  const normalizedRepo = normalizeRepo(repo);
  const normalizedRunId = normalizeRunId(runId);
  if (!normalizedRunId) {
    throw new Error("runId is required.");
  }
  const [runInfo, evalSummary] = await Promise.all([
    fetchRemoteRunInfo(normalizedRepo, normalizedRunId),
    fetchRemoteRunEval(normalizedRepo, normalizedRunId),
  ]);
  return {
    runInfo,
    evalSummary,
  };
};
const isTransformerRunInfo = (runInfo) => {
  try {
    requiredFilesFromRunInfo(runInfo);
    return true;
  } catch {
    return false;
  }
};
const bytesToMiB = (bytes) =>
  Number.isFinite(bytes) ? bytes / (1024 * 1024) : 0;
const formatBytes = (bytes) => {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  if (value < 1024) return `${value.toFixed(0)} B`;
  const kib = value / 1024;
  if (kib < 1024) return `${kib.toFixed(1)} KiB`;
  const mib = kib / 1024;
  if (mib < 1024) return `${mib.toFixed(2)} MiB`;
  const gib = mib / 1024;
  return `${gib.toFixed(2)} GiB`;
};
const clampScore = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return void 0;
  return Math.min(1, Math.max(0, numeric));
};
const toPercent = (value) => {
  const score = clampScore(value);
  if (score === void 0) return "n/a";
  return `${(score * 100).toFixed(2)}%`;
};
const pickEvalMetrics = (evalSummary) => {
  if (!evalSummary) return null;
  const output = {};
  const assign = (key, value) => {
    if (typeof value === "number" && Number.isFinite(value)) {
      output[key] = value;
    }
  };
  assign("scam_precision", evalSummary.scam_precision);
  assign("scam_recall", evalSummary.scam_recall);
  assign("scam_f1", evalSummary.scam_f1);
  assign("scam_fpr", evalSummary.scam_fpr);
  assign("macro_f1", evalSummary.macro_f1);
  assign("exact_match", evalSummary.exact_match);
  assign("scam_pr_auc", evalSummary.scam_pr_auc);
  return Object.keys(output).length > 0 ? output : null;
};
export {
  DEFAULT_HF_EXPERIMENTS_REPO,
  STORAGE_TRANSFORMER_SOURCE_KEY,
  bytesToMiB,
  clampScore,
  downloadAndActivateTransformerRun,
  fetchRemoteRunEval,
  fetchRemoteRunInfo,
  fetchRemoteRuns,
  formatBytes,
  getActiveTransformerSource,
  isTransformerRunInfo,
  listCachedTransformerRuns,
  normalizeTransformerSource,
  pickEvalMetrics,
  removeCachedTransformerRun,
  resolveActiveTransformerAssets,
  resolveTransformerAssetsForSource,
  setActiveTransformerSource,
  summarizeRemoteRun,
  toPercent,
  transformerSourceKey,
};
