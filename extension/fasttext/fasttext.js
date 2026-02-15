// Generated from extension/src/*.ts by `npm run extension:build`.
import fastTextModularized from "./fasttext_wasm.js";
const wasmUrl = new URL("./fasttext_wasm.wasm", import.meta.url).toString();
const fastTextModulePromise = fastTextModularized({
  locateFile: (path) => {
    if (path.endsWith(".wasm")) {
      return wasmUrl;
    }
    return new URL(path, import.meta.url).toString();
  },
});
let fastTextModule = null;
let postRunFunc = null;
let moduleReady = false;
const fastTextReady = fastTextModulePromise.then((module) => {
  fastTextModule = module;
  moduleReady = true;
  if (postRunFunc) {
    postRunFunc();
  }
  return module;
});
const addOnPostRun = (func) => {
  postRunFunc = func;
  if (moduleReady && postRunFunc) {
    postRunFunc();
  }
};
const thisModule = typeof globalThis !== "undefined" ? globalThis : void 0;
const trainFileInWasmFs = "train.txt";
const testFileInWasmFs = "test.txt";
const modelFileInWasmFs = "model.bin";
const getFloat32ArrayFromHeap = (len) => {
  if (!fastTextModule) {
    throw new Error("fastText WASM module is not ready.");
  }
  const dataBytes = len * Float32Array.BYTES_PER_ELEMENT;
  const dataPtr = fastTextModule._malloc(dataBytes);
  const dataHeap = new Uint8Array(
    fastTextModule.HEAPU8.buffer,
    dataPtr,
    dataBytes,
  );
  return {
    ptr: dataHeap.byteOffset,
    size: len,
    buffer: dataHeap.buffer,
  };
};
const heapToFloat32 = (heapBuffer) =>
  new Float32Array(heapBuffer.buffer, heapBuffer.ptr, heapBuffer.size);
class FastText {
  f;
  constructor() {
    if (!fastTextModule) {
      throw new Error(
        "fastText WASM module not initialized. Await fastTextReady first.",
      );
    }
    this.f = new fastTextModule.FastText();
  }
  /**
   * Loads the model file from the specified URL and returns `FastTextModel`.
   */
  loadModel(url) {
    const fetchFunc = (thisModule && thisModule.fetch) || fetch;
    const fastTextNative = this.f;
    const module = fastTextModule;
    if (!module) {
      return Promise.reject(new Error("fastText WASM module is not ready."));
    }
    return new Promise((resolve, reject) => {
      fetchFunc(url)
        .then((response) => response.arrayBuffer())
        .then((bytes) => {
          const byteArray = new Uint8Array(bytes);
          module.FS.writeFile(modelFileInWasmFs, byteArray);
        })
        .then(() => {
          fastTextNative.loadModel(modelFileInWasmFs);
          resolve(new FastTextModel(fastTextNative));
        })
        .catch((error) => {
          reject(error);
        });
    });
  }
  _train(url, modelName, kwargs = {}, callback = null) {
    const fetchFunc = (thisModule && thisModule.fetch) || fetch;
    const fastTextNative = this.f;
    const module = fastTextModule;
    if (!module) {
      return Promise.reject(new Error("fastText WASM module is not ready."));
    }
    return new Promise((resolve, reject) => {
      fetchFunc(url)
        .then((response) => response.arrayBuffer())
        .then((bytes) => {
          const byteArray = new Uint8Array(bytes);
          module.FS.writeFile(trainFileInWasmFs, byteArray);
        })
        .then(() => {
          const argsList = [
            "lr",
            "lrUpdateRate",
            "dim",
            "ws",
            "epoch",
            "minCount",
            "minCountLabel",
            "neg",
            "wordNgrams",
            "loss",
            "model",
            "bucket",
            "minn",
            "maxn",
            "t",
            "label",
            "verbose",
            "pretrainedVectors",
            "saveOutput",
            "seed",
            "qout",
            "retrain",
            "qnorm",
            "cutoff",
            "dsub",
            "qnorm",
            "autotuneValidationFile",
            "autotuneMetric",
            "autotunePredictions",
            "autotuneDuration",
            "autotuneModelSize",
          ];
          const args = new module.Args();
          argsList.forEach((key) => {
            if (key in kwargs) {
              args[key] = kwargs[key];
            }
          });
          args.model = module.ModelName[modelName];
          args.loss =
            "loss" in kwargs ? module.LossName[String(kwargs.loss)] : "hs";
          args.thread = 1;
          args.input = trainFileInWasmFs;
          fastTextNative.train(args, callback);
          resolve(new FastTextModel(fastTextNative));
        })
        .catch((error) => {
          reject(error);
        });
    });
  }
  trainSupervised(url, kwargs = {}, callback) {
    return this._train(url, "supervised", kwargs, callback || null);
  }
  trainUnsupervised(url, modelName, kwargs = {}, callback) {
    return this._train(url, modelName, kwargs, callback || null);
  }
}
class FastTextModel {
  f;
  constructor(fastTextNative) {
    this.f = fastTextNative;
  }
  isQuant() {
    return this.f.isQuant;
  }
  getDimension() {
    return this.f.args.dim;
  }
  getWordVector(word) {
    const heapBuffer = getFloat32ArrayFromHeap(this.getDimension());
    this.f.getWordVector(heapBuffer, word);
    return heapToFloat32(heapBuffer);
  }
  getSentenceVector(text) {
    if (text.includes("\n") && typeof console !== "undefined" && console.warn) {
      console.warn(
        "Sentence vector expects a single line; replacing newlines.",
      );
    }
    const normalized = text.replace(/\n/g, " ") + "\n";
    const heapBuffer = getFloat32ArrayFromHeap(this.getDimension());
    this.f.getSentenceVector(heapBuffer, normalized);
    return heapToFloat32(heapBuffer);
  }
  getNearestNeighbors(word, k = 10) {
    return this.f.getNN(word, k);
  }
  getAnalogies(wordA, wordB, wordC, k) {
    return this.f.getAnalogies(k, wordA, wordB, wordC);
  }
  getWordId(word) {
    return this.f.getWordId(word);
  }
  getSubwordId(subword) {
    return this.f.getSubwordId(subword);
  }
  getSubwords(word) {
    return this.f.getSubwords(word);
  }
  getInputVector(index) {
    const heapBuffer = getFloat32ArrayFromHeap(this.getDimension());
    this.f.getInputVector(heapBuffer, index);
    return heapToFloat32(heapBuffer);
  }
  predict(text, k = 1, threshold = 0) {
    return this.f.predict(text, k, threshold);
  }
  getInputMatrix() {
    if (this.isQuant()) {
      throw new Error("Cannot read input matrix from a quantized model.");
    }
    return this.f.getInputMatrix();
  }
  getOutputMatrix() {
    if (this.isQuant()) {
      throw new Error("Cannot read output matrix from a quantized model.");
    }
    return this.f.getOutputMatrix();
  }
  getWords() {
    return this.f.getWords();
  }
  getLabels() {
    return this.f.getLabels();
  }
  getLine(text) {
    return this.f.getLine(text);
  }
  saveModel() {
    if (!fastTextModule) {
      throw new Error("fastText WASM module is not ready.");
    }
    this.f.saveModel(modelFileInWasmFs);
    const content = fastTextModule.FS.readFile(modelFileInWasmFs, {
      encoding: "binary",
    });
    const bytes = Uint8Array.from(content);
    return new Blob([bytes], { type: "application/octet-stream" });
  }
  test(url, k, threshold) {
    const fetchFunc = (thisModule && thisModule.fetch) || fetch;
    const fastTextNative = this.f;
    const module = fastTextModule;
    if (!module) {
      return Promise.reject(new Error("fastText WASM module is not ready."));
    }
    return new Promise((resolve, reject) => {
      fetchFunc(url)
        .then((response) => response.arrayBuffer())
        .then((bytes) => {
          const byteArray = new Uint8Array(bytes);
          module.FS.writeFile(testFileInWasmFs, byteArray);
        })
        .then(() => {
          const meter = fastTextNative.test(testFileInWasmFs, k, threshold);
          resolve(meter);
        })
        .catch((error) => {
          reject(error);
        });
    });
  }
}
export { FastText, FastTextModel, addOnPostRun, fastTextReady };
