/**
 * fastText WebAssembly wrapper for browser/extension usage.
 * Based on the upstream fastText webassembly API with a minimal async init.
 */
import fastTextModularized from "./fasttext_wasm.js";

const wasmUrl = new URL("./fasttext_wasm.wasm", import.meta.url).toString();

const fastTextModulePromise = fastTextModularized({
  locateFile: (path: string) => {
    if (path.endsWith(".wasm")) {
      return wasmUrl;
    }
    return new URL(path, import.meta.url).toString();
  },
});

type HeapFloat32Buffer = {
  ptr: number;
  size: number;
  buffer: ArrayBufferLike;
};

let fastTextModule: any | null = null;
let postRunFunc: (() => void) | null = null;
let moduleReady = false;

const fastTextReady = fastTextModulePromise.then((module) => {
  fastTextModule = module;
  moduleReady = true;
  if (postRunFunc) {
    postRunFunc();
  }
  return module;
});

const addOnPostRun = (func: () => void): void => {
  postRunFunc = func;
  if (moduleReady && postRunFunc) {
    postRunFunc();
  }
};

const thisModule = typeof globalThis !== "undefined" ? globalThis : undefined;
const trainFileInWasmFs = "train.txt";
const testFileInWasmFs = "test.txt";
const modelFileInWasmFs = "model.bin";

const getFloat32ArrayFromHeap = (len: number): HeapFloat32Buffer => {
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

const heapToFloat32 = (heapBuffer: HeapFloat32Buffer): Float32Array =>
  new Float32Array(heapBuffer.buffer, heapBuffer.ptr, heapBuffer.size);

class FastText {
  private f: any;

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
  loadModel(url: string): Promise<FastTextModel> {
    const fetchFunc = (thisModule && thisModule.fetch) || fetch;
    const fastTextNative = this.f;
    const module = fastTextModule;
    if (!module) {
      return Promise.reject(new Error("fastText WASM module is not ready."));
    }

    return new Promise<FastTextModel>((resolve, reject) => {
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

  private _train(
    url: string,
    modelName: string,
    kwargs: Record<string, unknown> = {},
    callback: ((...args: unknown[]) => void) | null = null,
  ): Promise<FastTextModel> {
    const fetchFunc = (thisModule && thisModule.fetch) || fetch;
    const fastTextNative = this.f;
    const module = fastTextModule;
    if (!module) {
      return Promise.reject(new Error("fastText WASM module is not ready."));
    }

    return new Promise<FastTextModel>((resolve, reject) => {
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

  trainSupervised(
    url: string,
    kwargs: Record<string, unknown> = {},
    callback?: (...args: unknown[]) => void,
  ): Promise<FastTextModel> {
    return this._train(url, "supervised", kwargs, callback || null);
  }

  trainUnsupervised(
    url: string,
    modelName: string,
    kwargs: Record<string, unknown> = {},
    callback?: (...args: unknown[]) => void,
  ): Promise<FastTextModel> {
    return this._train(url, modelName, kwargs, callback || null);
  }
}

class FastTextModel {
  private f: any;

  constructor(fastTextNative: any) {
    this.f = fastTextNative;
  }

  isQuant(): boolean {
    return this.f.isQuant;
  }

  getDimension(): number {
    return this.f.args.dim;
  }

  getWordVector(word: string): Float32Array {
    const heapBuffer = getFloat32ArrayFromHeap(this.getDimension());
    this.f.getWordVector(heapBuffer, word);
    return heapToFloat32(heapBuffer);
  }

  getSentenceVector(text: string): Float32Array {
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

  getNearestNeighbors(word: string, k = 10): unknown {
    return this.f.getNN(word, k);
  }

  getAnalogies(
    wordA: string,
    wordB: string,
    wordC: string,
    k: number,
  ): unknown {
    return this.f.getAnalogies(k, wordA, wordB, wordC);
  }

  getWordId(word: string): number {
    return this.f.getWordId(word);
  }

  getSubwordId(subword: string): number {
    return this.f.getSubwordId(subword);
  }

  getSubwords(word: string): unknown {
    return this.f.getSubwords(word);
  }

  getInputVector(index: number): Float32Array {
    const heapBuffer = getFloat32ArrayFromHeap(this.getDimension());
    this.f.getInputVector(heapBuffer, index);
    return heapToFloat32(heapBuffer);
  }

  predict(text: string, k = 1, threshold = 0.0): unknown {
    return this.f.predict(text, k, threshold);
  }

  getInputMatrix(): unknown {
    if (this.isQuant()) {
      throw new Error("Cannot read input matrix from a quantized model.");
    }
    return this.f.getInputMatrix();
  }

  getOutputMatrix(): unknown {
    if (this.isQuant()) {
      throw new Error("Cannot read output matrix from a quantized model.");
    }
    return this.f.getOutputMatrix();
  }

  getWords(): unknown {
    return this.f.getWords();
  }

  getLabels(): unknown {
    return this.f.getLabels();
  }

  getLine(text: string): unknown {
    return this.f.getLine(text);
  }

  saveModel(): Blob {
    if (!fastTextModule) {
      throw new Error("fastText WASM module is not ready.");
    }
    this.f.saveModel(modelFileInWasmFs);
    const content = fastTextModule.FS.readFile(modelFileInWasmFs, {
      encoding: "binary",
    }) as Uint8Array;
    const bytes = Uint8Array.from(content);
    return new Blob([bytes], { type: "application/octet-stream" });
  }

  test(url: string, k: number, threshold: number): Promise<unknown> {
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
