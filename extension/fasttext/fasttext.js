/**
 * fastText WebAssembly wrapper for browser/extension usage.
 * Based on the upstream fastText webassembly API with a minimal async init.
 */
import fastTextModularized from './fasttext_wasm.js';

const wasmUrl = new URL('./fasttext_wasm.wasm', import.meta.url).toString();

const fastTextModulePromise = fastTextModularized({
  locateFile: (path) => {
    if (path.endsWith('.wasm')) {
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

const addOnPostRun = function (func) {
  postRunFunc = func;
  if (moduleReady && postRunFunc) {
    postRunFunc();
  }
};

const thisModule = typeof globalThis !== 'undefined' ? globalThis : undefined;
const trainFileInWasmFs = 'train.txt';
const testFileInWasmFs = 'test.txt';
const modelFileInWasmFs = 'model.bin';

const getFloat32ArrayFromHeap = (len) => {
  const dataBytes = len * Float32Array.BYTES_PER_ELEMENT;
  const dataPtr = fastTextModule._malloc(dataBytes);
  const dataHeap = new Uint8Array(fastTextModule.HEAPU8.buffer, dataPtr, dataBytes);
  return {
    ptr: dataHeap.byteOffset,
    size: len,
    buffer: dataHeap.buffer,
  };
};

const heapToFloat32 = (r) => new Float32Array(r.buffer, r.ptr, r.size);

class FastText {
  constructor() {
    if (!fastTextModule) {
      throw new Error('fastText WASM module not initialized. Await fastTextReady first.');
    }
    this.f = new fastTextModule.FastText();
  }

  /**
   * Loads the model file from the specified url, and returns the
   * corresponding `FastTextModel` object.
   */
  loadModel(url) {
    const fetchFunc = (thisModule && thisModule.fetch) || fetch;
    const fastTextNative = this.f;

    return new Promise(function (resolve, reject) {
      fetchFunc(url)
        .then((response) => response.arrayBuffer())
        .then((bytes) => {
          const byteArray = new Uint8Array(bytes);
          const FS = fastTextModule.FS;
          FS.writeFile(modelFileInWasmFs, byteArray);
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

    return new Promise(function (resolve, reject) {
      fetchFunc(url)
        .then((response) => response.arrayBuffer())
        .then((bytes) => {
          const byteArray = new Uint8Array(bytes);
          const FS = fastTextModule.FS;
          FS.writeFile(trainFileInWasmFs, byteArray);
        })
        .then(() => {
          const argsList = [
            'lr',
            'lrUpdateRate',
            'dim',
            'ws',
            'epoch',
            'minCount',
            'minCountLabel',
            'neg',
            'wordNgrams',
            'loss',
            'model',
            'bucket',
            'minn',
            'maxn',
            't',
            'label',
            'verbose',
            'pretrainedVectors',
            'saveOutput',
            'seed',
            'qout',
            'retrain',
            'qnorm',
            'cutoff',
            'dsub',
            'qnorm',
            'autotuneValidationFile',
            'autotuneMetric',
            'autotunePredictions',
            'autotuneDuration',
            'autotuneModelSize',
          ];
          const args = new fastTextModule.Args();
          argsList.forEach((k) => {
            if (k in kwargs) {
              args[k] = kwargs[k];
            }
          });
          args.model = fastTextModule.ModelName[modelName];
          args.loss = 'loss' in kwargs ? fastTextModule.LossName[kwargs.loss] : 'hs';
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
    const self = this;
    return new Promise(function (resolve, reject) {
      self
        ._train(url, 'supervised', kwargs, callback)
        .then((model) => {
          resolve(model);
        })
        .catch((error) => {
          reject(error);
        });
    });
  }

  trainUnsupervised(url, modelName, kwargs = {}, callback) {
    const self = this;
    return new Promise(function (resolve, reject) {
      self
        ._train(url, modelName, kwargs, callback)
        .then((model) => {
          resolve(model);
        })
        .catch((error) => {
          reject(error);
        });
    });
  }
}

class FastTextModel {
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
    const b = getFloat32ArrayFromHeap(this.getDimension());
    this.f.getWordVector(b, word);
    return heapToFloat32(b);
  }

  getSentenceVector(text) {
    if (text.indexOf('\n') !== -1) {
      'sentence vector processes one line at a time (remove \\n)';
    }
    text += '\n';
    const b = getFloat32ArrayFromHeap(this.getDimension());
    this.f.getSentenceVector(b, text);
    return heapToFloat32(b);
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

  getInputVector(ind) {
    const b = getFloat32ArrayFromHeap(this.getDimension());
    this.f.getInputVector(b, ind);
    return heapToFloat32(b);
  }

  predict(text, k = 1, threshold = 0.0) {
    return this.f.predict(text, k, threshold);
  }

  getInputMatrix() {
    if (this.isQuant()) {
      throw new Error("Can't get quantized Matrix");
    }
    return this.f.getInputMatrix();
  }

  getOutputMatrix() {
    if (this.isQuant()) {
      throw new Error("Can't get quantized Matrix");
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
    this.f.saveModel(modelFileInWasmFs);
    const content = fastTextModule.FS.readFile(modelFileInWasmFs, { encoding: 'binary' });
    return new Blob([new Uint8Array(content, content.byteOffset, content.length)], {
      type: 'application/octet-stream',
    });
  }

  test(url, k, threshold) {
    const fetchFunc = (thisModule && thisModule.fetch) || fetch;
    const fastTextNative = this.f;

    return new Promise(function (resolve, reject) {
      fetchFunc(url)
        .then((response) => response.arrayBuffer())
        .then((bytes) => {
          const byteArray = new Uint8Array(bytes);
          const FS = fastTextModule.FS;
          FS.writeFile(testFileInWasmFs, byteArray);
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
