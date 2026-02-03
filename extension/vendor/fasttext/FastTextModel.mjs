import { modelFileInWasmFs, testFileInWasmFs } from './constants.mjs';

class FastTextModel {
  core;
  ft;
  constructor(fastTextNative, fastTextModule) {
    this.ft = fastTextNative;
    this.core = fastTextModule;
  }
  getFloat32ArrayFromHeap = (len) => {
    const dataBytes = len * Float32Array.BYTES_PER_ELEMENT;
    const dataPtr = this.core._malloc(dataBytes);
    const dataHeap = new Uint8Array(this.core.HEAPU8.buffer, dataPtr, dataBytes);
    return {
      ptr: dataHeap.byteOffset,
      size: len,
      buffer: dataHeap.buffer
    };
  };
  heapToFloat32 = (r) => new Float32Array(r.buffer, r.ptr, r.size);
  /**
   * isQuant
   *
   * @return {bool}   true if the model is quantized
   */
  isQuant() {
    return this.ft.isQuant;
  }
  /**
   * getDimension
   *
   * @return {number}    the dimension (size) of a lookup vector (hidden layer)
   */
  getDimension() {
    return this.ft.args.dim;
  }
  /**
   * getWordVector
   *
   * @param {string}          word
   *
   * @return {Float32Array}   the vector representation of `word`.
   *
   */
  getWordVector(word) {
    const b = this.getFloat32ArrayFromHeap(this.getDimension());
    this.ft.getWordVector(b, word);
    return this.heapToFloat32(b);
  }
  /**
   * getSentenceVector
   *
   * @return {Float32Array}   the vector representation of `text`.
   *
   */
  getSentenceVector(text) {
    if (text.includes("\n")) ;
    text += "\n";
    const b = this.getFloat32ArrayFromHeap(this.getDimension());
    this.ft.getSentenceVector(b, text);
    return this.heapToFloat32(b);
  }
  /**
   * getNearestNeighbors
   *
   * returns the nearest `k` neighbors of `word`.
   *
   * @return words and their corresponding cosine similarities.
   *
   */
  getNearestNeighbors(word, k = 10) {
    return this.ft.getNN(word, k);
  }
  /**
   * getAnalogies
   *
   * returns the nearest `k` neighbors of the operation
   * `wordA - wordB + wordC`.
   *
   * @return words and their corresponding cosine similarities
   *
   */
  getAnalogies(wordA, wordB, wordC, k) {
    return this.ft.getAnalogies(k, wordA, wordB, wordC);
  }
  /**
   * getWordId
   *
   * Given a word, get the word id within the dictionary.
   * Returns -1 if word is not in the dictionary.
   */
  getWordId(word) {
    return this.ft.getWordId(word);
  }
  /**
   * getSubwordId
   *
   * Given a subword, return the index (within input matrix) it hashes to.
   */
  getSubwordId(subword) {
    return this.ft.getSubwordId(subword);
  }
  /**
   * getSubwords
   *
   * returns the subwords and their indicies.
   *
   * @return words and their corresponding indicies
   *
   */
  getSubwords(word) {
    return this.ft.getSubwords(word);
  }
  /**
   * getInputVector
   *
   * Given an index, get the corresponding vector of the Input Matrix.
   *
   * @return {Float32Array}   the vector of the `ind`'th index
   *
   */
  getInputVector(ind) {
    const b = this.getFloat32ArrayFromHeap(this.getDimension());
    this.ft.getInputVector(b, ind);
    return this.heapToFloat32(b);
  }
  /**
   * predict
   *
   * Given a string, get a list of labels and a list of corresponding
   * probabilities. k controls the number of returned labels.
   *
   * @return labels and their probabilities
   *
   */
  predict(text, k = 1, threshold = 0) {
    return this.ft.predict(text, k, threshold);
  }
  /**
   * getInputMatrix
   *
   * Get a reference to the full input matrix of a Model. This only
   * works if the model is not quantized.
   *
   * @return {DenseMatrix}
   *     densematrix with functions: `rows`, `cols`, `at(i,j)`
   *
   * example:
   *     let inputMatrix = model.getInputMatrix();
   *     let value = inputMatrix.at(1, 2);
   */
  getInputMatrix() {
    if (this.isQuant()) {
      throw new Error("Can't get quantized Matrix");
    }
    return this.ft.getInputMatrix();
  }
  /**
   * getOutputMatrix
   *
   * Get a reference to the full input matrix of a Model. This only
   * works if the model is not quantized.
   *
   * @return {DenseMatrix}
   *     densematrix with functions: `rows`, `cols`, `at(i,j)`
   *
   * example:
   *     let outputMatrix = model.getOutputMatrix();
   *     let value = outputMatrix.at(1, 2);
   */
  getOutputMatrix() {
    if (this.isQuant()) {
      throw new Error("Can't get quantized Matrix");
    }
    return this.ft.getOutputMatrix();
  }
  /**
   * getWords
   *
   * Get the entire list of words of the dictionary including the frequency
   * of the individual words. This does not include any subwords. For that
   * please consult the function get_subwords.
   *
   * @return {Pair.<Array.<string>, Array.<int>>}
   *     words and their corresponding frequencies
   *
   */
  getWords() {
    return this.ft.getWords();
  }
  /**
   * getLabels
   *
   * Get the entire list of labels of the dictionary including the frequency
   * of the individual labels.
   *
   * @return {Pair.<Array.<string>, Array.<int>>}
   *     labels and their corresponding frequencies
   *
   */
  getLabels() {
    return this.ft.getLabels();
  }
  /**
   * getLine
   *
   * Split a line of text into words and labels. Labels must start with
   * the prefix used to create the model (__label__ by default).
   *
   * @return {Pair.<Array.<string>, Array.<string>>}
   *     words and labels
   *
   */
  getLine(text) {
    return this.ft.getLine(text);
  }
  /**
   * saveModel
   *
   * Saves the model file in web assembly in-memory FS and returns a blob
   *
   * @return {Blob}           blob data of the file saved in web assembly FS
   *
   */
  saveModel() {
    this.ft.saveModel(modelFileInWasmFs);
    const content = this.core.FS.readFile(modelFileInWasmFs, {
      encoding: "binary"
    });
    return new Blob(
      [new Uint8Array(content, content.byteOffset, content.length)],
      { type: " application/octet-stream" }
    );
  }
  /**
   * test
   *
   * Downloads the test file from the specified url, evaluates the supervised
   * model with it.
   *
   * @return {Promise}   promise object that resolves to a `Meter` object
   *
   * example:
   * model.test("/absolute/url/to/test.txt", 1, 0.0).then((meter) => {
   *     console.log(meter.precision);
   *     console.log(meter.recall);
   *     console.log(meter.f1Score);
   *     console.log(meter.nexamples());
   * });
   *
   */
  test(url, k, threshold) {
    const fetchFunc = globalThis.fetch || fetch;
    const fastTextNative = this.ft;
    return new Promise((resolve, reject) => {
      fetchFunc(url).then((response) => {
        return response.arrayBuffer();
      }).then((bytes) => {
        const byteArray = new Uint8Array(bytes);
        const FS = this.core.FS;
        FS.writeFile(testFileInWasmFs, byteArray);
      }).then(() => {
        const meter = fastTextNative.test(testFileInWasmFs, k, threshold);
        resolve(meter);
      }).catch((error) => {
        reject(error);
      });
    });
  }
}

export { FastTextModel };
