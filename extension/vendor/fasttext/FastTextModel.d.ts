import type { FastTextCore, FastTextModule } from './core/fastText';
/**
 * `FastTextModel` represents a trained model.
 */
export declare class FastTextModel {
    core: FastTextModule;
    ft: FastTextCore;
    constructor(
    /** webassembly object that makes the bridge between js and C++ */
    fastTextNative: FastTextCore, fastTextModule: FastTextModule);
    getFloat32ArrayFromHeap: (len: number) => {
        ptr: number;
        size: number;
        buffer: ArrayBufferLike;
    };
    heapToFloat32: (r: {
        ptr: any;
        size: any;
        buffer: any;
    }) => Float32Array;
    /**
     * isQuant
     *
     * @return {bool}   true if the model is quantized
     */
    isQuant(): boolean;
    /**
     * getDimension
     *
     * @return {number}    the dimension (size) of a lookup vector (hidden layer)
     */
    getDimension(): number;
    /**
     * getWordVector
     *
     * @param {string}          word
     *
     * @return {Float32Array}   the vector representation of `word`.
     *
     */
    getWordVector(word: string): Float32Array;
    /**
     * getSentenceVector
     *
     * @return {Float32Array}   the vector representation of `text`.
     *
     */
    getSentenceVector(text: string): Float32Array;
    /**
     * getNearestNeighbors
     *
     * returns the nearest `k` neighbors of `word`.
     *
     * @return words and their corresponding cosine similarities.
     *
     */
    getNearestNeighbors(word: string, k?: number): import("./types/misc").Vector<import("./types/misc").Pair<number, string>>;
    /**
     * getAnalogies
     *
     * returns the nearest `k` neighbors of the operation
     * `wordA - wordB + wordC`.
     *
     * @return words and their corresponding cosine similarities
     *
     */
    getAnalogies(wordA: string, wordB: string, wordC: string, k: number): import("./types/misc").Vector<import("./types/misc").Pair<number, string>>;
    /**
     * getWordId
     *
     * Given a word, get the word id within the dictionary.
     * Returns -1 if word is not in the dictionary.
     */
    getWordId(word: string): number;
    /**
     * getSubwordId
     *
     * Given a subword, return the index (within input matrix) it hashes to.
     */
    getSubwordId(subword: string): number;
    /**
     * getSubwords
     *
     * returns the subwords and their indicies.
     *
     * @return words and their corresponding indicies
     *
     */
    getSubwords(word: string): import("./types/misc").Pair<string[], number[]>;
    /**
     * getInputVector
     *
     * Given an index, get the corresponding vector of the Input Matrix.
     *
     * @return {Float32Array}   the vector of the `ind`'th index
     *
     */
    getInputVector(ind: number): Float32Array;
    /**
     * predict
     *
     * Given a string, get a list of labels and a list of corresponding
     * probabilities. k controls the number of returned labels.
     *
     * @return labels and their probabilities
     *
     */
    predict(text: string, 
    /** max number of return entries, use -1 to return all */
    k?: number, 
    /** min possibility of return entries(0~1) */
    threshold?: number): import("./types/misc").Vector<import("./types/misc").Pair<number, string>>;
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
    getInputMatrix(): unknown;
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
    getOutputMatrix(): unknown;
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
    getWords(): import("./types/misc").Pair<import("./types/misc").Vector<string>, import("./types/misc").Vector<number>>;
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
    getLabels(): import("./types/misc").Pair<import("./types/misc").Vector<string>, import("./types/misc").Vector<number>>;
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
    getLine(text: string): import("./types/misc").Pair<import("./types/misc").Vector<string>, import("./types/misc").Vector<string>>;
    /**
     * saveModel
     *
     * Saves the model file in web assembly in-memory FS and returns a blob
     *
     * @return {Blob}           blob data of the file saved in web assembly FS
     *
     */
    saveModel(): Blob;
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
    test(url: RequestInfo | URL, 
    /** the number of predictions to be returned */
    k: number, 
    /** threshold */
    threshold: number): Promise<unknown>;
}
