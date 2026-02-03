'use strict';

Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });

const index = require('./helpers/modules/index.cjs');
const FastTextModel = require('./FastTextModel.cjs');
const constants = require('./constants.cjs');

const getFastTextClass = async (options) => {
  const { getFastTextModule } = options;
  const fastTextModule = await getFastTextModule();
  return class FastText {
    core;
    ft;
    fs;
    constructor() {
      this.core = fastTextModule;
      this.ft = new fastTextModule.FastText();
      this.fs = fastTextModule.FS;
    }
    /**
     * loadModel
     *
     * Loads the model file from the specified url, and returns the
     * corresponding `FastTextModel` object.
     */
    async loadModel(url) {
      const fastTextNative = this.ft;
      const arrayBuffer = await index.fetchFile(url);
      const byteArray = new Uint8Array(arrayBuffer);
      const FS = fastTextModule.FS;
      await FS.writeFile(constants.modelFileInWasmFs, byteArray);
      fastTextNative.loadModel(constants.modelFileInWasmFs);
      return new FastTextModel.FastTextModel(fastTextNative, this.core);
    }
    _train(url, modelName, kwargs = {}, callback = null) {
      const fetchFunc = globalThis.fetch || fetch;
      const fastTextNative = this.ft;
      return new Promise((resolve, reject) => {
        fetchFunc(url).then((response) => {
          return response.arrayBuffer();
        }).then((bytes) => {
          const byteArray = new Uint8Array(bytes);
          const FS = fastTextModule.FS;
          FS.writeFile(constants.trainFileInWasmFs, byteArray);
        }).then(() => {
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
            "autotuneModelSize"
          ];
          const args = new fastTextModule.Args();
          argsList.forEach((k) => {
            if (k in kwargs) {
              args[k] = kwargs[k];
            }
          });
          args.model = fastTextModule.ModelName[modelName];
          args.loss = "loss" in kwargs ? fastTextModule.LossName[kwargs.loss] : "hs";
          args.thread = 1;
          args.input = constants.trainFileInWasmFs;
          fastTextNative.train(args, callback);
          resolve(new FastTextModel.FastTextModel(fastTextNative, this.core));
        }).catch((error) => {
          reject(error);
        });
      });
    }
    /**
     * trainSupervised
     *
     * Downloads the input file from the specified url, trains a supervised
     * model and returns a `FastTextModel` object.
     */
    trainSupervised(url, kwargs = {}, callback) {
      const self = this;
      return new Promise((resolve, reject) => {
        self._train(url, "supervised", kwargs, callback).then((model) => {
          resolve(model);
        }).catch((error) => {
          reject(error);
        });
      });
    }
    /**
     * trainUnsupervised
     *
     * Downloads the input file from the specified url, trains an unsupervised
     * model and returns a `FastTextModel` object.
     *
     * @param {function}   callback
     *     train callback function
     *     `callback` function is called regularly from the train loop:
     *     `callback(progress, loss, wordsPerSec, learningRate, eta)`
     *
     * @return {Promise}   promise object that resolves to a `FastTextModel`
     *
     */
    trainUnsupervised(url, modelName, kwargs = {}, callback) {
      const self = this;
      return new Promise((resolve, reject) => {
        self._train(url, modelName, kwargs, callback).then((model) => {
          resolve(model);
        }).catch((error) => {
          reject(error);
        });
      });
    }
  };
};

exports.getFastTextClass = getFastTextClass;
