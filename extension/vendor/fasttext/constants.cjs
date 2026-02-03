'use strict';

Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });

const commonAssetsDir = "/fastText";
const commonAssetsModelsDir = "/fastText/models";
const trainFileInWasmFs = "train.txt";
const testFileInWasmFs = "test.txt";
const modelFileInWasmFs = "model.bin";
const IS_BROWSER = typeof window !== "undefined" && !!window.document && !!window.document.createElement;
const IS_WORKER = (
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  "importScripts" in globalThis && typeof globalThis.importScripts == "function"
);

exports.IS_BROWSER = IS_BROWSER;
exports.IS_WORKER = IS_WORKER;
exports.commonAssetsDir = commonAssetsDir;
exports.commonAssetsModelsDir = commonAssetsModelsDir;
exports.modelFileInWasmFs = modelFileInWasmFs;
exports.testFileInWasmFs = testFileInWasmFs;
exports.trainFileInWasmFs = trainFileInWasmFs;
