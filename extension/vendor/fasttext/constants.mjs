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

export { IS_BROWSER, IS_WORKER, commonAssetsDir, commonAssetsModelsDir, modelFileInWasmFs, testFileInWasmFs, trainFileInWasmFs };
