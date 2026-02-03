/// <reference types="emscripten" />
export interface InitializeFastTextModuleOptions {
    /**
     * In Node.js, it will load `fasttext.node.wasm` automatically, it is usually not necessary to use this option.
     *
     * ===================================================
     *
     * In others environments, if `document.currentScript.src` is falsy value, it will load `fasttext.common.wasm` from public root directly by default,
     * You can download it from:
     * https://github.com/yunsii/fasttext.wasm.js/blob/master/src/core/fastText.common.wasm
     *
     * You can also use `wasmPath` callback to custom `fasttext.common.wasm` full path.
     */
    wasmPath?: string | EmscriptenModule['locateFile'];
}
