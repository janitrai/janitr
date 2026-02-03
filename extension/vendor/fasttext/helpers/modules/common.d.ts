import type { InitializeFastTextModuleOptions } from './types';
/**
 * If `document.currentScript.src` is falsy value, it will load `fasttext.common.wasm` from public root directly by default,
 *
 * You can download it from https://github.com/yunsii/fasttext.wasm.js/blob/master/src/core/fastText.common.wasm
 *
 * You can also use `locateFile` callback to custom `fasttext.common.wasm` full path.
 */
export declare function initializeFastTextModule(options?: InitializeFastTextModuleOptions): Promise<import("../../core/fastText.common").FastTextModule>;
export declare function getFastTextModule(options?: InitializeFastTextModuleOptions): Promise<import("../../core/fastText.common").FastTextModule>;
