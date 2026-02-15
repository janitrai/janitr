declare const chrome: any;
declare const browser: any;

declare module "../vendor/fasttext/main/common.mjs" {
  export const getFastTextClass: (
    options?: Record<string, unknown>,
  ) => Promise<any>;
  export const getFastTextModule: (options?: Record<string, unknown>) => any;
}

declare module "./fasttext_wasm.js" {
  const fastTextModularized: (
    options?: Record<string, unknown>,
  ) => Promise<any>;
  export default fastTextModularized;
}

interface Window {
  __wasmTestResults?: unknown;
  __wasmTestDone?: boolean;
  __wasmTestError?: string;
}
