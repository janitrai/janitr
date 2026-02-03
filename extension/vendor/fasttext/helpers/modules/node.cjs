'use strict';

Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });

async function initializeFastTextModule(options = {}) {
  const { wasmPath, ...rest } = options;
  const fastTextModularized = (await import('../../core/fastText.node.js')).default;
  if (wasmPath) {
    return await fastTextModularized({
      // Binding js use the callback to locate wasm for now
      locateFile: (url, scriptDirectory) => {
        return typeof wasmPath === "string" ? wasmPath : wasmPath(url, scriptDirectory);
      },
      ...rest
    });
  }
  return await fastTextModularized(rest);
}
async function getFastTextModule(options = {}) {
  return await initializeFastTextModule(options);
}

exports.getFastTextModule = getFastTextModule;
exports.initializeFastTextModule = initializeFastTextModule;
