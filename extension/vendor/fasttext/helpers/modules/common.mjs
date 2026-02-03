async function initializeFastTextModule(options = {}) {
  const { wasmPath, ...rest } = options;
  const fastTextModularized = (await import('../../core/fastText.common.js')).default;
  return await fastTextModularized({
    // Binding js use the callback to locate wasm for now
    locateFile: (url, scriptDirectory) => {
      if (wasmPath) {
        return typeof wasmPath === "string" ? wasmPath : wasmPath(url, scriptDirectory);
      }
      return (scriptDirectory || "/") + url;
    },
    ...rest
  });
}
async function getFastTextModule(options = {}) {
  return await initializeFastTextModule(options);
}

export { getFastTextModule, initializeFastTextModule };
