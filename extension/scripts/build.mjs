import { copyFile, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { build } from "esbuild";
import { format as prettierFormat, resolveConfig } from "prettier";

const ORT_DIST_DIR = path.resolve("node_modules/onnxruntime-web/dist");
const ORT_VENDOR_DIR = path.resolve("extension/vendor/onnxruntime-web");
const ORT_RUNTIME_FILES = [
  "ort.wasm.min.mjs",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
];

const ENTRY_POINTS = [
  "extension/src/background.ts",
  "extension/src/content-script.ts",
  "extension/src/offscreen.ts",
  "extension/src/popup.ts",
  "extension/src/options.ts",
  "extension/src/fasttext/classifier.ts",
  "extension/src/fasttext/fasttext.ts",
  "extension/src/transformer/classifier-transformer.ts",
  "extension/src/transformer/model-repo.ts",
  "extension/src/tests/wasm-smoke.ts",
  "extension/src/tests/transformer-smoke.ts",
];

const GENERATED_OUTPUTS = ENTRY_POINTS.map((entryPoint) =>
  path.resolve(
    entryPoint
      .replace(/^extension\/src\//, "extension/")
      .replace(/\.ts$/u, ".js"),
  ),
);

const copyOrtRuntimeAssets = async () => {
  try {
    await stat(ORT_DIST_DIR);
  } catch {
    throw new Error(
      "Missing onnxruntime-web runtime files. Run `pnpm install` before `pnpm extension:build`.",
    );
  }

  await mkdir(ORT_VENDOR_DIR, { recursive: true });

  for (const fileName of ORT_RUNTIME_FILES) {
    const sourcePath = path.join(ORT_DIST_DIR, fileName);
    const destinationPath = path.join(ORT_VENDOR_DIR, fileName);
    try {
      await stat(sourcePath);
    } catch {
      throw new Error(
        `onnxruntime-web runtime file missing: ${sourcePath}. Check pinned package version.`,
      );
    }
    await copyFile(sourcePath, destinationPath);
  }
};

const formatGeneratedOutputs = async () => {
  for (const outputPath of GENERATED_OUTPUTS) {
    const source = await readFile(outputPath, "utf8");
    const config = (await resolveConfig(outputPath)) || {};
    const formatted = await prettierFormat(source, {
      ...config,
      filepath: outputPath,
    });
    if (formatted !== source) {
      await writeFile(outputPath, formatted, "utf8");
    }
  }
};

await copyOrtRuntimeAssets();

await build({
  entryPoints: ENTRY_POINTS,
  outbase: "extension/src",
  outdir: "extension",
  bundle: false,
  format: "esm",
  target: ["es2022"],
  charset: "utf8",
  logLevel: "info",
  legalComments: "none",
  banner: {
    js: "// Generated from extension/src/*.ts by `npm run extension:build`.",
  },
});

await formatGeneratedOutputs();
