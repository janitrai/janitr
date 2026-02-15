import { build } from "esbuild";

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
