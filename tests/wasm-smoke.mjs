import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promises as fs } from "node:fs";
import { chromium } from "playwright";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const extensionRoot = path.join(repoRoot, "extension");

const executablePath =
  process.env.PLAYWRIGHT_CHROME_EXECUTABLE_PATH ||
  process.env.CHROME_PATH ||
  process.env.CHROMIUM_PATH ||
  undefined;

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".ftz": "application/octet-stream",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
};

const serveFile = async (req, res) => {
  const url = new URL(req.url || "/", "http://localhost");
  const requestedPath = decodeURIComponent(url.pathname);
  const safePath = path.normalize(requestedPath).replace(/^\.(\.[/\\])*/, "");
  const absolutePath = path.join(extensionRoot, safePath);

  try {
    const stat = await fs.stat(absolutePath);
    if (stat.isDirectory()) {
      res.writeHead(403);
      res.end("Forbidden");
      return;
    }
    const ext = path.extname(absolutePath);
    const contentType = mimeTypes[ext] || "application/octet-stream";
    const data = await fs.readFile(absolutePath);
    res.writeHead(200, {
      "Content-Type": contentType,
      "Cache-Control": "no-store",
    });
    res.end(data);
  } catch (err) {
    res.writeHead(404);
    res.end("Not found");
  }
};

const server = http.createServer((req, res) => {
  serveFile(req, res);
});

const listen = () =>
  new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      resolve(address.port);
    });
  });

const assert = (condition, message) => {
  if (!condition) {
    throw new Error(message);
  }
};

const run = async () => {
  const port = await listen();
  const url = `http://127.0.0.1:${port}/tests/wasm-smoke.html`;

  const launchOptions = {
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  };
  if (executablePath) {
    launchOptions.executablePath = executablePath;
  }
  const browser = await chromium.launch(launchOptions);

  try {
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: "networkidle0" });
    await page.waitForFunction("window.__wasmTestDone === true", {
      timeout: 120000,
    });

    const error = await page.evaluate(() => window.__wasmTestError || null);
    if (error) {
      throw new Error(error);
    }

    const results = await page.evaluate(() => window.__wasmTestResults || null);
    assert(
      Array.isArray(results) && results.length > 0,
      "No results returned.",
    );

    for (const result of results) {
      assert(typeof result.probability === "number", "Probability missing.");
      assert(
        result.probability >= 0 && result.probability <= 1,
        "Probability out of range.",
      );
      assert(typeof result.isFlagged === "boolean", "isFlagged missing.");
      assert(result.threshold !== undefined, "threshold missing.");
      assert(
        result.scores && typeof result.scores === "object",
        "scores missing.",
      );
      if (result.probability >= result.threshold) {
        assert(
          result.isFlagged === true,
          "isFlagged should be true when probability >= threshold.",
        );
      }
    }

    console.log("WASM smoke test passed.");
  } finally {
    await browser.close();
    server.close();
  }
};

run().catch((err) => {
  console.error(err.stack || err.message || err);
  server.close();
  process.exit(1);
});
