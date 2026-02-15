import path from "node:path";
import { test, expect } from "playwright/test";
import { chromium } from "playwright";

test("transformer smoke test flags obvious scam text on extension test page", async ({}, testInfo) => {
  test.setTimeout(240000);
  test.skip(
    process.platform === "linux" && !process.env.DISPLAY,
    "Requires DISPLAY (headed Chromium extension test).",
  );

  const extensionRoot = path.resolve(path.dirname(testInfo.file), "..");
  const userDataDir = testInfo.outputPath("chromium-profile-transformer");
  const chromeExecutable =
    process.env.CHROME_PATH ||
    process.env.CHROMIUM_PATH ||
    process.env.PLAYWRIGHT_CHROME_EXECUTABLE_PATH ||
    undefined;

  const launchOptions: Parameters<typeof chromium.launchPersistentContext>[1] =
    {
      headless: false,
      args: [
        `--disable-extensions-except=${extensionRoot}`,
        `--load-extension=${extensionRoot}`,
        "--no-sandbox",
        "--disable-setuid-sandbox",
      ],
    };

  if (chromeExecutable) {
    launchOptions.executablePath = chromeExecutable;
  }

  const context = await chromium.launchPersistentContext(
    userDataDir,
    launchOptions,
  );

  try {
    const existingWorker = context
      .serviceWorkers()
      .find((worker) => worker.url().startsWith("chrome-extension://"));
    const worker =
      existingWorker ||
      (await context.waitForEvent("serviceworker", { timeout: 20000 }));
    const extensionId = new URL(worker.url()).host;

    const page = await context.newPage();
    await page.goto(
      `chrome-extension://${extensionId}/tests/transformer-smoke.html`,
    );
    await page.waitForFunction(
      () => (window as any).__transformerTestDone === true,
      null,
      { timeout: 240000 },
    );

    const error = await page.evaluate(
      () => (window as any).__transformerTestError || null,
    );
    expect(error).toBeNull();

    const result = await page.evaluate(
      () => (window as any).__transformerTestResults || null,
    );
    expect(result).toBeTruthy();
    expect(result.engine).toBe("transformer");
    expect(result.label).toBe("scam");
    expect(typeof result.scamScore).toBe("number");
  } finally {
    await context.close();
  }
});
