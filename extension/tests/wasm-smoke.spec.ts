import path from 'node:path';
import { test, expect } from 'playwright/test';
import { chromium } from 'playwright';

test('WASM smoke test loads in the extension', async ({}, testInfo) => {
  test.setTimeout(120000);

  const extensionRoot = path.resolve(path.dirname(testInfo.file), '..');
  const userDataDir = testInfo.outputPath('chromium-profile');
  const chromeExecutable =
    process.env.CHROME_PATH ||
    process.env.CHROMIUM_PATH ||
    process.env.PLAYWRIGHT_CHROME_EXECUTABLE_PATH ||
    undefined;

  const launchOptions: Parameters<typeof chromium.launchPersistentContext>[1] = {
    headless: false,
    args: [
      `--disable-extensions-except=${extensionRoot}`,
      `--load-extension=${extensionRoot}`,
      '--no-sandbox',
      '--disable-setuid-sandbox',
    ],
  };

  if (chromeExecutable) {
    launchOptions.executablePath = chromeExecutable;
  } else {
    launchOptions.channel = 'chrome';
  }

  const context = await chromium.launchPersistentContext(userDataDir, launchOptions);

  try {
    const existingWorker = context
      .serviceWorkers()
      .find((worker) => worker.url().startsWith('chrome-extension://'));
    const worker =
      existingWorker ||
      (await context.waitForEvent('serviceworker', { timeout: 10000 }));
    const extensionId = new URL(worker.url()).host;

    const page = await context.newPage();
    await page.goto(`chrome-extension://${extensionId}/tests/wasm-smoke.html`);
    await page.waitForFunction(() => (window as any).__wasmTestDone === true, null, {
      timeout: 120000,
    });

    const error = await page.evaluate(() => (window as any).__wasmTestError || null);
    expect(error).toBeNull();

    const results = await page.evaluate(() => (window as any).__wasmTestResults || null);
    expect(Array.isArray(results)).toBeTruthy();
    expect(results.length).toBeGreaterThan(0);

    for (const result of results) {
      expect(typeof result.probability).toBe('number');
      expect(result.probability).toBeGreaterThanOrEqual(0);
      expect(result.probability).toBeLessThanOrEqual(1);
      expect(typeof result.isScam).toBe('boolean');
      expect(result.threshold).toBeDefined();
      expect(result.scores && typeof result.scores).toBe('object');

      if (result.probability >= result.threshold) {
        expect(result.isScam).toBe(true);
      }
    }
  } finally {
    await context.close();
  }
});
