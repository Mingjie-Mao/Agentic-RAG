import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  timeout: 240_000,
  expect: { timeout: 15_000 },
  workers: 1,
  globalSetup: "./tests/global-setup.ts",
  reporter: [
    ["list"],
    ["json", { outputFile: process.env.RAG_BROWSER_REPORT ?? "../artifacts/s1-browser-tests.json" }],
  ],
  use: {
    baseURL: process.env.RAG_BASE_URL ?? "http://127.0.0.1:8000",
    viewport: { width: 1440, height: 960 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
});
