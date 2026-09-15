import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve("..");
const questions = JSON.parse(
  fs.readFileSync(path.join(root, "fixtures/questions.json"), "utf8"),
).questions;
const accounts = JSON.parse(
  fs.readFileSync(path.join(root, "fixtures/catalog.json"), "utf8"),
).users;
fs.mkdirSync(path.join(root, "artifacts/s1-questions"), { recursive: true });

for (const q of questions) {
  test(`${q.id} ${q.question}`, async ({ page }, testInfo) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto("/");
    const account = accounts.find((a: any) => a.id === q.user);
    await page
      .locator('select[aria-label="账号"]')
      .selectOption(account.username);
    await page.getByRole("button", { name: "登录工作空间" }).click();
    await expect(page.getByLabel("输入问题")).toBeVisible();
    const responsePromise = page.waitForResponse(
      (r) => r.url().endsWith("/api/chat") && r.request().method() === "POST",
      { timeout: 220_000 },
    );
    await page.getByLabel("输入问题").fill(q.question);
    await page.getByLabel("发送问题", { exact: true }).click();
    const response = await responsePromise;
    const result = await response.json();
    const text = (result.claims ?? []).map((c: any) => c.text).join("\n");
    const foundSources = (result.citations ?? []).map((c: any) =>
      c.document_id.replace(/^seed-/, ""),
    );
    const checks = {
      http_ok: response.status() === 200,
      status_ok:
        q.expected === "conflict"
          ? result.status === "conflict"
          : result.status === q.expected,
      expected_facts: q.facts.every((fact: string) => text.includes(fact)),
      expected_sources: q.sources.every((key: string) =>
        foundSources.includes(key),
      ),
      no_forbidden_marker: (q.forbidden ?? []).every(
        (s: string) => !JSON.stringify(result).includes(s),
      ),
    };
    fs.writeFileSync(
      path.join(root, `artifacts/s1-questions/${q.id}.json`),
      JSON.stringify(
        {
          question: q,
          checks,
          response: result,
          model_execution: "real_local_ollama",
          interface: "chromium_browser",
        },
        null,
        2,
      ),
    );
    expect(response.status()).toBe(200);
    const card = page.getByTestId("answer-card").last();
    await expect(card).toBeVisible();
    expect(checks.status_ok).toBeTruthy();
    expect(checks.no_forbidden_marker).toBeTruthy();
    if (q.expected === "conflict") {
      expect(checks.expected_facts).toBeTruthy();
      expect(checks.expected_sources).toBeTruthy();
      expect(result.message).toMatch(/冲突/);
    }
    if (q.expected === "answered") {
      expect(checks.expected_facts).toBeTruthy();
      expect(checks.expected_sources).toBeTruthy();
      await card.locator(".citation-card").first().click();
      await expect(
        page.locator(".evidence-content .source-text"),
      ).toBeVisible();
      const evidenceResponse = await page.request.get(
        result.citations[0].preview_url,
      );
      expect(evidenceResponse.status()).toBe(200);
      const evidence = await evidenceResponse.json();
      expect(evidence.text).toBe(result.citations[0].text);
      const original = await page.request.get(evidence.original_url);
      expect(original.status()).toBe(200);
      if (evidence.media_type === "application/pdf")
        expect((await original.body()).subarray(0, 5).toString()).toBe("%PDF-");
      if (["Q01", "Q08"].includes(q.id)) {
        await page.screenshot({
          path: path.join(root, `artifacts/s1-${q.id}-evidence.png`),
          fullPage: true,
        });
        await testInfo.attach("问答与证据", {
          path: path.join(root, `artifacts/s1-${q.id}-evidence.png`),
          contentType: "image/png",
        });
      }
    } else if (q.expected === "insufficient_evidence") {
      expect(result.citations).toEqual([]);
      await expect(card).toContainText("未找到足够依据");
    }
    await page.getByRole("button", { name: "问答记录", exact: true }).click();
    await expect(
      page.getByTestId("answer-card").filter({ hasText: q.question }).first(),
    ).toBeVisible();
    expect(errors).toEqual([]);
  });
}

test("网页上传、处理状态与私有范围", async ({ page }) => {
  await page.goto("/");
  await page
    .locator('select[aria-label="账号"]')
    .selectOption("support@xingqiao.demo");
  await page.getByRole("button", { name: "登录工作空间" }).click();
  await page.getByRole("button", { name: "添加资料", exact: true }).click();
  await page
    .getByLabel("上传文件")
    .setInputFiles({
      name: "网页验收记录.md",
      mimeType: "text/markdown",
      buffer: Buffer.from(
        "# 网页验收记录\n\n这是一份仅供网页上传验收的虚构文档，验收标记为 UI-S1-OK。" + Date.now(),
      ),
    });
  await page.getByLabel("访问范围").selectOption("private");
  const response = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/documents") && r.request().method() === "POST",
  );
  await page.getByRole("button", { name: "上传并处理", exact: true }).click();
  const uploaded = await (await response).json();
  expect(["queued", "processing"]).toContain(uploaded.status);
  await page.getByRole("button", { name: /^资料库/ }).click();
  const row = page.locator(`[data-document-id="${uploaded.id}"]`);
  await expect(row).toContainText("可检索", { timeout: 120_000 });
  await expect(row).toContainText("私有");
  fs.writeFileSync(
    path.join(root, "artifacts/s1-upload.json"),
    JSON.stringify(
      {
        document_id: uploaded.id,
        version_id: uploaded.version_id,
        initial_status: uploaded.status,
        final_status: "ready",
        scope: "private",
      },
      null,
      2,
    ),
  );
  await page.screenshot({
    path: path.join(root, "artifacts/s1-documents.png"),
    fullPage: true,
  });
});
