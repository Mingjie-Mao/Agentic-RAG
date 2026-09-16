import { test, expect } from '@playwright/test';
import path from 'node:path';

// Seed documents were ingested before the layout parser, so they carry no
// coordinates. A freshly uploaded PDF goes through the current pipeline and
// must therefore highlight the exact span that was indexed.
test('S4 PDF 证据：打开对应页并按解析坐标高亮', async ({ page }) => {
  await page.goto('/');
  await page.locator('select[aria-label="账号"]').selectOption('support@xingqiao.demo');
  await page.getByRole('button', { name: '登录工作空间' }).click();

  await page.getByRole('button', { name: '添加资料', exact: true }).click();
  await page.getByLabel('上传文件').setInputFiles(path.resolve('../fixtures/s2', 'multipage-table.pdf'));
  const accepted = page.waitForResponse(
    (r) => r.url().endsWith('/api/documents') && r.request().method() === 'POST',
  );
  await page.getByRole('button', { name: '上传并处理', exact: true }).click();
  const document = await (await accepted).json();

  await page.getByRole('button', { name: /^资料库/ }).click();
  await expect(page.locator(`[data-document-id="${document.id}"]`)).toContainText('可检索', {
    timeout: 120000,
  });

  const processing = await (await page.request.get(`/api/documents/${document.id}/processing`)).json();
  const located = processing.chunks.find(
    (c: any) => (c.locator.provenance ?? []).length > 0 && c.locator.page,
  );
  expect(located, '当前解析管线必须给 PDF 分块写入页码与坐标').toBeTruthy();

  await page.getByRole('button', { name: '知识问答', exact: true }).click();
  const pending = page.waitForResponse(
    (r) => r.url().endsWith('/api/chat') && r.request().method() === 'POST',
    { timeout: 220_000 },
  );
  await page.getByLabel('输入问题').fill('地区配额表里区域 1 的每日配额是多少次？');
  await page.getByLabel('发送问题', { exact: true }).click();
  const result = await (await pending).json();
  const index = result.citations.findIndex((c: any) => c.document_id === document.id);
  expect(index, '答案必须引用刚上传的 PDF').toBeGreaterThanOrEqual(0);

  const card = page.getByTestId('answer-card').last();
  await card.locator('.citation-card').nth(index).click();

  const viewer = page.getByTestId('pdf-evidence');
  await expect(viewer).toBeVisible({ timeout: 60000 });
  await expect(viewer.locator('canvas')).toBeVisible();

  const marks = page.getByTestId('pdf-mark');
  await expect(marks.first()).toBeVisible({ timeout: 30000 });
  const count = await marks.count();
  expect(count).toBeGreaterThan(0);

  // A highlight that is zero-sized or outside the page is not a highlight.
  const canvasBox = await viewer.locator('canvas').boundingBox();
  expect(canvasBox).not.toBeNull();
  for (let i = 0; i < count; i += 1) {
    const box = await marks.nth(i).boundingBox();
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThan(1);
    expect(box!.height).toBeGreaterThan(1);
    expect(box!.x).toBeGreaterThanOrEqual(canvasBox!.x - 2);
    expect(box!.y).toBeGreaterThanOrEqual(canvasBox!.y - 2);
    expect(box!.x + box!.width).toBeLessThanOrEqual(canvasBox!.x + canvasBox!.width + 2);
    expect(box!.y + box!.height).toBeLessThanOrEqual(canvasBox!.y + canvasBox!.height + 2);
  }

  await page.screenshot({ path: path.resolve('../artifacts', 's4-pdf-highlight.png'), fullPage: true });
});

// The debug page must explain a retrieval, not just decorate it: which query ran,
// what came back, and which candidates actually reached the model.
test('S4 检索调试页：候选、准入与阶段耗时', async ({ page }) => {
  await page.goto('/');
  await page.locator('select[aria-label="账号"]').selectOption('support@xingqiao.demo');
  await page.getByRole('button', { name: '登录工作空间' }).click();

  const pending = page.waitForResponse(
    (r) => r.url().endsWith('/api/chat') && r.request().method() === 'POST',
    { timeout: 220_000 },
  );
  await page.getByLabel('输入问题').fill('星桥标准套餐每分钟可以调用多少次 API？');
  await page.getByLabel('发送问题', { exact: true }).click();
  const result = await (await pending).json();

  const card = page.getByTestId('answer-card').last();
  await card.getByRole('button', { name: /检索过程/ }).click();
  const panel = page.getByTestId('trace-panel');
  await expect(panel).toBeVisible();

  await expect(panel).toContainText('原始问题');
  await expect(panel).toContainText('实际检索问题');
  await expect(panel).toContainText(result.trace.original_question);

  const rows = panel.locator('.trace-table tbody tr');
  await expect(rows).toHaveCount(result.trace.candidates.length);
  expect(result.trace.candidates.length).toBeGreaterThan(0);

  // Admission must be reported per candidate, and agree with the returned evidence.
  const admitted = result.trace.candidates.filter((c: any) => c.admitted);
  expect(admitted.length).toBeGreaterThan(0);
  await expect(panel.locator('.trace-table tbody tr.admitted')).toHaveCount(admitted.length);
  const dropped = result.trace.candidates.filter((c: any) => !c.admitted);
  for (const candidate of dropped) {
    expect(['below_min_similarity', 'context_budget_exhausted']).toContain(
      candidate.excluded_because,
    );
  }
  expect(admitted.map((c: any) => c.evidence_id)).toEqual(
    admitted.map((_: any, i: number) => `E${i + 1}`),
  );

  const labels: Record<string, string> = {
    dense: '向量检索',
    bm25: '关键词检索',
    hybrid: '混合检索（RRF 融合）',
  };
  await expect(panel.getByTestId('trace-method')).toHaveText(labels[result.trace.method]);
  await expect(panel).toContainText('上下文');

  // Hybrid must expose where each candidate came from, or the ranking change is unreadable.
  if (result.trace.method === 'hybrid') {
    await expect(panel).toContainText('关键词名次');
    await expect(panel).toContainText('向量名次');
    const ranked = result.trace.candidates.filter(
      (c: any) => c.bm25_rank !== null || c.dense_rank !== null,
    );
    expect(ranked.length).toBe(result.trace.candidates.length);
    for (const candidate of result.trace.candidates) {
      expect(candidate.fusion_score).toBeGreaterThan(0);
    }
  }

  // The reader must be told what was searched, and only what they may see.
  const scope = page.getByTestId('scope-note').last();
  await expect(scope).toContainText(`${result.trace.scope.readable_documents} 份资料`);
  expect(result.trace.scope.searchable_documents).toBeLessThanOrEqual(
    result.trace.scope.readable_documents,
  );
  const listed = await (await page.request.get('/api/documents')).json();
  expect(listed.length).toBe(result.trace.scope.readable_documents);
  await expect(card.getByTestId('answer-status')).toHaveText('附原文依据');
  await page.screenshot({ path: '../artifacts/s4-trace-panel.png', fullPage: true });
});

// Chunk locations are bound to one version and one processing configuration.
// The document page must say which, or a citation cannot be audited later.
test('S4 文档页：版本身份与处理来源可核对', async ({ page }) => {
  await page.goto('/');
  await page.locator('select[aria-label="账号"]').selectOption('support@xingqiao.demo');
  await page.getByRole('button', { name: '登录工作空间' }).click();
  await page.getByRole('button', { name: /^资料库/ }).click();

  const listed = await (await page.request.get('/api/documents')).json();
  const target = listed.find((d: any) => d.status === 'ready');
  expect(target, '需要至少一份已处理完成的资料').toBeTruthy();

  const detail = await (await page.request.get(`/api/documents/${target.id}/processing`)).json();
  expect(detail.versions.length).toBeGreaterThan(0);
  const active = detail.versions.filter((v: any) => v.active);
  expect(active.length, '同一份资料只能有一个生效版本').toBe(1);
  expect(active[0].version_id).toBe(detail.active_version_id);

  await page.locator(`[data-document-id="${target.id}"]`).locator('.document-title').click();
  const panel = page.getByTestId('version-panel');
  await expect(panel).toBeVisible();
  await expect(panel.locator('.version')).toHaveCount(detail.versions.length);

  const row = panel.locator(`[data-version-id="${active[0].version_id}"]`);
  await expect(row).toHaveClass(/active/);
  await expect(row).toContainText('当前生效');
  await expect(row).toContainText(active[0].filename);
  await expect(row).toContainText(active[0].content_hash.slice(0, 16));
  await expect(row).toContainText(active[0].parser);
  await expect(row).toContainText(`${active[0].chunks} 个分块`);

  // The chunk count shown must match the chunks the same endpoint returns.
  expect(active[0].chunks).toBe(detail.chunks.length);
  await page.screenshot({ path: '../artifacts/s4-version-panel.png', fullPage: true });
});

// After S6 reprocessing, the seeded demo corpus carries layout coordinates too, so a
// citation from a seeded PDF must highlight rather than fall back to "no coordinates".
test('S6 种子资料重新处理后：引用可在页内高亮', async ({ page }) => {
  await page.goto('/');
  await page.locator('select[aria-label="账号"]').selectOption('support@xingqiao.demo');
  await page.getByRole('button', { name: '登录工作空间' }).click();

  await expect(page.getByLabel('输入问题')).toBeVisible();
  const pending = page.waitForResponse(
    (r) => r.url().endsWith('/api/chat') && r.request().method() === 'POST',
    { timeout: 220_000 },
  );
  await page.getByLabel('输入问题').fill('星桥产品上传接口单个文件的大小上限是多少？');
  await page.getByLabel('发送问题', { exact: true }).click();
  const result = await (await pending).json();
  const index = result.citations.findIndex((c: any) => c.document_id === 'seed-upload-guide');
  expect(index, '答案应引用那份种子 PDF').toBeGreaterThanOrEqual(0);

  const card = page.getByTestId('answer-card').last();
  await card.locator('.citation-card').nth(index).click();
  const viewer = page.getByTestId('pdf-evidence');
  await expect(viewer).toBeVisible({ timeout: 60000 });
  await expect(page.getByTestId('pdf-mark').first()).toBeVisible({ timeout: 30000 });
  expect(await page.getByTestId('pdf-mark').count()).toBeGreaterThan(0);

  // A highlight over a blank page is not evidence. Chinese PDFs need CMap data, and
  // without it pdf.js renders nothing while still reporting a successful render, so
  // assert that the canvas actually has ink on it.
  const inked = await page.evaluate(() => {
    const canvas = document.querySelector('[data-testid="pdf-evidence"] canvas') as HTMLCanvasElement;
    const context = canvas.getContext('2d')!;
    const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
    let dark = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] < 200 && data[i + 1] < 200 && data[i + 2] < 200) dark += 1;
    }
    return dark / (data.length / 4);
  });
  expect(inked, 'PDF 页面必须渲染出实际内容，而不是一张白页').toBeGreaterThan(0.001);

  await page.screenshot({ path: '../artifacts/s6-seed-highlight.png', fullPage: true });
});
