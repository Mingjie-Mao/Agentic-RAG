import { test, expect } from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs';

/**
 * The five-minute demo, executed rather than described (S8).
 *
 * A recording proves a demo happened once. This runs the same beats against the
 * released build every time, asserts what each beat is supposed to show, and saves a
 * screenshot per beat — so the demo cannot quietly drift away from the system.
 */

const shots = path.resolve('../artifacts/s8-demo');
fs.mkdirSync(shots, { recursive: true });
const beat = async (page: any, name: string) =>
  page.screenshot({ path: path.join(shots, `${name}.png`), fullPage: true });

const TABLE_DOC = path.resolve('../fixtures/s2', 'multipage-table.pdf');

test('浏览器回归：上传、证据、检索过程、权限、边界', async ({ page }) => {
  test.setTimeout(600_000);
  const notes: Record<string, unknown> = {};

  // 0:00–0:45 upload a document with headings and a table, watch it become searchable
  await page.goto('/');
  await page.locator('select[aria-label="账号"]').selectOption('support@xingqiao.demo');
  await page.getByRole('button', { name: '登录工作空间' }).click();
  await page.getByRole('button', { name: '添加资料', exact: true }).click();
  await page.getByLabel('上传文件').setInputFiles(TABLE_DOC);
  const accepted = page.waitForResponse(
    (r) => r.url().endsWith('/api/documents') && r.request().method() === 'POST',
  );
  await page.getByRole('button', { name: '上传并处理', exact: true }).click();
  const uploaded = await (await accepted).json();
  await page.getByRole('button', { name: /^资料库/ }).click();
  const row = page.locator(`[data-document-id="${uploaded.id}"]`);
  await expect(row).toContainText('可检索', { timeout: 180_000 });
  await row.locator('.document-title').click();
  await expect(page.getByTestId('version-panel')).toContainText('当前生效');
  const processing = await (
    await page.request.get(`/api/documents/${uploaded.id}/processing`)
  ).json();
  expect(processing.chunks.length).toBeGreaterThan(0);
  notes.beat1 = { chunks: processing.chunks.length, parser: processing.versions[0].parser };
  await beat(page, '1-upload-and-parse');
  await page.getByLabel('关闭解析检查').click();

  // 0:45–1:45 ask, then open the citation and land on the exact page
  await page.getByRole('button', { name: '知识问答', exact: true }).click();
  const answered = page.waitForResponse(
    (r) => r.url().endsWith('/api/chat') && r.request().method() === 'POST',
    { timeout: 240_000 },
  );
  await page.getByLabel('输入问题').fill('星桥产品上传接口单个文件的大小上限是多少？');
  await page.getByLabel('发送问题', { exact: true }).click();
  const answer = await (await answered).json();
  expect(answer.status).toBe('answered');
  expect(answer.citations.length).toBeGreaterThan(0);
  const card = page.getByTestId('answer-card').last();
  await card.locator('.citation-card').first().click();
  await expect(page.getByTestId('pdf-evidence')).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId('pdf-mark').first()).toBeVisible({ timeout: 30_000 });
  notes.beat2 = { status: answer.status, citations: answer.citations.length };
  await beat(page, '2-answer-and-citation');

  // 1:45–2:45 the retrieval inspector: which query ran, what came back, what was used
  await card.getByRole('button', { name: /检索过程/ }).click();
  const panel = page.getByTestId('trace-panel');
  await expect(panel.getByTestId('trace-method')).toHaveText('混合检索（RRF 融合）');
  await expect(panel).toContainText('关键词名次');
  await expect(panel).toContainText('向量名次');
  const admitted = answer.trace.candidates.filter((c: any) => c.admitted).length;
  await expect(panel.locator('.trace-table tbody tr.admitted')).toHaveCount(admitted);
  notes.beat3 = {
    method: answer.trace.method,
    candidates: answer.trace.candidates.length,
    admitted,
    total_ms: answer.trace.total_ms,
  };
  await beat(page, '3-retrieval-inspector');

  // 2:45–3:45 permissions: the same document, a different reader
  const engineer = await page.context().browser()!.newContext();
  const other = await engineer.newPage();
  await other.goto('/');
  await other.locator('select[aria-label="账号"]').selectOption('engineer@xingqiao.demo');
  await other.getByRole('button', { name: '登录工作空间' }).click();
  await other.getByRole('button', { name: /^资料库/ }).click();
  await expect(other.locator(`[data-document-id="${uploaded.id}"]`)).toHaveCount(0);
  const probe = (await other.request.get(`/api/documents/${uploaded.id}`)).status();
  expect(probe).toBe(404);
  notes.beat4 = { private_document_visible_to_other_group: false, detail_status: probe };
  await beat(other, '4-permission-boundary');
  await engineer.close();

  // 3:45–5:00 the boundary: no evidence is said, not guessed
  const refused = page.waitForResponse(
    (r) => r.url().endsWith('/api/chat') && r.request().method() === 'POST',
    { timeout: 240_000 },
  );
  await page.getByLabel('输入问题').fill('公司去火星出差的补贴是多少？');
  await page.getByLabel('发送问题', { exact: true }).click();
  const refusal = await (await refused).json();
  expect(refusal.status).toBe('insufficient_evidence');
  expect(refusal.citations).toEqual([]);
  await expect(page.getByTestId('answer-card').last()).toContainText('未找到足够依据');
  notes.beat5 = { status: refusal.status, citations: refusal.citations.length };
  await beat(page, '5-refusal');

  fs.writeFileSync(
    path.join(shots, 'beats.json'),
    JSON.stringify({ document_id: uploaded.id, ...notes }, null, 2) + '\n',
  );

  // Leave nothing behind, so the next run starts from the same state. The CSRF header
  // is required for every mutating request, which is why it is set explicitly here.
  const removed = await page.request.delete(`/api/documents/${uploaded.id}`, {
    headers: { 'X-Requested-With': 'AgenticRAG' },
  });
  expect(removed.status()).toBe(200);
  expect((await removed.json()).deleted).toBe(true);
});
