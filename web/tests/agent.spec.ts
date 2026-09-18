import { test, expect } from '@playwright/test';

test('Agent 页面公开执行模式、工具边界与可审计轨迹', async ({ page }) => {
  await page.goto('/');
  await page.locator('select[aria-label="账号"]').selectOption('admin@xingqiao.demo');
  await page.getByRole('button', { name: '登录工作空间' }).click();
  await page.getByRole('button', { name: '知识 Agent', exact: true }).click();

  await expect(page.getByRole('heading', { name: '让 Agent 分步查证资料' })).toBeVisible();
  await expect(page.getByLabel('Agent 任务目标')).toBeVisible();
  await expect(page.getByLabel('执行方式')).toHaveValue('auto');
  await expect(page.getByLabel('指定资料（可选）')).toContainText('跨资料检索');
  await expect(page.getByText(/资料读取每一步重新鉴权/)).toBeVisible();
});
