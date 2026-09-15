import {test,expect} from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs';

for(const filename of ['headings.docx','formula-missing.xlsx']) {
  test(`S2 上传、原文位置、分块和重复识别：${filename}`,async({page})=>{
    await page.goto('/');
    await page.locator('select[aria-label="账号"]').selectOption('support@xingqiao.demo');
    await page.getByRole('button',{name:'登录工作空间'}).click();
    async function upload(){
      await page.getByRole('button',{name:'添加资料',exact:true}).click();
      await page.getByLabel('上传文件').setInputFiles(path.resolve('../fixtures/s2',filename));
      const pending=page.waitForResponse(r=>r.url().endsWith('/api/documents')&&r.request().method()==='POST');
      await page.getByRole('button',{name:'上传并处理',exact:true}).click();
      const response=await pending;expect(response.status()).toBe(202);return response.json();
    }
    const first=await upload();
    await page.getByRole('button',{name:/^资料库/}).click();
    const row=page.locator(`[data-document-id="${first.id}"]`);
    await expect(row).toContainText('可检索',{timeout:120000});
    await row.getByRole('button',{name:filename,exact:true}).click();
    const dialog=page.getByRole('dialog',{name:'解析与分块检查'});
    await expect(dialog).toBeVisible();
    await expect(dialog).toContainText(filename.endsWith('xlsx')?'结果未缓存':'段落');
    await dialog.getByRole('button',{name:/分块结果/}).click();
    await expect(dialog.locator('.processing-block').first()).toBeVisible();
    const preview=await page.request.get(`/api/documents/${first.id}/processing`);
    const data=await preview.json();
    expect(data.blocks.length).toBeGreaterThan(0);expect(data.chunks.length).toBeGreaterThan(0);
    expect(data.chunks.every((c:any)=>c.locator.sources.length>0)).toBe(true);
    const original=await page.request.get(data.original_url);
    expect(await original.body()).toEqual(fs.readFileSync(path.resolve('../fixtures/s2',filename)));
    await page.screenshot({path:path.resolve('../artifacts',`s2-${filename}.png`),fullPage:true});
    await page.getByLabel('关闭解析检查').click();
    const second=await upload();
    expect(second.id).toBe(first.id);expect(second.version_id).toBe(first.version_id);expect(second.deduplicated).toBe(true);
  });
}
