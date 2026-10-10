import assert from "node:assert/strict";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  configFile: false,
  server: { middlewareMode: true, hmr: false, ws: false, watch: null },
  optimizeDeps: { noDiscovery: true, include: [] },
});
after(() => server.close());
const { SourceExcerpt, highlightText, AgentTimeline, elapsedAt } =
  await server.ssrLoadModule("/src/DemoUX.tsx");

test("all literal supporting quotes are highlighted, including repetition and overlap", () => {
  const text = "允许 120 次。超过返回 HTTP 429。允许 120 次。";
  const html = renderToStaticMarkup(
    React.createElement(
      React.Fragment,
      null,
      highlightText(text, ["120 次", "HTTP 429", "允许 120 次", ""]),
    ),
  );
  assert.equal((html.match(/<mark>/g) ?? []).length, 3);
  assert.match(html, /<mark>HTTP 429<\/mark>/);
  assert.equal(html.replace(/<\/?mark>/g, ""), text);
});

test("Markdown formats headings, emphasis, lists and tables while retaining citations", () => {
  const text =
    "## 标准套餐\n\n每分钟 120 次，超额返回 HTTP 429。\n\n- **保留依据**\n\n| 套餐 | 配额 |\n| --- | --- |\n| 标准 | 120 次 |";
  const html = renderToStaticMarkup(
    React.createElement(SourceExcerpt, {
      text,
      quotes: ["120 次", "HTTP 429"],
    }),
  );
  assert.match(html, /<h4>标准套餐<\/h4>/);
  assert.match(html, /<strong>保留依据<\/strong>/);
  assert.match(html, /<table>/);
  assert.match(html, /<mark>HTTP 429<\/mark>/);
  assert.match(html, /查看原文文本/);
});

test("source HTML and dangerous links cannot become executable markup", () => {
  const text =
    "## 标题\n\n<script>alert(1)</script>\n\n[点击](javascript:alert(1))\n\n原句 <img src=x onerror=alert(1)>。";
  const html = renderToStaticMarkup(
    React.createElement(SourceExcerpt, { text, quotes: ["原句"] }),
  );
  assert.doesNotMatch(html, /<script|onerror=|href="javascript:/);
  const marked = renderToStaticMarkup(
    React.createElement(
      React.Fragment,
      null,
      highlightText("<script>bad</script>", ["<script>bad</script>"]),
    ),
  );
  assert.match(marked, /&lt;script&gt;/);
});

test("cancelled tasks display a boundary notice and actual lifecycle events", () => {
  const task = {
    status: "cancelled",
    created_at: "2026-10-09T08:00:00",
    updated_at: "2026-10-09T08:00:02Z",
    events: [
      {
        sequence: 1,
        event_type: "task_created",
        tool_name: null,
        payload: {},
        evidence_refs: [],
      },
      {
        sequence: 2,
        event_type: "task_cancelled",
        tool_name: null,
        payload: {},
        evidence_refs: [],
      },
    ],
  };
  const html = renderToStaticMarkup(
    React.createElement(AgentTimeline, { task, onCancel: async () => {} }),
  );
  assert.match(html, /任务已创建/);
  assert.match(html, /下一个执行边界/);
  assert.match(html, /额度不退回/);
  assert.equal(elapsedAt(task.created_at, task.updated_at), 2);
});
