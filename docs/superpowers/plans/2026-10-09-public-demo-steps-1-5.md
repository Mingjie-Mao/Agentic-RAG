# 公开展示与访客入口 1–5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking. 本次按用户要求在当前会话逐项执行，不分派子代理。

**Goal:** 完成并核验版本一致性、页面定位、首页结构、真实 Agent 展示和免登录受限访客入口。

**Architecture:** 首页和真实历史回放由 Cloudflare Pages 提供静态资源，`/live` 提供 React 应用；API 经现有 Tunnel 访问服务端。访客复用现有 cookie 会话、ACL 和每日调用预算；Agent 继续使用 LangGraph。历史回放明确标注录制时间与执行方式。

**Tech Stack:** HTML/CSS、React/TypeScript、FastAPI、SQLAlchemy、LangGraph、Cloudflare Pages、pytest。

---

本轮范围仅为已确认的计划 1–5。已有功能进入核验流程，不重复实现；保留当前工作区内其他开发改动与所有评测产物。Core 正式指标维持 pending，不运行或调整评测方法。本文件记录计划和实际验收结果，不将检查结果冒充新实现。

## Task 1：核对源码、构建与线上版本

**Files:** `site/index.html`、`web/dist/index.html`、`.runtime/pages-live/site/`、`scripts/publish_live_page.sh`。

- [x] 记录 `git status --short`，辨认本轮网站文件与其他已有改动。
- [x] 使用 SHA-256 比较线上首页、Live HTML、回放、截图与发布暂存目录；比较源码首页与发布首页。
- [x] 检查 `/release.json`、`/api/demo/status` 和两个公开页面，记录时间、状态及实际页面内容。
- [x] 将版本核验结果写入本文件；若存在差异，先定位构建或缓存原因，再决定最小修复。

**验证方式:** Python httpx 读取指定公开资源，`hashlib.sha256` 比较实际字节；浏览器确认首页和 Live 页面。预期所有指定静态资源一致，公开入口 HTTP 200。

## Task 2：明确两个页面定位

**Files:** `site/index.html`、`web/index.html`、`web/src/main.tsx`、`docs/public-demo-ux.md`。

- [x] 检查首页首屏是否说明项目价值并提供直接体验入口。
- [x] 检查 Live 品牌、虚构资料说明、项目介绍与 GitHub 返回链接。
- [x] 检查实时体验与历史回放的说明，确保没有把 Workflow 回放宣传为动态 Agent 效果。

**验证方式:** 浏览器检查渲染后的标题、入口与说明；对照源码与文档。预期首页承担项目介绍，Live 承担受限体验。

## Task 3：核验首页结构

**Files:** `site/index.html`、`tests/test_public_site.py`。

- [x] 检查正文顺序：价值说明 → 使用场景 → 核心能力 → 架构 → 评测 → 工程取舍。
- [x] 运行 `.venv/bin/pytest tests/test_public_site.py -q`；预期所有测试通过，章节、链接、图片与预渲染数据完整。
- [x] 浏览器检查桌面页面及英文切换；确认历史评测与 Core pending 保持明确区分。

## Task 4：核验真实 Agent 展示

**Files:** `site/shots/agent-demonstration-20261009.png`、`site/replay.html`、`site/replay-20261009T072349Z.json`、`scripts/record_public_demo.py`。

- [x] 检查真实录制包含任务目标、实际事件、工具、结果与引用，并有模型身份、录制时间和执行方式。
- [x] 检查首页展示真实截图并链接至可操作历史回放。
- [x] 浏览器操作回放步骤，确认工具事件、答案和支持原文可查看；不重新调用模型或占用访客额度。

**预期:** 展示记录中的真实 Workflow 多步执行；明确不能由该示例推断动态模式性能或正式评测成绩。

## Task 5：核验免登录受限访客入口

**Files:** `app/security.py`、`app/trial.py`、`app/main.py`、`web/src/main.tsx`、`tests/test_demo_entry.py`、`tests/test_trial.py`。

- [x] 公开 `/live` 未登录页面只有“进入只读演示”，没有账号、密码或所属组织表单；清楚说明虚构资料与共享每日额度。
- [x] 点击公开入口，确认服务端建立独立访客会话并进入应用，刷新后会话仍有效。
- [x] 运行 `.venv/bin/pytest tests/test_demo_entry.py tests/test_trial.py -q`；预期只读限制、预算不重置、独立 member 身份和禁用场景均通过。
- [x] 保存公开入口和进入后的截图，记录每日额度和仍依赖本机 Tunnel 的实际限制。

## 验收记录

执行过程中补充每项的证据和结果。全部完成后记录相关测试数、公开页面验证时间及剩余限制；不自动纳入其他开发改动或提交整个工作区。

### 核验发现的最小修正（执行前记录）

- Task 3：将首页导航中的“核心能力”移到“架构”之前，与正文顺序保持一致。
- Task 4：回放新增 `<p id="execution" class="muted"></p>`，在 `render()` 中由 `scenario.mode` 显示执行方式：`workflow` 显示“执行方式：固定 Workflow · 本例展示授权检索与证据提取，不代表动态 Agent 的执行效果。”；无 mode 的问答录制显示“执行方式：单轮 RAG 问答 · 历史录制结果。”。通过 `textContent` 写入，保持原录制 JSON 不变。
- 将两处静态页面修正用现有发布脚本发布，然后再次比较公开文件哈希并操作公开回放。
- Task 5：实际线上会话缺少 Secure。修改 `pages/proxy.js`，HTTPS 响应以官方 `Headers.getAll("Set-Cookie")` 读取各 cookie，以 `new Response(response.body, response)` 建立可写响应，逐条补充缺失的 `; Secure`；保留已有属性和多 cookie 的独立性。Node 本地测试使用标准 `getSetCookie()`。新增 `pages/proxy.test.mjs` 验证 HTTPS、HTTP、多 cookie、响应原状态以及离线错误；先确认 Secure 测试失败，再修复并运行 `node --test pages/proxy.test.mjs`，预期 4 项通过。

### 最终结果（2026-10-09，悉尼时间）

| 计划项 | 状态 | 验收证据 |
| --- | --- | --- |
| 1 版本核对 | 完成 | 线上 10 项 HTML / JS / CSS / JSON / 截图均 HTTP 200，SHA-256 与最新发布构建一致；源码首页和 Live 构建也一致。 |
| 2 页面定位 | 完成 | 首页说明项目价值；Live 为星桥虚构企业环境、只读访客体验，提供项目与 GitHub 返回链接；历史回放明确不代表实时执行或正式评测。 |
| 3 首页结构 | 完成 | 正文和导航顺序为场景、核心能力、架构、评测、取舍；中英文页面可查看，真实主截图已加载。历史数据预渲染，Core 70 保持待正式运行。 |
| 4 真实 Agent 展示 | 完成 | 录制任务 0e50c06e-2e87-4b2a-a2cc-aa78de15117a 含 15 个原始事件、search_documents 与 retrieve_evidence、120 次与 HTTP 429 答案及支持原文。公开回放步骤、最终结果和引用已手动操作，新增固定 Workflow 标注。 |
| 5 免登录访客入口 | 完成 | 公开页面无账号密码；点击进入独立 member 演示访客，刷新保持会话；后端显示只读与共享 2/10 余量，进入不会重置。缺少请求校验头返回 403，HTTPS cookie 为 Secure / HttpOnly / SameSite=strict。 |

最新部署：`https://37d4717d.agentic-rag.pages.dev`；生产域名仍为 `https://agentic-rag.pages.dev`。构建时间 `2026-10-09T08:12:31Z`（悉尼 19:12:31）；最终 HTTP 验收约悉尼 19:13:14。

本地实际验收记录：`.runtime/public-demo-captures/website-steps-1-5-20261009T081314Z.json`。该文件是网站验收记录，不是模型或 Core 评测。前一次缺少 Secure 的检查记录保留为 `website-steps-1-5-20261009T081038Z.json`，不覆盖。

验证：`pytest tests/test_public_site.py tests/test_demo_entry.py tests/test_trial.py -q` 为 11 项通过；`node --test pages/proxy.test.mjs` 为 4 项通过。TypeScript / Vite 构建通过，相关 Python 文件 ruff 通过，`git diff --check` 通过。

浏览器截图位于 `/private/tmp/agentic-rag-visitor-entry-20261009.png`、`/private/tmp/agentic-rag-public-visitor-20261009.png`、`/private/tmp/agentic-rag-public-home-20261009.png` 与 `/private/tmp/agentic-rag-public-replay-20261009.png`。

实际限制：Live 的 API 仍依赖本机和 Tunnel 在线；静态介绍与真实历史回放独立部署在 Pages，本机离线时可访问。当前录制展示固定 Workflow，不能证明动态策略提升。没有开展 Core 评测，也没有修改或删除其他开发中的评测文件。
