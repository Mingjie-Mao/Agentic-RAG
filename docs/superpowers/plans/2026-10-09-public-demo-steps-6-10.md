# 公开演示 6–10 Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [x]`) syntax for tracking. 本次按用户要求在当前会话逐项执行，沿用 writing-plans 的文件与验收记录方式。

**Goal:** 完成引导式场景、真实等待反馈、准确 Agent 状态、清楚的引用阅读和可恢复的测试资料清理。

**Architecture:** 保留原问答接口与 LangGraph 编排，UI 只读取真实阶段和审计事件。Markdown 阅读复用成熟组件；测试上传写本地收据，成功或失败立即清理，中断后按收据或严格双标记软删除，保留原件和历史记录。

**Tech Stack:** React/TypeScript、react-markdown / remark-gfm、FastAPI、SQLAlchemy、pytest、Node test、Cloudflare Pages。

---

## Task 6：提供四个有目的的体验入口

**Files:** `web/src/DemoUX.tsx`、`web/src/main.tsx`、`web/src/style.css`。

- [x] 四个入口在回答后仍可找到：核对引用、比较政策版本、资料冲突、依据不足拒答。
- [x] 点击只填入问题，发送后才消耗额度；每个入口解释要观察的能力。
- [x] Agent 保留三个业务任务示例；权限对照入口明确标注固定角色历史快照。
- [x] 在浏览器操作四个入口，核对问题、说明和权限对照链接，不自动运行四次模型。

## Task 7：真实等待、停止等待和超时恢复

**Files:** `web/src/main.tsx`、`app/chat_progress.py`、`tests/test_demo_entry.py`。

- [x] 将经过秒数计时与网络轮询分离；进度和结果查询使用 8 秒网络超时。
- [x] 120 秒后停止等待原 HTTP 请求并继续查询结果；明确后端可能仍执行、已预留额度不退回，未知结果先查历史再重试。
- [x] 恢复结果时清除过期错误提示；状态轮询异常不得将旧请求的错误写入新请求。
- [x] 显示可访问历史回答的实际中位耗时；无样本显示未测，配额为 0 时显示回放入口。
- [x] 实际运行一次公开问答，检查阶段、计时、停止等待、结果恢复和额度变化。其余异常情况用针对性测试验证，不重置共享额度。

## Task 8：统一 Agent 模式、状态和执行轨迹

**Files:** `web/src/DemoUX.tsx`、`web/src/main.tsx`、`tests/test_agent.py`、`tests/test_agent_async_memory.py`。

- [x] 所有已知模式使用准确名称，访客文案和选择项仅说明当前可用功能。
- [x] 时间线包含排队、开始、实际工具和终态，原始审计仍可展开。
- [x] 取消处理中防止重复点击，取消后的说明准确表达在执行边界生效。
- [x] 验证已有真实任务轨迹及取消业务测试，不以回放推断动态模式效果。

## Task 9：清楚呈现引用与原文

**Files:** `web/src/DemoUX.tsx`、`web/src/main.tsx`、`web/src/PdfEvidence.tsx`、`web/src/style.css`、`web/tests/demo-ux.test.mjs`。

- [x] Markdown 标题、列表、表格与强调使用 react-markdown / remark-gfm，保留原文文本切换。
- [x] 同一行所有支持句子均高亮，引用只绑定当前答案的 quotes；保持文本转义和安全 URL 默认行为。
- [x] 加载可见、错误可重试，关闭或切换引用后过期请求不覆盖新状态。
- [x] 确认侧栏放大、PDF 缩放与真实解析坐标高亮；坐标缺失时如实说明。
- [x] 用 SSR 单元测试验证多句高亮、HTML / 危险链接处理与格式化结构，再手动查看真实引用。

## Task 10：隔离并清理网页测试残留

**Files:** `web/tests/fixtures.ts`、`web/tests/global-setup.ts`、`scripts/cleanup_browser_uploads.py`、`tests/test_browser_upload_cleanup.py`。

- [x] 核对已有 8 条同名记录已软删除，公开列表和 Agent 选择器不出现验收资料。
- [x] 测试成功或断言失败均执行清理；上传成功时写 `.runtime/browser-uploads/` 收据，收据不含会话或密码。
- [x] 中断后可按收据恢复清理，仅允许虚构测试账号所有、私有、非 seed 文档；旧遗留仍要求文件名和双标记。
- [x] 使用内存数据库验证误删保护、缺失文件和中断收据恢复；真实数据库默认 dry-run，不覆盖评测产物。

## 验收与发布

- [x] 运行相关 pytest、Node 组件测试、TypeScript / Vite 构建与 diff 检查；只增加能验证边界的测试。
- [x] 发布前端，核对公开 HTML / JS / CSS 与发布构建哈希，保存真实浏览器截图。
- [x] 将每项完成证据、测试数、当前共享额度和实际限制补充在本文件，汇报 6–10。

## 最终验收记录（2026-10-09，悉尼时间）

| 步骤 | 结果与证据 |
| --- | --- |
| 6 体验入口 | 四个场景均可填入正确问题；连续点击四项后额度保持 2/10。回答后可重新展开入口。Agent 三个业务示例及固定角色历史权限对照链接可见。 |
| 7 等待与恢复 | 真实公开问答显示生成阶段和独立秒数；23 秒点击停止等待，随后通过状态和历史接口取回 120 次答案，总耗时 39.6462 秒，仅计一次额度。恢复后无过期错误提示。近期 6 次真实可访问回答的中位数 38 秒。120 秒自动停止原请求的路径已实现并构建通过；本次线上手动走查未等待到 120 秒。状态过期、租户隔离、失败与预算边界用单元测试核验。 |
| 8 Agent 状态 | 公开界面仅显示自动路由与固定工作流，并明确两者在访客下均执行固定路径。既有真实任务显示创建、开始、工具、核验与终态，保留审计详情。测试确认重复取消返回“已结束”409、不重复写取消事件、不退预留额度，且不能取消他人的任务。 |
| 9 引用阅读 | 实际 Markdown 引用完成原文 / 格式化切换、句子高亮、侧栏放大。实际 PDF 第 1 页成功渲染并显示 1 个解析坐标高亮；缩放到 125% 后仍保留该高亮。Markdown SSR 测试验证多句 / 重复 / 重叠高亮、标题列表表格和安全文本。引用高亮只绑定当前答案，加载失败支持再次鉴权重试。 |
| 10 残留与隔离 | 8 条旧同名验收记录均已软删除，0 条活动记录，11 个历史版本保留。公开资料库 21 份资料及 Agent 选择器没有验收标题。清理脚本 dry-run 返回 matched=0。新增测试上传收据与启动恢复，跳过重复识别的已有文档，并限制为支持测试账号私有、非 seed 上传；边界与原件保留由内存数据库测试核验。 |

验证结果：相关 pytest **34 passed**；Node SSR / 代理测试 **8 passed**；TypeScript 应用构建与测试 fixture 类型检查通过；Vite 构建、相关 ruff、`git diff --check` 通过。没有运行正式 Core 或修改评测产物。没有用浏览器自动回归命令覆盖历史截图或报告；线上界面由当前会话实际操作。

最新部署：`https://dbf20029.agentic-rag.pages.dev`，生产入口 `https://agentic-rag.pages.dev/live`。构建时间 `2026-10-09T09:04:43Z`，即悉尼 20:04:43；最终验收约悉尼 20:06:23。公开首页、Live HTML、回放、构建信息和 JS / CSS 共 6 项资源均 HTTP 200，SHA-256 与发布构建一致。

本地验收文件：`.runtime/public-demo-captures/website-steps-6-10-20261009T090623Z.json`。真实问答记录 ID `0840c953-9824-4bd9-b88b-76729f113acc`，模型使用既有本地配置；该走查不属于正式评测。

截图：`/private/tmp/agentic-rag-four-scenarios-20261009.png`、`/private/tmp/agentic-rag-real-waiting-20261009.png`、`/private/tmp/agentic-rag-quoted-source-20261009.png`、`/private/tmp/agentic-rag-pdf-reading-20261009.png`。

剩余实际限制：Live API 仍依赖本机与 Tunnel 在线，访客当前共享余量 **1/10**，未重置或绕过额度；回放不消耗额度。强制中断恢复依赖已收到上传响应并写出的本地收据，不能覆盖服务器创建后、客户端尚未观察响应就被杀死的极短窗口；旧 Markdown 遗留另有双标记兜底。测试收据、本地运行文件与日志不提交。
