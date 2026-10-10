# 公开展示与演示交付

2026-10-09 完成原计划 1–16，2026-10-10 完成招聘展示补充；生产入口为 https://agentic-rag.pages.dev/、https://agentic-rag.pages.dev/live 和 https://agentic-rag.pages.dev/replay.html。最新展示部署为 `https://e592214b.agentic-rag.pages.dev`，英文概览为 `/overview.html`；下面 1–16 的测试数据保留为当时的验收快照。

## 2026-10-10 招聘展示补充

首屏改为可验证企业知识 Agent；重复图改为真实事件生成的执行摘要，包含两个实际工具调用、证据与原始回答。新增 Agent 编排、跨文档检索和版本语义能力，增加 RAG / Workflow / Hybrid / Dynamic 对比与当前开放范围。历史结果和工程规模分组，移除无独立 CI 链接的 683 宣传卡，提供有日期与范围说明的本地验证记录。截图支持键盘打开、Esc / 明确按钮关闭及焦点返回；引用图局部放大，手机指标单列。独立英文技术概览和已知边界章节已发布。

本轮 6 项相关测试及 TypeScript / Vite 构建通过，15 项公开资源哈希匹配；实际手机和键盘交互验收通过。没有运行模型或 Core，也没有修改原始历史数据。详细记录见 `docs/superpowers/plans/2026-10-10-recruiter-showcase.md`。

## 完成范围

| 计划 | 结果 |
| --- | --- |
| 1–5 | 版本核对、项目 / 产品定位、首页重排、真实 Agent 截图与回放、无需账号密码的独立只读访客会话。 |
| 6–10 | 四种体验场景、真实阶段与等待反馈、停止等待与结果恢复、准确模式和审计轨迹、格式化引用 / PDF 阅读、验收残留软删除和中断恢复。 |
| 11–16 | 当前鉴权与角色快照对照、无 API 静态回放、历史评测与 pending 口径、移动端与文字对比度、最终回归和交付记录。 |

详细记录：`docs/superpowers/plans/2026-10-09-public-demo-steps-1-5.md`、`2026-10-09-public-demo-steps-6-10.md`、`2026-10-09-public-demo-steps-11-16.md`。运行与发布说明见 `docs/public-demo-ux.md`。

## 发布与验证

- 原计划 1–16 部署 `https://c4330829.agentic-rag.pages.dev`；构建于 2026-10-09 悉尼 20:37:16。
- 12 项公开 HTML、JS、CSS、JSON、PDF worker、CMap 和字体资源 HTTP 200，SHA-256 与发布暂存构建一致。
- 最新完整 pytest：1164 passed、36 skipped、1 项依赖弃用警告；8 项 Node 组件 / 代理测试通过。TypeScript 应用构建、网页测试 fixture 类型检查、ruff、文档一致性和 diff 检查通过。36 项跳过测试未计为通过。
- 验收期间其他开发新增测试后，曾有 12 项共享 fixture 找不到的错误；当前文件改为显式引用后，源测试与使用方联合运行 200 项通过，最新完整回归无错误。保留先前验收记录，不将中间失败计为通过。
- 支持组与工程组的当前权限与快照一致；访客请求支持组文档为 200，工程组文档与不存在文档均为 404。客户端指定 admin、其他租户或工程组字段不能改变访客身份。
- 仅静态文件的预览没有 API，真实历史回放仍可查看答案、原文和权限对照；在线入口显示服务不可用与回放链接。
- 375px 手机视口检查菜单、场景、输入与引用面板，无整页横向溢出；原文关闭后恢复引用按钮焦点。辅助文字加深，原文阅读保持至少 12px 的辅助文字和 15px 正文。
- 实际问答走查在 23 秒停止等待，39.6 秒后自动取回答案；没有重复提交，额度只计一次。PDF 渲染和 125% 缩放后仍有真实解析坐标高亮。
- 8 条旧网页验收记录全部软删除，11 个历史版本保留；公开资料库与 Agent 选择器无这些记录。

本地最终验收数据：`.runtime/public-demo-captures/website-steps-11-16-20261009T094846Z.json`；先前的权限与无 API 走查记录通过其中的引用保留。这是网站验收，不能作为 Core 成绩或动态策略提升证据。冻结评测数据、历史实验文件和其他开发改动均保留。

## 文件职责

- `site/index.html` / `site/replay.html`：项目介绍、历史评测、静态真实回放。
- `site/replay-20261009T072349Z.json` / `site/permission-demo-20261009.json`：有时间和身份信息的历史执行 / 固定虚构角色鉴权快照；另一次旧录制仍保留。
- `web/src/main.tsx` / `DemoUX.tsx` / `PdfEvidence.tsx` / `style.css`：访客入口、场景、等待状态、审计、引用和响应式阅读。
- `app/security.py` / `trial.py` / `main.py` / `chat_progress.py`：现有会话复用、独立访客、硬额度、只读检查和短期 UI 遥测。
- `scripts/prerender_site.py` / `publish_live_page.sh` / `pages/`：历史数据静态渲染、Pages 发布、静态资源与 API 代理、安全 cookie。
- `web/tests/fixtures.ts` / `global-setup.ts` / `scripts/cleanup_browser_uploads.py`：新建测试上传收据、失败清理与中断后恢复。

## 实际边界

Live API 仍经 Tunnel 依赖本机运行，静态介绍与历史回放可独立访问。访客每天共享 10 次额度（UTC 零点重置），最终验收时剩余 1 次，本轮 11–16 未调用模型或重置额度。

120 秒是客户端停止等待原 HTTP 请求的时间，不代表模型被终止；后端可能继续运行，页面查询结果，已预留额度不退回。遥测适用于当前单进程部署，未来多进程需要共享状态存储。

访客自动路由目前执行固定 Workflow。历史回放、权限快照和历史正确率不能代表当前动态 Agent 性能；Core 70 的正式 Strict Task Success 仍为 pending。

测试中断恢复依赖已收到的上传响应收据，服务器创建后、响应尚未被客户端观察的窗口无法保证覆盖；旧 Markdown 验收资料另有严格双标记兜底。清理保留原件和版本，不删除实验报告。

部署来自当前工作区，`3dffa7d-working-tree` 不是这些改动的新 Git 提交。该工作区还含其他评测和策略开发改动，本轮没有提交或发布整个混合工作区到 GitHub。

## 后续性能改动（2026-10-10）

最新前端部署为 `https://9f8ceb38.agentic-rag.pages.dev`，正式 Live 入口已更新。公开 Demo 使用独立 11438 请求队列、固定模型预热、权限 / 版本感知精确缓存，并提供“新问题”清空本页追问上下文。单轮重复问题实测缓存 HTTP 耗时 0.135s、零模型调用；首次真实生成 57.430s，期间有其他负载，不作为受控加速结论。详细边界和记录见 `docs/demo-performance.md`。

无损 JSON 压缩在完整 Dev47 输入筛选中未达到预登记收益门槛，候选不启用。独立 GPU 部署配置已准备，但缺实际服务器地址或云预算，迁移与速度 / 质量对照仍 pending。没有修改 Core 成绩、冻结数据或历史产物。

9 项发布资源 HTTP 200 且哈希与构建一致；该轮完整 pytest 快照为 1286 passed / 36 skipped，8 项 Node 测试通过。悉尼 06:19 复查恢复本机 API，后台运行原监督脚本；公开状态接口 200，浏览器正常进入只读演示，相关 24 项测试再次通过。当前仍依赖本机和 Tunnel，重启会使缓存重新建立。截图：`/private/tmp/demo-latency-live-20261010.png`。

## 公开页免登录（2026-10-10）

部署 `https://6d573397.agentic-rag.pages.dev`，构建于悉尼 21:00:11。正式 `/live` 打开即自动恢复或建立受限访客会话，移除手动进入和退出登录按钮；公开页面始终不显示账号密码表单。会话过期的 401 仅恢复并重试一次，不自动重发模型失败、额度拒绝或其他错误。首页 English technical overview 入口已移除，旧概览 URL 保留。

实时连接不可用时，当前页直接显示可操作的真实历史回放，并每 30 秒尝试重连。静态 shell / 回放由 Pages 提供，Live HTML 不缓存旧版入口。此轮发现旧 Tunnel 域名 DNS 已失效，已创建替代连接并更新 origin；本机 API 和实验进程未重启。实时推理仍依赖本机，并未完成独立托管。

正式域名已完成无有效会话自动进入、刷新和只读权限核对；仅静态文件预览实际查看了回放答案及引用原文。10 项发布资源返回 200 且哈希一致，18 项 Node 测试、19 项相关 pytest 与构建通过。本轮无模型请求、无人工额度重置，跨 UTC 日期后共享额度为 10/10。完整记录见 `docs/superpowers/plans/2026-10-10-login-free-demo.md` 与 `.runtime/public-demo-captures/login-free-release-20261010T100408Z.json`。
