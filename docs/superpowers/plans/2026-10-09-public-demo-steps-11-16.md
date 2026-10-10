# 公开演示 11–16 Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [x]`) syntax for tracking. 按用户要求在当前会话执行，沿用 writing-plans 的计划和验收文件方式。

**Goal:** 完成权限对照、独立静态回放、正确评测展示、移动端阅读、最终回归及交付说明。

**Architecture:** 固定虚构角色的权限快照展示实际鉴权，不授权访客切换身份。首页与回放独立部署在 Pages，模型与 API 仍由现有后端提供。主评测遵守冻结 Core 协议；本轮只验收展示与产品边界。

**Tech Stack:** HTML/CSS、React/TypeScript、Cloudflare Pages、FastAPI、SQLAlchemy、pytest、Node test、浏览器手动验收。

---

## Task 11：权限对照

**Files:** `site/permission-demo-20261009.json`、`site/replay.html`、`web/src/main.tsx`、`tests/test_demo_entry.py`。

- [x] 对照数据库当前 can_read 与固定角色快照，核对同文档支持组 / 工程组允许和拒绝。
- [x] 公开访客只可查看明确标注的历史快照；API 仍检查当前用户权限，不能指定角色取得新权限。
- [x] 浏览器检查权限表与 Live 中的入口；如快照变化，另存新时间戳文件，不覆盖旧记录。

## Task 12：离线备用

**Files:** `site/replay.html`、`site/index.html`、`pages/proxy.js`、`web/src/main.tsx`。

- [x] 在只有静态资源、无 API 的本地预览上操作真实回放、引用和权限表。
- [x] 核对历史 / 实时标注、录制时间、模型身份、失败提示及返回路径。
- [x] 验证 API 不可用时公开入口给出明确说明和回放入口，不让访问者一直等待登录。

## Task 13：评测口径与静态数据

**Files:** `site/index.html`、`site/data.json`、`scripts/prerender_site.py`、`tests/test_public_site.py`。

- [x] 核对 Core 70 的主指标为 pending，历史 80 / 147 / 39 题分别标明口径与时期。
- [x] 核对预渲染表格与原 JSON 一致，页面显示不依赖运行时 data.json 请求。
- [x] 将工程取舍和已知局限保留为可展开详情；不运行、调参或覆盖评测产物。

## Task 14：字体、对比度与移动端

**Files:** `web/src/style.css`、`site/index.html`、`site/replay.html`。

- [x] 浏览器检查桌面与 375px 页面，标题、导航、输入、证据开关和回放操作均能使用。
- [x] 检查可见文字尺寸和淡色辅助文字；修复实际发现的低对比度或溢出。
- [x] 核对长文、表格、菜单、错误提示和键盘焦点，重置临时 viewport。

## Task 15：最终回归

**Files:** `tests/`、`web/tests/demo-ux.test.mjs`、`pages/proxy.test.mjs`、`scripts/publish_live_page.sh`。

- [x] 运行完整 pytest；依赖本机服务的项目如跳过，单独记录，不计为通过。
- [x] 运行 Node 组件 / 代理测试、前端构建、测试 fixture 类型检查、ruff 和 diff 检查。
- [x] 公开访客会话、页面路径、静态资源哈希、引用与资料残留最终核验；复用已有真实回答，不再消耗模型额度。

## Task 16：交付

**Files:** `docs/public-demo-ux.md`、本计划文件、`docs/public-demo-delivery.md`。

- [x] 汇总 1–16 的完成记录、发布入口、检查结果和实际限制。
- [x] 保存桌面 / 移动端截图与本地验收记录，保留历史文件与其他开发改动。
- [x] 发布本轮修正后核对线上版本，向用户汇报；不自动提交整个混合工作区。

## 最终验收记录（2026-10-09，悉尼时间）

| 步骤 | 结果 |
| --- | --- |
| 11 权限对照 | 当前 can_read 与两份固定虚构文档快照一致。公开访客允许资料 GET 为 200，无权资料和未知资料均为 404；提交 admin / engineering / 其他租户字段不能覆盖服务端 member 与支持组身份。没有增加角色切换权限。 |
| 12 离线备用 | 单独启动只有静态文件的预览，API health 为 404，回放为 200；实际操作最终答案、引用和权限表仍成功。无 API 的 Live 入口明确显示服务不可用并提供回放链接；没有停止真实线上服务。 |
| 13 评测展示 | Core 70 维持正式 Strict Task Success pending；历史 80 / 147 / 39 题分开标注。表格预渲染，新测试证明数值 / 安全统计 / 引用校验空值显示 pending，不冒充 0、通过或失败。原 site/data.json 与历史评测产物未修改。 |
| 14 可读性与移动端 | 在 375px 手机视口核验首页菜单、Agent、引用和权限表，没有整页横向溢出（首页带滚动条时 clientWidth 为 360px）。引用侧栏 311px，支持原句高亮；关闭后焦点返回原引用按钮。辅助文字颜色加深、资料标题增至 14px；保留深色登录故事的原色。 |
| 15 最终回归 | 最新完整 pytest 1164 passed / 36 skipped / 1 依赖弃用警告；8 项 Node 测试通过，应用与测试 fixture 类型检查、构建、ruff、文档一致性和 diff 检查通过。验收期间新测试曾出现 12 项共享 fixture 缺失，当前显式引用后联合运行 200 项通过，全量回归无错误。没有运行 Core。复用真实历史回答和已验证的停止等待结果，本轮未调用模型。 |
| 16 交付 | 总交付说明为 docs/public-demo-delivery.md；三个分段计划均已完成。最终 Pages 部署、资源哈希、截图和本地验收数据已保存；其他开发改动和所有历史记录保留，未提交混合工作区。 |

最终部署 `https://c4330829.agentic-rag.pages.dev`，构建时间 `2026-10-09T09:37:16Z`（悉尼 20:37:16）。公开 12 项 HTML / JS / CSS / JSON / PDF worker / CMap / 字体资源全部 HTTP 200，SHA-256 与构建一致。最终线上访客仍为 member / support，Secure、HttpOnly、SameSite=Strict 均保留；进入演示增加 15 秒超时提示。

最终本地验收记录 `.runtime/public-demo-captures/website-steps-11-16-20261009T094846Z.json`，先前 `20261009T092704Z` 的记录保留。手机截图 `/private/tmp/agentic-rag-mobile-evidence-final.png`、`/private/tmp/agentic-rag-mobile-permissions-final.png`；离线入口 `/private/tmp/agentic-rag-offline-entry-final.png`；最终首页 `/private/tmp/agentic-rag-final-home.png`。

实际边界：Live API 仍依赖本机和 Tunnel，静态首页和回放独立部署。访客最终剩余 1/10，本轮没有重置额度。Core 主成绩仍 pending；固定 Workflow 回放不能证明动态策略提升。120 秒自动停止的是客户端原请求，后端可能继续执行。测试中断收据无法覆盖响应尚未被客户端观察就被杀死的窗口。
