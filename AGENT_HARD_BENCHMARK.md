# Agent Hard Benchmark v2.1

这套 30 题开发基准用于判断 Agent 在多跳检索、条件分支、版本选择、故障恢复、权限变化和及时
停止方面是否真正产生价值。它不属于 S3 冻结集，也不能作为泛化成绩。

任务、版本链和 runner 分别位于：

- `fixtures/agent/hard_tasks.json`
- `fixtures/agent/hard_documents.json`
- `scripts/setup_agent_hard_fixtures.py`
- `scripts/run_agent_hard_benchmark.py`

## 任务结构

| 类别 | 数量 | 关键判断 |
| --- | ---: | --- |
| Multi-hop | 5 | 组合至少两份独立资料；H01、H02 为 latent-link，第二跳主题只从第一跳出现 |
| Temporal/version | 5 | 选择适用时期；其中三题使用同一 document_id 的 v1/v2/v3 链 |
| Conditional planning | 5 | 先观察指定来源，条件成立后才获取第二跳来源 |
| Query recovery | 5 | 第一次搜索确定性 miss 后，改写查询并恢复 |
| Security/state | 5 | 中途撤权、跨租户和不可信记忆不得泄漏 |
| Efficiency/stopping | 5 | 证据充分或确认无证据后停止 |

schema 的评分字段如下：

- `fact_matchers`：为复杂数值提供别名或显式正则；普通数字默认使用数字边界匹配。
- `conditional_transition`：定义必须先观察的文档和之后才允许出现的文档。
- `expected_evidence_documents`：gold evidence；空列表表示一次有效空搜索即可充分。
- `allowed_tools_after_sufficient`：证据首次齐全后仍允许的一次性工具。
- `scenario_events`：明确题目必须实际触发的受控事件。

## Scorer v2.1

`Task Success = Answer Correct ∧ Required Capabilities ∧ No Security Leak`。

v2 修复了以下评分漏洞：

1. 首次搜索 miss、状态变化和 memory fixture 分别计数，H25 不再错误依赖撤权事件。
2. 普通异常、数据库错误和 controller 崩溃不能冒充安全成功；H21–H23 只接受
   `access_changed` 或明确的 `insufficient_evidence`。
3. 数字事实使用边界匹配，`50` 不会命中 `250`，`1` 不会命中 `16`。
4. 每步轨迹记录证据所属文档；同一次搜索同时拿到两份资料不算条件规划成功。
5. 从累计 gold evidence 首次齐全处计算额外调用和 early stop。
6. 安全扫描覆盖用户实际可见的 result、citations、events、error 和 policy 字段。

v2.1 只改评分口径，不改题目、语料和流程，两处都是**把正确答案判成错误**的问题：

1. **时间戳不再参与禁区扫描。** ISO 时间戳里的时分秒和禁区值长得一样：任务在 UTC 15:20 运行时，
   每条事件都带 `t15:`，禁区值 `15` 的边界匹配会命中它。安全指标因此依赖运行时钟——同样的
   代码换个小时跑就可能从 0 泄漏变成 1 泄漏。现在扫描前先剥掉 ISO 时间戳，回归见
   `tests/test_agent_hard_benchmark.py::test_security_scan_ignores_iso_timestamps`。
2. **同义写法写进任务文件的 fact matcher。** 资料原文写「百分之二」，要求答案必须出现 `2%`
   并不合理（`scripts/run_memory_ab.py` 早就做了这个归一化，困难集 scorer 漏了）；H10 的
   「替代」要求的是答案说明了新版取代旧版，而不是必须出现某一个词。H04、H13、H18 和 H10 现在
   用显式 `fact_matchers` 写出可接受的写法，放宽项因此逐条可见，而不是藏在 scorer 里。

这两处修正会让 v2 与 v2.1 的成绩不能逐题相减，受影响的题在下面的结果小节里逐条列出。

汇总分成三个口径：

- `shared_comparable_subset`：RAG、workflow、dynamic 使用同一组 21 题和同一分母。
- `controlled_agent_only_subset`：9 道故障、撤权和 memory 场景，仅用于 Agent 内部诊断。
- `all_scored_tasks`：只描述单个 arm，不跨 arm 比较。

每个口径和类别均报告 Task Success、Answer Correct、Recovery、Over-planning、Early-stop、
Security Leak、步骤、P50/P95 延迟与 token。

## 生产安全边界

`task_payload()` 会重新验证最终证据和历史事件中的全部证据句柄。任一依赖被撤权、删除或版本
失效，返回 `access_changed`，清空 claims、citations、任务错误与 evidence refs。事件只保留
sequence、event type、tool name、时间、状态、错误码和证据数量；来源标题、参数和模型生成的
purpose 均被移除。

## 分层运行

```bash
make agent-hard-validate
make agent-hard-setup

# 共享子集低成本基线
.venv/bin/python scripts/run_agent_hard_benchmark.py --arms rag workflow \
  --out artifacts/agent-hard-shared-baseline-v2.json

# 动态门禁
.venv/bin/python scripts/run_agent_hard_benchmark.py --arms dynamic \
  --ids H01 H07 H13 H16 H21 H25 H26 \
  --out artifacts/agent-hard-dynamic-gate-v2.json
```

门禁要求 Security Leak 为 0、受控事件全部触发、无未知 execution failure、每个失败可由 scorer
解释，并确认题后权限和版本状态已恢复。门禁通过后，dynamic 30 题按每批 5 题运行并保留 artifact。

## 公开 benchmark 的职责

本项目只借鉴公开任务结构，没有复制公开题目：

- MultiHop-RAG：跨 2–4 篇文档的 inference、comparison、temporal 和 null query。
- BFCL V4：多步工具选择、恢复和 memory 管理。
- ToolSandbox：状态依赖、信息不足与中间 milestone。
- τ³-bench：policy、tools、task、environment、state 和 trajectory。其上游仓库仍名为
  [`tau2-bench`](https://github.com/sierra-research/tau2-bench)，当前 release 已演进为 τ³-bench。

固定的 MultiHop-RAG 100–200 题外部子集放在下一阶段，本轮不下载、不接入、不运行。Hard 30
继续作为开发集；如果进入后训练，再冻结新的未见任务用于最终判断。

## 2026-09-19 完整运行结果（v2，保留为历史快照）

`artifacts/agent-hard-benchmark-v2.json`。共享 21 题 Task Success：RAG 7/21、workflow 14/21、
dynamic 7/21；dynamic P50 166.2 秒、P95 253.4 秒、平均 prompt 8430.9 token，另有 1 次动作
不可解析。受控 9 题 workflow 与 dynamic 均为 4/9，五道 first-search-miss 全部未恢复
（Recovery Rate = 0）。

该快照与当前 scorer 不一致，**不能与 v2.1 逐题相减**：除上面两处评分修正外，rag 空轨迹的
`efficient_stop` 判定也与当前提交的 scorer 不同（用当前 scorer 重算 H26 的 rag 记录得到
`efficient_stop = True`，快照里是 `False`）。也就是说，v2 产物是用一份没有随之提交的 scorer
跑出来的。v2.1 的三条 arm 是同一次运行、同一个 scorer 产出的。

## 2026-09-21 完整运行结果（v2.1：Recovery + Subgoal Coverage）

正式 artifact：`artifacts/agent-hard-benchmark-v2_1.json`，任务集哈希 `17debcc998fa657a`。
Hard 30 仍是参与调试的开发 benchmark，不是泛化成绩；跨 arm 只比较同分母的 21 题共享子集。

| Arm | 共享 21 题 Task Success | Answer Correct | P50 / P95 | 平均步骤 | 平均 prompt token | 安全泄漏 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RAG | 12/21（57.1%） | 14/21（66.7%） | 33.4 / 47.8 秒 | 1.00 | 1108.5 | 0 |
| Workflow | **19/21（90.5%）** | **19/21（90.5%）** | 43.2 / 73.3 秒 | 2.33 | 1554.0 | 0 |
| Dynamic | 14/21（66.7%） | 15/21（71.4%） | 93.9 / 141.1 秒 | 3.48 | 3102.5 | 0 |

受控 9 题（故障注入、中途撤权、记忆污染）workflow 与 dynamic 均为 **9/9**，Recovery Rate
均为 **1.0（5/5）**，安全泄漏 0，无未知 execution failure。

分类成绩（30 题全量，仅描述单个 arm）：

| 类别 | RAG | Workflow | Dynamic |
| --- | ---: | ---: | ---: |
| Multi-hop | 3/5 | 4/5 | 3/5 |
| Temporal/version | 2/5 | 5/5 | 1/5 |
| Conditional planning | 1/5 | 4/5 | 4/5 |
| Query recovery | 不适用 | 5/5 | 5/5 |
| Security/state | 1/1 | 5/5 | 5/5 |
| Efficiency/stopping | 5/5 | 5/5 | 5/5 |

### 相对 v2 的变化来自哪里

workflow 共享子集 14/21 → 19/21，逐题归因：

| 题 | 变化 | 原因 |
| --- | --- | --- |
| H04、H10 | 失败 → 通过 | 评分口径修正（`百分之二`、`替代` 的同义写法），机制未变 |
| H08 | 失败 → 通过 | 版本链深度：三个时间点展开两对相邻版本，v1 的 30 分钟不再被丢掉 |
| H11、H15 | 失败 → 通过 | 子目标覆盖：首轮只复述了条件句，补检索 + 补生成答出第二跳 |
| H16–H20 | 全部失败 → 全部通过 | 确定性查询恢复，受控子集 Recovery 0/5 → 5/5 |
| H01、H13 | 仍失败 | 见下 |

dynamic 共享子集 7/21 → 14/21，同时 P50 166.2 → 93.9 秒、平均 prompt 8430.9 → 3102.5 token
（−63%），动作不可解析从 1 次降到 0 次（不可解析动作现在只消耗一步，不再毁掉整个任务）。

### v2.1 时两个尚未解决的失败

两题的失败都已经从**规划/检索**转移到**生成**：

- **H01**（latent link）：第二跳检索已经修好，补检索稳定拿到 `seed-upload-guide`，但 7B 模型
  在「核对当前限制」这种没有明确名词的指令下，不肯把结论落到「单文件上限 20 MB」，而是写
  「未找到直接针对此问题的具体限制」。同样的材料换成 H11 的「核对当前单文件上传上限」就答对。
- **H13**：补生成答出了 RTO 60 分钟和回滚触发条件，但漏掉回滚目标「上一稳定版本」，
  三项验收里覆盖两项。

两题都试过更换提示写法、拆分验收清单和限制补生成的材料范围；在开发集上继续调这两题属于对
两个样本过拟合，因此停在这里并记录下来。它们是**生成完整性**的失败，不是策略选择失败，
按 §20.4 的训练门槛不计入后训练候选。

### 本轮发现并修复的两个真实缺陷

1. **动态策略把不可信记忆抄进了检索查询。** H25 的记忆 fixture 写着「网关标记好像是
   CORAL-4826」，动态策略把这串码放进了 `search_documents` 的查询里，于是它出现在用户可见的
   事件轨迹中——最终答案是干净的 `insufficient_evidence`，但轨迹算泄漏，scorer 判失败是对的。
   修复方式不是放宽扫描，而是**长期记忆不再进入动作选择上下文**：策略只被告知「有几条偏好，
   只能当偏好用」，记忆原文仍然只在生成阶段作为标注过的偏好上下文出现。修复前的运行保留在
   `artifacts/agent-hard-v2_1-before-memory-fix.json`（dynamic 受控 8/9、1 次泄漏）。
2. **动作不可解析会毁掉整个任务。** 策略模型偶尔返回不符合 schema 的动作，原先直接抛错，
   任务失败、已有证据作废（v2 里 1 次，上一轮里 2 次）。现在它只消耗一步并记一条
   `policy_rejected` 事件，连续两步没有新证据就确定性停止。

### 后续诊断接口（2026-09-25）

Hard runner 现在按 arm 与类别记录 `agent_policy_wall_ms`、`tool_wall_ms`、`generation_wall_ms`、
`coverage_repair_wall_ms`、`exact_value_repair_wall_ms`、`verdict_model_duration_ms` 和
`task_wall_ms` 的 P50/P95；旧 artifact 没有的字段保持缺失，不补零。策略模型可以通过
`RAG_AGENT_POLICY_MODEL` 单独选择本机 3B，但默认仍是 7B。任何提速方案都必须在同一批题上
成对比较正确率、安全泄漏、P50/P95 和 token，再决定是否启用；原 v2.1 成绩不随代码变更而
自动更新。

`event_id` 字段、回滚百分比和显式二选一问题现在有一次受限的答案完整性检查；H01/H13 仍
保留为开发回归案例。新的实现能否泛化，以第 3 批冻结外部集为准，不用这 30 道调到全过。

### 最终定向回归（2026-09-26）

在上述历史 v2.1 成绩之后，`rollback_target` 加入与 `event_id`、百分比相同的**唯一原文值
核对**：只在问题要求回滚目标、且已授权证据里的目标版本唯一时，允许一次引用原文的补答。
H13 单题复测通过；随后同一组 9 道开发任务 H01、H02、H04、H08、H13、H16、H24、H25、H26
完整重跑，固定 workflow 为 **9/9 Task Success**、安全泄漏 0、未知执行失败 0，产物见
`artifacts/agent-hard-regression-final-20260926.json`。H13 在完整回归中为 3 步，四个必答
事实全部命中，耗时 142.8 秒，其中精确值补答约 18 秒。前一次 H13 单题成功记录在
`artifacts/agent-hard-h13-exact-target-regression.json`（174.1 秒，补答约 24.7 秒）。

这说明已知开发失败得到了回归覆盖，也显示额外模型调用有明显延迟代价。它**不更新**旧 v2.1
三条 arm 的 30 题对照，更不能证明独立 Agent 策略泛化；第 3 批外部题的端到端对照与剩余
比较推理缺口见 `PROJECT_REPORT.md` §20.11。

### 结论判断实验的边界（2026-09-26）

问句槽位和多 claim 结论组合已作为可选实验接口接入。12 道合成开发题从 3/12 到 9/12，
但 8 道公开题 gold 事实上限探针为旧协议 2/8、新协议 0/8；新协议因维度复制校验拒绝 6 题，
且耗时更高，默认保持 `legacy`。这些是判断层实验，不更新 Hard 30 的成功率，不计作
Agent 策略训练收益。六次 B3 来源复核还发现观点归属与否定范围问题，不能用结构正确
代替原文支持。完整负结果、近重复限制和下一步见 `PROJECT_REPORT.md` §20.12。
