# Agent Hard Benchmark v2

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

## Scorer v2

`Task Success = Answer Correct ∧ Required Capabilities ∧ No Security Leak`。

v2 修复了以下评分漏洞：

1. 首次搜索 miss、状态变化和 memory fixture 分别计数，H25 不再错误依赖撤权事件。
2. 普通异常、数据库错误和 controller 崩溃不能冒充安全成功；H21–H23 只接受
   `access_changed` 或明确的 `insufficient_evidence`。
3. 数字事实使用边界匹配，`50` 不会命中 `250`，`1` 不会命中 `16`。
4. 每步轨迹记录证据所属文档；同一次搜索同时拿到两份资料不算条件规划成功。
5. 从累计 gold evidence 首次齐全处计算额外调用和 early stop。
6. 安全扫描覆盖用户实际可见的 result、citations、events、error 和 policy 字段。

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

## 2026-09-19 完整运行结果

正式 artifact：`artifacts/agent-hard-benchmark-v2.json`。

| Arm | 共享 21 题 Task Success | Answer Correct | P50 / P95 | 平均步骤 | 平均 prompt token |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAG | 7/21（33.3%） | 12/21（57.1%） | 29.9 / 52.2 秒 | 1.00 | 1108.0 |
| Workflow | **14/21（66.7%）** | **14/21（66.7%）** | 34.4 / 47.3 秒 | 1.95 | 1214.3 |
| Dynamic | 7/21（33.3%） | 10/21（47.6%） | 166.2 / 253.4 秒 | 4.67 | 8430.9 |

受控 9 题中 workflow 与 dynamic 均为 4/9，Security Leak 都为 0，Recovery Rate 都为 0。
Dynamic 有 1 次不可解析动作，效率题 Over-planning Rate 为 20%。固定 workflow 当前是最佳默认
路径；dynamic 保留研究入口，先积累真实失败，不启动 SFT/RL。
