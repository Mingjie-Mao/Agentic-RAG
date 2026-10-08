# Enterprise-RAG Benchmark Package v1

本包是之后 RAG / Workflow / Dynamic Agent / Hybrid 的统一评测协议。唯一主表来源为 **Core 70 题的 Strict Task Success Rate**。当前完成数据整理和评测基础设施；**Core 70 与 Security 16 的题目标注已由当前 GPT 会话审核，尚未正式冻结、执行或产生系统正确率**。GPT 审核已获用户授权，保存模型家族、具体服务版本未知的说明、理由和独立性，不要求把模型标签伪装成人工金标。External 150 的新协议标注审核尚未完成。

## 当前状态（2026-10-08）

- **Core 70 / Security 16 / External 150：正式运行 0 次**，主表为 pending。External 150 的新协议标注审核未完成。
- **Dev 47：** 已多次冻结运行用于开发。最新 v6（2026-10-07）严格成功 RAG 38、Workflow 38、Dynamic 38、Hybrid 37。2026-10-06 之后的新答案由 Claude Opus 5.5 审核（`independent=false`），不符合本协议规定的 GPT 审核，只作开发诊断；GPT 审核请求包已生成未执行。
- 修复经过、逐轮结果与问题清单统一记录在[项目报告 §20.1 与 §20.26–20.29](../../../PROJECT_REPORT.md#201-现状总览已完成当前问题与计划2026-10-08)。

| Split | 数量 | 用途 | 调参 |
| --- | ---: | --- | --- |
| Dev | 47 | 已用 35 题 + v5 困难 12 题，按规范化题面去重 | 可以 |
| Core Test | 70 | 七类各 10 题，完整保留为最终测试 | 不可以 |
| Security | 16 | 组权限、跨租户、检索后撤权、旧版本范围，各 4 题 | 不可以 |
| External | 150 | 固定 MultiHop-RAG r2 的外部泛化比较 | 不可以 |

四部分没有完全相同的题目。只查了规范化题面精确重合，不表示语义或预训练独立性已获证明。Core 来自原有候选 70 题，保留其原文与题面；原有 `fixtures/unseen_v2` 的 **70 core + 2 safety** 不改写。包内安全集使用独立租户与新文档，不混入 70 题分母。

External 150 题曾经运行，其错误也已被分析。因此，本包重复测量可比较固定外部语料上的表现，不能再称作首次 unseen 验证。将来如果需要新的外部一次验收，应另建有来源的版本，不能悄悄替换本包 150 题。

## 内容与文件

- `manifest.json`：分组、来源哈希、主指标、使用历史及注册方法矩阵。
- `core/tasks.json` / `core/documents.json` / `core/review.json`：Core 标注、原文清单与待审核包。
- `security/`：16 个安全任务、独立文档和待审核包。
- `dev/selection.json` / `external/selection.json`：可保留在仓库的选择、题面及标注哈希。
- `.runtime/benchmark-package/v1/dev`、`.runtime/benchmark-package/v1/external`：本地完整标注和第三方新闻。正文、问题和答案不放入可提交的包文件。
- `ablation-plan.json`：设计说明；实际执行矩阵以 `manifest.json` 和各 suite 的 `method_matrix` 为准。
- `annotation-review-summary.json`、Core/Security 的 `reviewed.json`：2026-10-03 的同会话 GPT 题目标注审核，`independent=false`。这是 gold/协议审核，不是模型答题结果。

上游文件在 `.runtime/multihop`，由现有 `build_multihop_subset.py` 的固定哈希校验；缺缓存时先用该脚本的 `--download` 获取固定数据，再执行 `.venv/bin/python scripts/benchmark_package.py materialize` 重建被忽略的 Dev/External 文件。不会调用生成模型或执行测试题，也不会覆盖已审核或冻结的私有 suite。

## 每题标注

`question`、`gold_answer`、`required_facts`、`gold_documents`、`gold_evidence`、`task_type`、`requires_version`、`expected_behavior`、`answerable` 均已保存。条件真值、适用版本、禁止输出和安全事件另有显式字段。

金标证据使用 **文档 ID + 原文字节哈希 + 生效时间 + 字符范围 + 引文**。先保留稳定原文定位，不编造尚未入库的 chunk ID；入库后的冻结快照绑定真实 version/chunk ID、解析文字、实际搜索文本/向量哈希与 ACL。当前原文证据范围偏粗，且 Core 有十组重复任务结构；本次 GPT 审核确认受控任务的事实与协议有效性，不认证真实企业分布的代表性。已去掉原本只给 Agent 的内部金标 document_id 提示。

历史版本题分别要求旧版本、新版本和变化结论；比较题注明差值计算，不能只命中一个数字就得分。`expected_behavior.strict_scoring=false`，期望工具和顺序仅用于失败分析。一次检索同时获得充分证据可以得满分。

## 三层评分

**Retrieval**：累计 Gold Document Recall；第一次检索的 distinct-document Recall@1/5/10；依据实际取回证据由 GPT 审核的 Gold Fact Recall / Evidence Coverage。未记录证据或未审核事实时记为 `null`，不能用答案引用反推已经检索到全部证据。此处 K 是不同文档数量，不能和 chunk Recall@K 混称。

**Agent Policy**：steps、tool calls、失败与非法调用、重复调用/查询、空或失败查询、过早/过晚停止、证据充分后的无效调用、失败检索后的恢复、耗时，以及 generation/policy/judge 的实测 tokens（不包含尚未记录的 embedding tokens）。语义停止诊断需要检查 trace 和事实覆盖，不凭“找到 gold 文档”认定证据充分。没有测量的字段保留 `null`；合法执行路径不影响严格正确率。

**Final Answer**：一题仅当所有 required facts 得到支持、没有关键错误、条件及版本正确、每条 claim/citation 得到支持、应拒答时拒答、执行安全且规定安全事件实际触发，才记 Strict Task Success=1；否则为 0。执行失败进入完整分母，不能删去失败题提高成绩。授权变化场景允许规定的拒答/授权终止，意外崩溃不通过。ACL canary 检查全部用户可见输出；已获授权的历史内容不因工具读取就算 ACL 泄漏，版本范围违约检查最终断言和引用。

同时报告事实召回、Fact Precision、citation correctness、faithfulness、完整性和七类拆分。Fact Precision 需要 GPT 将答案全部原子事实拆成 `answer_facts`，逐项绑定真实 claim 的下标并记录支持判断，明确 `atomic_facts_complete=true`；未做原子事实审核时 precision 保留 `null`。Faithfulness 按 claim 条数统计，一条长 claim 中任何重要错误使整条不 supported。事实召回按 required fact 个数计算。组件指标为适用题目的宏平均，均附适用分母；没有事实或引用的合法拒答题不作为 100% 计入这些平均值。

答案审核包隐藏方法名称，不把执行路径作为答案正确依据。模型审核填写每个 fact/claim/citation 的语义判断和五个严格布尔检查；不得只打 1–5 分。失败分析另看原始 tool trace，保持模型来源与缺测信息。模型评测是模型评测，不宣称独立人工准确率。

## 公平条件与消融

四种主方法共享语料、ACL、生成模型、embedding、检索策略、规则改写、证据契约、重排状态、context budget、最大任务时间、模型调用预算和最大 steps。任务内的 `max_steps` 对 Agent 相同；Plain RAG 的单轮结构是其方法差异。所有有效配置及模型 digest 在冻结时记录，按题轮换执行顺序，失败成本也计入。

当前已在运行前注册 8 路矩阵，四路主方法与四路 RAG 消融。Core 执行前会冻结全部矩阵，不能看到测试结果后增加配置：

| 比较 | 单一变化 |
| --- | --- |
| `rag_bm25` → `rag_hybrid` | BM25 → Hybrid Retrieval |
| `rag_hybrid` → `rag_rewrite` | 开启规则 Query Rewrite |
| `rag_rewrite` → `rag` | 开启共享任务/证据契约 |
| `rag` → `rag_rerank` | 开启 passage reranker |
| `rag` / `workflow` / `dynamic` / `hybrid` | 检索与契约相同，比较策略/编排 |

这些配置没有“语义 Evidence Coverage 已有效”的含义。任务契约消融与 slot semantic control 是不同组件；后者继续默认关闭，需先在 Dev 校准。本文没有示例百分比代替测量。主表只列四种主方法，其他行列在同一 Core 的消融表中。配对差值与 bootstrap 区间会保存；Core 按十个实体任务家族聚类抽样，不能把同一模板的七类题视作七个独立样本。统计不显著时不宣称提升。

## 使用流程

先验证本地数据，不执行题目：

```bash
.venv/bin/python scripts/benchmark_package.py validate
```

只用 Dev 调参与预检查。现有入库程序支持本包多个租户和文档 owner；新闻原文留在 `.runtime`：

```bash
.venv/bin/python scripts/setup_unseen_benchmark.py --suite-dir .runtime/benchmark-package/v1/dev
.venv/bin/python scripts/run_unseen_benchmark.py --suite-dir .runtime/benchmark-package/v1/dev --freeze
.venv/bin/python scripts/run_unseen_benchmark.py --suite-dir .runtime/benchmark-package/v1/dev --output .runtime/benchmark-results/dev-v1.json
```

Dev 下一轮可以 `--freeze --refresh-dev-freeze`，使用新的结果文件。该参数不能用于 Core/Security/External。

Dev 确定方案后，核验各测试 split 的 `review.json` 与 `reviewed.json`，给每项保存 named reviewer、model（未知部署版本须明示）、`independent`、理由和 required checks。Core/Security 已完成此次标注审核，External 仍待审核。未通过的任务先修订并重新审核/生成版本；不可带着拒绝或缺项冻结。修改题目、标注或原文会使旧审核失效。

以 Core 为例，审核通过后入库，再冻结真实环境并只运行完整矩阵一次：

```bash
.venv/bin/python scripts/setup_unseen_benchmark.py --suite-dir benchmarks/enterprise_rag/v1/core
.venv/bin/python scripts/run_unseen_benchmark.py --suite-dir benchmarks/enterprise_rag/v1/core --freeze
.venv/bin/python scripts/run_unseen_benchmark.py --suite-dir benchmarks/enterprise_rag/v1/core --output .runtime/benchmark-results/core-v1.json
```

Security、External 使用 manifest 中的各自 suite 路径及独立结果文件。第一次启动在原 suite 旁保存唯一运行登记。换 `--output` 不能绕过；中断仅允许同一冻结状态/输出的 `--resume`，已有完成结果不重做。进程中断且无法确认结果的尝试记录为失败或缺失遥测，不重抽答案。

运行后输出原始结果和匿名 `.answer-review.json`。GPT 完成逐事实/引用审核后生成最终报告：

```bash
.venv/bin/python scripts/benchmark_package.py report --raw .runtime/benchmark-results/core-v1.json --review .runtime/benchmark-results/core-v1.answer-reviewed.json --output .runtime/benchmark-results/core-v1.report.json
```

审核前 strict accuracy 是 `null`。即使有答案审核，没有可验证的冻结登记也不能作为 README 主成绩。Core 主表、分类表、消融表、安全失败与 External 泛化分开报告；绝不平均成一个项目总 accuracy。

协议参考：[CRAG](https://arxiv.org/abs/2406.04744) 对问题类型和可靠性进行评估；[Ragas](https://arxiv.org/abs/2309.15217) 区分检索、证据利用和生成质量。本包使用项目特有的事实、版本、ACL 和审计规则，不直接沿用论文分数，二者成绩不可横比。

## 附属题集：观测依赖（`dynamic/`）

Dev 47 大多一次检索就够，无法判断 Agent 能否“用中间结果决定下一步”。`dynamic/` 是由 `scripts/build_dynamic_benchmark.py` 从结构化虚构企业世界生成的附属题集，标准答案由数据直接算出：

| Split | 租户 | 题数 / 文档数 | 状态 |
| --- | --- | --- | --- |
| `dynamic/dev` | 青禾智造 | 30 / 63 | 已冻结，允许调参；四种主方法 4–7/30，Planner 14/30（Claude 审核，非独立） |
| `dynamic/test` | 远帆数科 | 40 / 85 | 待 GPT 任务审核，未入库、未运行 |

题型包括桥接、深链桥接、俗称改写、条件分支、数量未知的展开、事件锚定版本、冲突追查、对照题与不可回答题。“是否需要动态”按实测标注：用原问题做一次检索，漏掉任一金标文档即标为需要（`artifacts/dynamic-benchmark-dev-necessity.json`）。Dev 与 Test 实体不重叠，但由同一套模板生成，Dev 上调出的成绩偏乐观。该题集不属于 `manifest.json` 的四个 split，也不进入 Core 主表；manifest 已被历次冻结运行哈希绑定，因此不修改。
