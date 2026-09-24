# P2 打分器校准：预注册（2026-09-24，在 v3 gold 冻结与任何打分结果之前写定）

目标：从 4 个语义打分器里，选出**最能复现 v3 gold 判据的低成本打分器**。选出后只进 shadow
（记录、不拦截），与 `app/qa.py::shadow_scores` 的现状一致；从 shadow 升级为拦截需要另一次冻结评估。

顺序不可调换：校准判据（CODEBOOK v3 + 最终审核）→ 冻结 gold → 比较打分器 → 选择。
**本文件在 gold 冻结之前提交到磁盘，此后不改**；如需改，写成 v2 并说明是在看到哪些结果之后改的。

## 1. 数据

- gold：`gold-v3-claims.csv`，由 `scripts/build_semantic_gold_v3.py --freeze`（`make semantic-freeze`）生成并冻结，冻结清单
  `gold-v3-manifest.json` 记录文件 sha256。评估脚本在哈希不符时拒绝运行。
- 只用 provenance 属于 gold 档的行；`model_consensus`（两位标注者一致但未经最终审核）不进评估。
- 权重：第 2 轮「一致且非 unsupported」的 claim 按 20% 抽样进入审核，每行权重 = 该层条目数 / 抽中数；
  其余 gold 行权重 1。**加权指标为主**，不加权作敏感性分析——两者结论不一致时如实报告。

## 2. 划分

按 `sha256("p2-split|" + item_id)` 模 10：`< 4` 为 dev，其余为 test。同一道题的全部 claim 落在同一侧。
连续分数的阈值**只在 dev 上定**；test 只报告，不回头调任何东西（阈值、prompt、证据截断长度）。

## 3. 打分器（按成本从低到高）

| 名称 | 输入 | 做法 |
| --- | --- | --- |
| `rule` | claim、引用证据 | claim 的内容词（英文小写去停用词、数字；中文字符二元组）在证据中的覆盖率 |
| `embedding` | 同上 | bge-m3（Ollama）；claim 与每个证据块的余弦，取最大 |
| `cross_encoder` | 同上 | bge-reranker-v2-m3；(claim, 证据块) 分数取最大。**它是相关性模型，不是 NLI 模型**，结果照此解读 |
| `llm_judge` | 同上 | qwen2.5:7b-instruct，温度 0、seed 42；只给 codebook 规则摘要（不含任何条目）；直接输出三类标签 |

`llm_judge` 的 prompt 一次写定，不做迭代；若要改，只能看 dev 结果，改完 test 重新作为一次独立报告。

## 4. 任务与指标

**主任务**：二分类 `supported` 对 `not_supported`（`partial` + `unsupported`）。P2 要测的是「把没有证据的
说法放行」这一种错误，`partial` 放行同样是放行了一部分无据断言。

在 test 上报告（全部带按题目重抽的 bootstrap 95% CI，2000 次，seed 0）：

1. **放行率**：gold 为 `unsupported` 的 claim 中被判 `supported` 的加权比例（越低越好）——关键指标；
2. **均衡准确率**：supported 召回率与 not_supported 召回率的平均——主排序指标；
3. AUC（仅连续打分器，与阈值无关）；
4. Cohen's kappa（二分类）；`llm_judge` 另报三分类混淆矩阵与 kappa。

阈值：连续打分器在 dev 上取使均衡准确率最高的阈值（并列取较高者，更保守）。

**次任务**：relevance（`answers` 对其余），同样 4 个打分器，输入为 (question, claim)，只报 AUC / 均衡准确率，
不参与选择。

## 5. 选择规则

**`llm_judge` 只报告，不参与基准与选择。** 它与 gold 的最终审核人同为 LLM，§14A.1 已定这一臂
「只作参考，不作排序依据」；它的数字用来回答「本地小模型离 gold 标准差多远」，不用来挑选。
以下规则只作用于 `rule` / `embedding` / `cross_encoder`。

1. 若某打分器 test 均衡准确率的 CI 下界 ≤ 0.5，视为不优于随机，淘汰。
2. 剩余者中取均衡准确率点估计最高者为基准 B。
3. 满足「均衡准确率 CI 上界 ≥ B 的点估计」且「放行率 ≤ B 的放行率 + 0.10」的打分器里，
   选**每条 claim 中位耗时最低**的一个。
4. 若第 1 步后没有剩余，结论是「没有打分器能复现 v3 判据」，不选、不上 shadow 之外的任何位置。

## 6. 已知局限（写进报告，不因结果删改）

- gold 的最终审核人是 GPT，人工逐条复核为 0（负责人 2026-09-23 决定的项目标准）。
- `llm_judge` 与 gold 标注者同为 LLM，可能共享偏差；它与 gold 的一致可能高估其对人工判据的复现程度。
- 第 2 轮的审核是分层抽样，类别比例与自然分布不同，因此以加权指标为主。
- 样本量约百余条 claim；CI 会很宽，「不显著差于」不等于「一样好」。
