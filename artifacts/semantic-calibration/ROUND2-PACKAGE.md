# 第 2 轮：剩余全量标注包

第 1 轮标了前缀 50 claim / 30 answer，一致率 entailment 0.90、completeness 0.967、
verdict_reading 0.933、relevance 0.74，17 条分歧归并成 4 条规则决定。结论是
**可用金标只有 18 条 claim + 2 条 answer，其中 `unsupported` 仅 6 条**，不足以估计 P2 最关键的
「unsupported 被判成 supported」的比例。因此这一轮把剩余部分一次标完。

## 文件

| 文件 | 行数 | 给谁 |
| --- | ---: | --- |
| `claims-to-label-rest126.csv` | 126 | GPT 与本项目模型**各自独立**标注 |
| `answers-to-label-rest73.csv` | 73 | 同上 |
| `consensus-to-confirm-claims.csv` | 32 | 人工快速确认（accept / override） |
| `consensus-to-confirm-answers.csv` | 28 | 同上 |
| `CODEBOOK.md` | — | 判据，**仍是 v2，不变** |

两个 rest 文件是冻结文件 `claims-to-label.csv` / `answers-to-label.csv` 去掉前缀后的剩余部分
（claim 第 51–176 行、answer 第 31–103 行），顺序不变，因此仍然是分层轮转的平衡样本。

## 给 GPT 的硬要求（与第 1 轮相同）

不要给它看：`machine-prepass-*`、`gpt-prepass-*`（第 1 轮它自己的答案）、`gold-*`、
`agreement-*`、`to-adjudicate-*`、系统输出的结论字段、数据集金标。
**判据仍用 CODEBOOK v2**，不要给它第 1 轮的裁决说明——那会把裁决结果变成提示。

提示词沿用 `GPT-LABELING-PACKAGE.md` 里的原文，只把文件名换成 rest 版本。
文件较大（209 KB / 40 KB），**以附件上传，不要粘贴**。

## 审核标准（2026-09-23 更新）

项目负责人确定：**GPT 审核即最终审核，不做人工逐条复核**。第 1 轮的 `consensus-confirmed-*.csv`
即由 GPT 完成，已按 `gpt_final_review_*` 记入 gold。第 2 轮沿用同一标准，下文「人工确认」
一节相应改由 GPT 执行，但有一条新增要求：

**做最终审核的 GPT 必须是一个新会话，且看不到哪一份标签是它自己标的。** 第 2 轮里 GPT 同时是
两个标注者之一；如果让同一会话、在知道来源的情况下裁决分歧，它会系统性地偏向自己的标签。
交给它的分歧表只写「标注 A / 标注 B」，不写 machine / gpt。

## 人工确认包怎么用（历史说明：第 1 轮）

`consensus-to-confirm-*.csv` 里每行是**两模型已经一致**的条目，附上 question、claim/answer、
被引用原文与那个一致标签。做的是**确认**不是重标：

- 同意 → `accept_or_override` 填 `accept`；
- 不同意 → 填 `override`，并在 `override_entailment` / `override_relevance`
  （或 `override_completeness` / `override_verdict_reading`）里写正确值。

确认后这些行统一升级为 `human_adjudicated`，不再叫 `model_consensus`。
**32 条待确认 claim 里有 7 条标为 `unsupported`，这 7 条务必逐条看**——它们是 P2 安全指标的
直接样本。

## 这一轮的三条程序约束

1. **新规则不回写旧标签。** 第 2 轮如果又出现需要新判据的情形，那是 CODEBOOK v3；
   一旦升到 v3，**受影响的第 1 轮样本要统一重裁**，不能只在新样本上用新规则。
2. **`unsupported` 的一致项全审。** `make semantic-agreement` 现在默认对
   `entailment=unsupported` 的一致项做 100% 抽检，其余保持 20%
   （`--critical-label` / `--critical-value` / `--critical-fraction` 可调）。
3. **目标不是凑满 176 条**，而是达到可校准的构成：`unsupported` ≥ 25–30 条经人确认、
   `supported` 有足够对照、四种题型都有覆盖、B1/B2 与 RAG/workflow 不严重偏斜。
   全量标完后先核对这四项，达不到就补，不达标不开始 P2。
