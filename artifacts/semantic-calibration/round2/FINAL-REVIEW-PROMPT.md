# 最终审核说明（CODEBOOK v3）——请在全新的 GPT 会话里使用

**必须开一个新会话**，不要在做过标注的会话里继续。附件共 4 个：

- `CODEBOOK-annotator.md`（v3，规则 1–14）
- `final-review-entailment.csv`（83 行）
- `final-review-relevance.csv`（176 行）
- `final-review-answers.csv`（27 行）

把下面这段原样发给 GPT：

---

你是语义标注的最终审核人，按附件 `CODEBOOK-annotator.md`（v3）执行。你的判断就是最终标签。

通用要求：

1. 每一行都先把 `question` 单元格**展开读完**（规则 5），再下判断。
2. 只根据 `cited_evidence` 判 entailment，不用外部知识。
3. 只填 `final_*` 列和 `note` 列；不改其它列，不增删行，不改行序。
4. 推翻现有标签，或在 A/B 两份里选一份时，`note` 写依据的规则号（例如 `R12`、`R4`）。
5. 三个文件都填完后，原文件名返回。

**文件 1：`final-review-entailment.csv`**，按 `reason` 列分四类：

| reason | 你看到什么 | 要做什么 |
| --- | --- | --- |
| `disagreement` | `label_A`、`label_B` 是两位标注者给出的不同标签，来源不公开 | 独立判断；可以选 A、选 B，也可以两者都不选 |
| `audit_unsupported_agreement` | 两位标注者一致判 `unsupported`（`current_label`） | 独立核对。这一类全部复核，因为「把没证据的说法判成有证据」是整个校准要测的错误 |
| `audit_sample_20pct` | 两位标注者一致的标签（`current_label`） | 独立核对 |
| `v3_rule12_recheck` | 按关键词筛出的「可能是关系断言」的 claim，`current_label` 是现有标签 | 按规则 12 复核。关键词会误筛，若 claim 其实不是关系断言，按规则 1–10 判即可，通常就是确认原标签 |

**文件 2：`final-review-relevance.csv`**：这一轮是盲标，没有给出任何现有标签。
按规则 5、11、14 为每条 claim 填 `final_relevance`。
提醒：合取问句（while / and 能拆成两个是非问）的单支 claim 判 `answers`；
比较问句（一致 / 变化 / 相同 / 不同）里只陈述一侧、不下关系判断的 claim 判 `related`；
wh 问句看 claim 的断言对象是不是答案实体。

**文件 3：`final-review-answers.csv`**：`current_*` 是现有标签。按规则 6、7、8、13 独立核对，
填 `final_completeness` 和 `final_verdict_reading`。注意规则 13：只罗列数值、没有选定分支 →
`partial` + `unclear`；`other_choice` 只用于回答明确选了某个非是非分支。

---

## 收回之后（给 Claude 的步骤，不发给 GPT）

1. 放回 `artifacts/semantic-calibration/round2/`，文件名不变。
2. 合并进 gold：provenance 按实际来源分档——`gpt_final_review_v3_accept` / `_override` /
   `_disagreement_resolved`；relevance 统一为 `gpt_final_review_v3_blind`。
3. 核构成：unsupported 目标 25–30 条以上；四类题型、B1/B2、RAG/workflow 均有覆盖。
4. 之后才开始 P2 打分器校准。
