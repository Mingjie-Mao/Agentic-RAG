# 模型预标结果（平衡前缀）

**这不是金标。** 全部标签由模型按 CODEBOOK v2 产出，`provenance = model_prepass_not_gold`。
人工标注文件仍然空白、仍然盲标。本文的用途有两个：说明判据在实际数据上的表现，
以及给下一步的 P1 取舍提供**待人工确认**的方向。

规模：claim 50 / 176，answer 30 / 103，Yes-bias 20 / 37。

## 一、标签分布

| 标签 | 分布 |
| --- | --- |
| entailment | supported 24、partial 15、unsupported 11 |
| relevance | answers 40、related 10 |
| completeness | complete 26、partial 4 |
| verdict_reading | yes 11、not_applicable 11、no 3、unclear 3、other_choice 2 |

三点读法：

1. **`unsupported` 占 22%**（11/50）。引用校验全部通过的前提下仍有五分之一的 claim 推不出来——
   这正是 §10 写过的边界第一次被量化：引文逐字存在 ≠ claim 成立。
2. **`partial` 占 30%**，其中多数带 `echo` 注记：claim 复述了问题的限定语，而证据只覆盖一部分。
   这是规则 1 的直接后果，也是「复述式生成」的代价。
3. **`verdict_reading` 里有 3 条 `no`、11 条 `yes`，而这些回答里往往根本没有出现 yes/no 字样**
   （例：问「是否改变」，答「保持一致」）。这从标注侧再次确认了结论字段的必要性。

## 二、Yes-bias 归因（20 条）

| 类别 | 条数 |
| --- | ---: |
| 检索没拿齐（文档级，机械判定后人工确认） | 7 |
| **E 取对文档、取错段落（块级检索失败）** | 4 |
| A 比较 / 关系推理本身错 | 3 |
| B claims 含糊或自相矛盾 | 2 |
| D 问题或 gold 本身可争议 | 2 |
| F claim 被自己引用的原文推不出 | 1 |
| **C claims 清楚但结论分类器读反** | **1** |

**检索合计 11/20（55%），分类器只有 1/20（5%）。**

如果人工归因得到同样的方向，那么 P1 的取舍就很清楚：**子问题检索 + 块级覆盖校验**覆盖过半错误，
而 **typed verdict / 分类器约束只覆盖约 5%**——先做前者。这与「先修 output protocol」的直觉相反，
所以更需要人工确认，不能拿模型预标当结论。

## 三、顺带发现的三个系统缺陷

1. **订阅样板分块被当成证据。** 609 篇语料里 **37 篇（6.1%）正文开头就是「Stay ahead of the
   trend / Please enter a valid email address」这类订阅样板**，中位长度 440 字符，足以占满第一个
   700 字分块；34 篇来自 The Independent。这些块照样被检索、被引用：本次 20 条归因里有 2 条、
   50 条 claim 里有 1 条，引用的就是这种块。系统里已有 `_boilerplate_only` 过滤，但它只匹配中文
   fixture 的免责声明，对英文语料完全无效。**属于 P1 检索层，本轮不修。**
2. **中英文混答。** 英文问题得到中文 claims（本次 3 例）。不影响 entailment 判定，但对读者是缺陷。
3. **问题里存在事实性笔误。** 例：某题写「Business Today 在 2022-10-07 的报道」，而文章是
   2023-10-07。属于 benchmark 质量问题（D 类），修 benchmark 而不是修系统。

## 四、这份预标接下来怎么用

- 人按 v2 盲标同一批前缀；
- 另一个模型（GPT）独立标同一批，见 `GPT-LABELING-PACKAGE.md`；
- `make semantic-agreement` 算两份模型标注的一致率，只把**分歧项 + 20% 抽检**交给人；
- 人工标签回来后，本文件的标签作为「LLM judge」那一臂参与 P2 校准。

**在人工标签回来之前，本文的任何数字都不得写进正式结论。**
