# 交给另一个模型（GPT）独立标注的说明

目的不是「省掉人工」，而是**把人要看的东西从 558 个判断压缩到只剩分歧项**。
因此有一条硬要求：**GPT 必须在看不到本项目任何已有标签的情况下标**，否则两边就不独立，
一致率也就没有意义。

## 交给 GPT 什么

**优先给这两个前缀文件**，它们正好是模型预标覆盖的范围，一致率能 100% 对上：

- `claims-to-label-prefix50.csv`（50 行，79 KB）
- `answers-to-label-prefix30.csv`（30 行，17 KB）
- `CODEBOOK.md`（v2，判据）

想一次标完再用全量的 `claims-to-label.csv`（176 行）/ `answers-to-label.csv`（103 行）；
超出前缀的部分暂时没有对照，要等预标补齐才能算一致率。

文件偏大，建议**以附件上传**而不是粘贴，避免中途被截断——`cited_evidence` 一列很长，
截断会直接导致 entailment 判错。

**不要给** `machine-prepass-*.csv`（我的标签）、不要给金标答案、不要给系统输出的结论字段、
也不要说「另一个模型标成了什么」。

## 提示词（直接粘贴）

> 你是标注员，不是助手。请严格按照我给你的 codebook 标注，不要解释、不要改写数据、不要给建议。
>
> 规则全文见附件 CODEBOOK.md v2，其中最容易标错的四条，请特别遵守：
>
> 1. **entailment 判 claim 全文**。问题里的限定语被复述进 claim，照样算实质断言：claim 说了
>    A、B、C、D 四件事而证据只支持两件，就是 `partial`，不是 `supported`。判 `partial` 时在
>    note 里写 `echo`（未支持部分来自问题的复述）或 `missing-fact`（claim 自己引入的事实）。
> 2. **来源归属与日期不算实质断言**。分块只有首块带 `Source:` 头，claim 里的「某某媒体的文章
>    指出……」「2023 年 10 月 12 日」如果无法核验，**不要因此判 partial**，在 note 里写
>    `attribution-unverifiable` 或 `metadata-unverifiable`。与可见信息冲突的，仍判 `unsupported`。
> 3. **否定式断言**（「这篇文章并未表示 X」）判 `partial`，note 写 `negative-claim-unfalsifiable`；
>    引用块里明确说了 X，才判 `unsupported`。
> 4. **completeness：复述不算覆盖**。把问题换成陈述句、没有给出具体数值/事实/判断的分项，按漏项
>    处理。唯一例外是是非题——把命题肯定地陈述一遍就已经交付了判断，算 `complete`。
>
> `verdict_reading` 只看回答文字读出什么结论：全部分句被肯定读作 `yes`，任一分句被明确否定读作
> `no`，只处理了一部分读作 `unclear`，「更大还是更小」「一致还是不一致」这类非是非题读作
> `other_choice`（在 note 写选了哪一支），问名称/数值/清单的读作 `not_applicable`。
>
> 输出：原样返回 CSV，只填写空白的标签列与 note 列，**不要改动 item_id、claim_index、question、
> claim、cited_evidence、answer 这些原始列**，不要增删行。

## 拿回来之后放哪里

存成 `gpt-prepass-claims.csv` / `gpt-prepass-answers.csv`（保持原列名，新增的标签列名不限，
合并脚本按列名里的 `entailment` / `relevance` / `completeness` / `verdict_reading` 关键字识别）。

然后跑：

```bash
make semantic-agreement
```

它会输出两边的一致率、按标签的混淆矩阵，以及 `to-adjudicate.csv`——**只有分歧项**，
那才是需要人看的部分。
