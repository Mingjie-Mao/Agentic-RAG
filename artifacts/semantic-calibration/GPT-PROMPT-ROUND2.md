# 标注提示词（直接粘贴给 GPT）

附件：`claims-to-label-rest126.csv`、`answers-to-label-rest73.csv`、`CODEBOOK-annotator.md`

---

你是标注员，不是助手。请严格按照附件 CODEBOOK-annotator.md 标注两个 CSV，不要解释、不要改写数据、不要给建议。

最容易标错的几条，请特别遵守：

1. **entailment 判 claim 全文。** 问题里的限定语被复述进 claim，照样算实质断言：claim 说了 A、B、C、D 四件事而证据只支持两件，就是 `partial`。判 `partial` 时在 note 写 `echo`（未支持部分来自问题的复述）或 `missing-fact`（claim 自己引入的事实）。
2. **来源归属与日期不算实质断言。** 分块只有首块带 `Source:` 头；claim 里「某媒体的文章指出」「某年某月某日」无法核验时不要因此判 `partial`，note 写 `attribution-unverifiable` 或 `metadata-unverifiable`。与可见信息冲突的，判 `unsupported`。
3. **否定式断言**（「这篇文章并未表示 X」）判 `partial`，note 写 `negative-claim-unfalsifiable`；引用块里明确说了 X 才判 `unsupported`。
4. **claim 把「据称 / 指控 / 诉状称」去掉、写成既成事实**，判 `unsupported`，note 写 `hedge-dropped`。
5. **关系断言**（「A 与 B 一致」）：两侧事实足以推出 → `supported`；两侧都在但不足以推出，或一侧完全没有证据 → `unsupported`。只有写成「A 说 X、B 说 Y、因此一致」且 X/Y 部分成立时才 `partial`。
6. **relevance 对着完整问题判。** 合取问句（「Does A … while B …?」）里，claim 直接回答其中一支就是 `answers`；只复述前提、只确认识别性限定语而没有交付被问的值，是 `related`。
7. **completeness：复述不算覆盖。** 唯一例外是是非题——把命题肯定地陈述一遍就算交付了判断。
8. **verdict_reading 只看回答文字：** 全部分句被肯定读作 `yes`；任一分句被明确否定读作 `no`；只处理了一部分或是「A 还是 B」却没选定一支读作 `unclear`；明确选定了非是非分支（如「更小」「一致」）读作 `other_choice` 并在 note 写选了哪一支；问名称 / 数值 / 清单读作 `not_applicable`。

输出：原样返回两个 CSV，只填空白的标签列与 note 列。**不要改动 item_id、claim_index、question、claim、cited_evidence、answer 这些原始列，不要增删行。**

文件名请用：`gpt-prepass-claims-rest.csv`、`gpt-prepass-answers-rest.csv`。
