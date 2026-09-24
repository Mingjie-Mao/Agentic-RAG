# 待裁决：4 个规则决定，不是 32 个逐项判断

两份独立模型标注（我 / GPT，互不可见）在同一批 50 条 claim + 30 条 answer 上的一致率：

| 标签 | 一致率 | 分歧 |
| --- | ---: | ---: |
| entailment | **0.90** | 5 |
| completeness | **0.967** | 1 |
| verdict_reading | **0.933** | 2 |
| relevance | **0.74** | 13 |

前三项的一致率足以把「两模型一致」的部分当作可用的暂定标签；**relevance 0.74 是判据问题**，
因为分歧是单向的：我判 `answers` 而 GPT 判 `related` 有 12 次，反向只有 1 次。

分歧归并之后只剩 **4 个规则决定**。每条都给出我的建议与理由；你只需回答「采纳 / 不采纳」。

---

## 决定 1｜合取问句里只回答一支的 claim（影响 12 条）

问句形如「Does A suggest X, **while** B indicates Y?」，claim 只陈述了 X 或只陈述了 Y。

- 我判 `answers`，GPT 判 `related`。
- **建议采纳 `answers`。** 理由是结构性的：relevance 是 **claim 层**标签，completeness 才是
  answer 层的。要求单条 claim 回答整道复合问题，等于把 answer 层的判据搬到 claim 层，
  relevance 会退化成 completeness 的副本——codebook「单位」一节写的正是「单条 claim 判断不了
  整道题是否答完整」。
- 采纳后 relevance 一致率从 0.74 升到约 0.98，剩 1 条真分歧（见决定 4）。

## 决定 2｜claim 断言「两侧之间的关系」，而引用只覆盖一侧（影响 3 条）

例：「两篇报道在 X 上保持一致」「两家媒体的立场一致」。

- 我的判法：**两侧都在引用里但支持不足 → `partial`；有一侧完全不在引用里 → `unsupported`**。
  GPT 一律更宽（`supported` / `partial`）。
- **建议采纳我的判法。** claim 的实质断言就是那个「关系」，只有一侧材料在构造上推不出关系；
  而两侧都在、只是证据薄，属于「多断言只支持一部分」。

## 决定 3｜回答给了数值但没有选定分支（影响 2 条，3 个标签格）

例：问「更大还是更小」，答「Rogers 下跌 2.2%」；问「A 还是两者共同趋势」，答只肯定了 A 支的两句。

- completeness：我 `partial`，GPT `complete`；verdict_reading：我 `other_choice` / `yes`，GPT 均为 `unclear`。
- **建议 completeness 采纳我的 `partial`**（必答项是那个比较/选择，没给就是漏项），
  **verdict_reading 采纳 GPT 的 `unclear`**，并在 codebook 里写死：
  **`other_choice` 只用于回答**明确选定了**某一支的情形；没选定就是 `unclear`。**
  规则 7（全部分句被肯定 → `yes`）只适用于**合取**问句，不适用于「A 还是 B」的**析取**问句。

## 决定 4｜剩下 3 条个案（各 1 条，无法归并）

| 条目 | 我 | GPT | 我的建议 |
| --- | --- | --- | --- |
| `B2-MH-10482bbfd0ca#1`：claim 写「与 GPT-3.5 比较」，引用原文写的是 GPT-4 | partial | unsupported | **采纳 GPT**：与可见信息冲突，按规则 3 就该判 `unsupported`，我判轻了 |
| `B1-MH-0a7a2c277008#2`：claim 写「Kelce 出席 Swift 的演唱会」，原文是 Swift 去看 Kelce 的比赛 | unsupported | partial | **采纳我的**：方向写反属于与证据冲突，不是「否定式断言无法证否」 |
| `B2-MH-0f910380b759#3`：问「哪位名人」，claim 主语是 Kelce 的意图，只确认了问题里的一个识别性限定语 | related | answers | **这条请你定**。按决定 1 的口径，wh 问句的必答项只有「名字」，确认限定语不算回答；但它确实提到了 Taylor Swift。两种读法都站得住 |

---

## 采纳后的账

- 我在 7 条非 relevance 分歧里**让步 4 条**（决定 3 的两个标签格 + 决定 4 的第一条）——
  这正说明两模型交叉是有用的，不是走过场。
- 三档 provenance：决定 1–4 覆盖的条目落为 `human_adjudicated`；两模型一致且未抽检的落为
  `model_consensus`；抽检样本（claim 7 条 + answer 6 条，按 `sha256(item_id)` 固定抽取）
  仍需你抽查，以防两个模型共享同一个盲点。
- **只有 `human_adjudicated` 这一档能叫金标。** 校准报数时三档分开写。

---

# 裁决结果（已应用，2026-09-23）

四条决定由 `scripts/build_semantic_gold.py` 逐条写死并应用，产出 `gold-claims.csv` /
`gold-answers.csv`。决定 2 按你的改法执行——**`partial` 只用于可分离的多断言**，
关系断言只有 `supported` / `unsupported` 两种结局。

应用过程中改掉了我自己两处错误：

1. 决定表里 `B1-MH-0a7a2c277008#2` 写了两次（relevance 一次、entailment 一次），后者会覆盖前者、
   悄悄丢掉 relevance 裁决——ruff 的重复键检查抓到的；
2. 枚举决定 1 的条目时漏了 `B2-MH-179cc0ffdb05#1/#2`（正是「Does A suggest X, **while**
   B indicates Y」），却写进了两条不在前缀里的条目。修正后没有 `contested_unresolved` 残留。

**我在这轮一共让步 5 个标签格**：决定 2 的两条关系断言（partial → supported）、
GPT-3.5 与 GPT-4 那条（partial → unsupported）、以及决定 3 的两个 `verdict_reading`
（other_choice / yes → unclear）。GPT 则在 12 条 relevance 上让步。

## provenance 分档

| 档 | claim | answer | 能否作金标 |
| --- | ---: | ---: | --- |
| `human_adjudicated` | 17 | 2 | 能 |
| `rule_applied`（一致但被规则改写） | 1 | 0 | 能 |
| `model_consensus_audit_pending`（待你抽检） | 7 | 6 | 抽检确认后才能 |
| `model_consensus` | 25 | 22 | **不能** |

## 一个必须现在说的算术

**可用金标 = 18 条 claim + 2 条 answer。这不够做 P2 校准。**

而且是**结构性**的：两个模型越一致，人碰过的条目越少，金标反而越小。这轮一致率 0.90，
所以 50 条里只有 18 条经过人的判断。

校准要看的核心指标是「把 `unsupported` 判成 `supported` 的比例」，它只能在 `unsupported`
样本上估。50 条里 `unsupported` 占 26%，但落进可用金标的只有 **6 条**——在 6 条上估一个比例，
置信区间会宽到无法在四个评分器之间排序，这正是 §14A 用 34 条得到区间重叠的老路。

按 26% 的比例倒推：

| 想要的 `unsupported` 样本 | 需要的已标 claim |
| ---: | ---: |
| 11 | ~50 |
| 20 | ~91 |
| 30 | ~136 |

所以建议：

1. **你用 15 分钟批量确认 33 条一致项**（它们两模型一致，是浏览确认不是逐条判断），
   金标立刻变成 50 条 claim / 30 条 answer；
2. 同时让 GPT 与我把剩余 126 条 claim / 73 条 answer 也标掉，按这轮的经验分歧约 20–30%，
   届时分歧大概率仍能归并成几条规则，而不是几十个逐项判断；
3. 全量 176 条标完并确认后，`unsupported` 约 39 条，才够把四个评分器分开。

**在金标达到这个量级之前不要开始 P2 校准**——在 18 条上跑出来的 precision/recall 只会重复
§14A 的结论「区间重叠、谁都没资格拦截」，而且这次连「样本太小」这个理由都是自己造成的。

---

# CODEBOOK v3（2026-09-24）

第 2 轮比对时发现：上面四项决定只写进了 `scripts/build_semantic_gold.py`，**没有写进 codebook**，
第 2 轮发给 GPT 的标注包因此仍是 v2。Claude 一侧按「v2 + 四项决定」标，GPT 按 v2 标——
第 2 轮 relevance 一致率只有 0.698，38 处分歧里 31 处就来自这个版本差。

处理：

- 四项决定写成规则 11–14，升为 v3（v2 存档为 `CODEBOOK-v2.md` / `CODEBOOK-annotator-v2.md`）。
- 决定 4 原本只是单条个案。Claude 在第 2 轮自行按宽口径推广（wh 问句里凡是只确认限定语的都判
  `related`），这个口径与第 1 轮同题 #1、#2 的 gold（`answers`）冲突。v3 改取窄口径：看 claim 的
  断言对象是不是答案实体。
- 按版本纪律，两轮全部样本按 v3 统一重裁：relevance 176 条全部由新会话 GPT 盲标；entailment 复核
  第 2 轮分歧、全部一致的 `unsupported`、20% 抽样，以及两轮中所有可能的关系断言（规则 12）；
  answer 层复核第 2 轮全部 `partial` / `unclear` / `other_choice`、20% 抽样，以及第 1 轮受规则 13
  影响的条目。审核包见 `round2/`，生成脚本 `scripts/build_semantic_final_review.py`。

第 2 轮 claim 独立比对（v3 重裁之前）：entailment 一致率 0.921（116/126），relevance 0.698（88/126）。

## v3 最终审核结果与 v3.1（2026-09-24）

新会话 GPT 返回三份审核文件，结构校验全部通过（行数、键、行序、非 `final_*` 列未改、取值合法）。

- **entailment**：10 条分歧里采纳 GPT 原标注 7 条、Claude 3 条；其余 73 条带现有标签的审核行
  （41 条一致 `unsupported`、15 条抽样、17 条规则 12 复核）**改判 0 条**。第 1 轮是 32 条改判 1 条。
  零改判可能是标签本来就对，也可能是审核人被现有标签锚定——两者无法从数据上区分，作为局限写进报告。
- **relevance（盲标）**：与第 1 轮 v2 gold 一致 47/50，与第 2 轮 GPT 原标注一致 104/126、与 Claude
  一致 86/126，不是照抄；38 处第 2 轮分歧中 wh 题 28 条判 `answers`、是非题 7 条判 `related`，与 v3 规则相符。
- **answer**：27 条全部接受原标签，改判 0 条。

**冲突与 v3.1**：盲标把 `B1-MH-050827e2c17f#1/#2` 从 `answers` 改成了 `related`，而这两条是负责人
9/23 按决定 1 逐条裁过的。冲突来自 Claude 写 v3 时自行补的「规则 11 与规则 5 的边界」，那条边界未经
负责人批准。负责人决定**保留原裁定**，并把规则 11 的边界按先例改写（v3.1）：问句要求核实「某来源说了
什么」时，回答该分句的 claim 算 `answers`，即使问句同时问两者是否相似；只问关系的问句，一侧陈述才算
`related`。按 v3.1 统一套用，盲标里与之矛盾的共 9 条 relevance 改为 `answers`（2 条负责人裁定、
7 条规则套用），逐条写在 `scripts/build_semantic_gold_v3.py::RELEVANCE_RULINGS`。entailment 不受影响。
