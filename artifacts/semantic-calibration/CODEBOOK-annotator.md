# 语义标注 codebook v3（标注者版）

> 本文件是 CODEBOOK v3 的节选，只含标注 entailment / relevance / completeness /
> verdict_reading 所需的部分。**判据规则 1–14 与完整版相同**；唯一差别是规则 14 的先例改写成了不带条目编号的例子，以免对待重审条目形成锚定。

> 规则 9、10 与 `retrieval_incomplete_confirmed` 是模型试标过程中新发现的情形，**在人工标注开始
> 之前**补入，不修改任何既有定义。此后不再改判据。
>
> **版本纪律**：若后续批次出现 v2 无法覆盖的情形，那是 CODEBOOK v3，而不是在 v2 上打补丁。
> 一旦升到 v3，**受影响的既有样本必须按 v3 统一重裁**——新规则不允许只作用于新样本，
> 否则同一批 gold 里会混着两套判据。

v1 的判据在 21 条 claim / 14 条 answer / 8 条 Yes-bias 的试标里暴露 7 处歧义
（记录见 `GUIDELINE-ISSUES.md`）。v2 把这 7 处逐条定死。**标注开始后不再改判据**；
真要再改，就作废已标部分重标，而不是让前后两套判据混在同一批 gold 里。

**标签是类别，不是 1–5 分。**

## 单位

| 标签 | 单位 | 文件 |
| --- | --- | --- |
| entailment、relevance | 单条 claim | `claims-to-label.csv` |
| completeness、verdict_reading | 整条 answer | `answers-to-label.csv` |

单条 claim 判断不了「整道题是否答完整」，所以 completeness 不在 claim 层。

---

## entailment：被引用的原文是否支持这条 claim

| 值 | 判据 |
| --- | --- |
| `supported` | 引用原文足以推出 claim 的**全部实质断言**，不需要额外常识 |
| `partial` | claim 含多个实质断言，证据只支持其中一部分 |
| `unsupported` | 证据推不出、与证据冲突，或需要关键外部信息 |

**`partial` 要严格**：措辞不同、换了说法、概括得更粗，只要实质断言被证据支持，仍是 `supported`。
只有「claim 说了 A 和 B，证据只给了 A」才是 `partial`。

### 规则 1｜判 claim 的**全文**，不是只判它相对问题新增的部分

生成器常把问题整句改写成陈述句，于是问题里的每个限定语都变成 claim 的断言。**这些限定语照样
算实质断言。**

理由：用户读到的是整句话，并且它后面挂着引用。claim 说「Google 因为默认搜索协议、对互联网外观
的影响、手机与应用商店的法律结果、损害新闻出版商收入而处于反垄断讨论的中心」，而证据只支持其中
两项时，系统**确实**在把没有依据的话讲得像有依据——这正是评分器要抓的东西。按「只判新增部分」
去判，会把这种错误从 gold 里抹掉。

代价是 `partial` 会偏多。这不是判据的缺陷，而是一个结论：**复述式生成会常态化地产出部分无依据的
claim**。为了让后续分析能把两种 `partial` 分开，请在 `note` 里写明属于哪一种：

- `note: echo` —— 未被支持的部分来自问题的限定语（claim 复述问题）；
- `note: missing-fact` —— 未被支持的部分是 claim 自己引入的事实。

### 规则 2｜来源归属不算实质断言

分块只有首块带 `Source:` 头，中间块没有。claim 却常写「**The Age** 的文章指出……」。
**只判内容，不判这个归属**；无法核验时在 `note` 写 `attribution-unverifiable`。

### 规则 3｜文档头里的元数据（日期、作者、媒体）同理

能从引用块或文档头核验就核验；**无法核验时不作为判 `partial` 的理由**，在 `note` 记
`metadata-unverifiable`。claim 写了一个与可见信息**冲突**的日期，仍然算 `unsupported`。

### 规则 9｜claim 把「据称」去掉了，判 `unsupported`

原文写的是「某人指控 / 诉状称 / 据报道」，claim 写成既成事实，这是**强度不匹配**，不是措辞差异。
证据支持的是「存在这样一项指控」，不是「这件事成立」——判 `unsupported`，`note` 写 `hedge-dropped`。

实例：原文「Wired 刊登的评论文章*指控* Google 操纵搜索以最大化广告收入，Google 强烈否认」，
claim 写成「Google manipulates Search to maximize ad revenue」。

### 规则 10｜解释性套话不算实质断言

claim 结尾常挂一句「…, indicating different aspects of the controversy」这类评论性短语。
它不提出可核验的事实，**不因此判 `partial`**；`note` 写 `meta-qualifier`。
与规则 1 的区别：规则 1 管的是**事实性**限定语（数字、事件、对象），本条管的是**评论性**尾巴。

### 规则 4｜对来源的否定式断言判 `partial`

「这篇文章并未明确表示 X」这类断言，片段证明不了整篇文章没说过某事。
**只要引用块本身不与之冲突，判 `partial`**，`note` 写 `negative-claim-unfalsifiable`；
引用块明确说了 X，则判 `unsupported`。

---

## relevance：这条 claim 是否回答了问题所问

| 值 | 判据 |
| --- | --- |
| `answers` | 直接回答问题要求 |
| `related` | 有关，但没有真正回答所问 |
| `off_topic` | 基本无关 |

### 规则 5｜必须对着**完整问题**判

表格软件会把长单元格显示成一行。**判 relevance 前先把问题单元格展开读完**。
试标前期按截断问题判错过 3 条：三条 claim 都在复述前提，没有一条回答「两家立场是否一致」，
按完整问题读应当是 `related` 而不是 `answers`。

---

## completeness：整条回答是否覆盖问题要求

| 值 | 判据 |
| --- | --- |
| `complete` | 每个必答项都有**实质内容** |
| `partial` | 至少答了一部分，漏了必答项，或某必答项只有复述没有内容 |
| `missing` | 基本没有完成问题要求 |

### 规则 6｜复述不算覆盖

把问题换成陈述句、没有给出具体数值 / 事实 / 判断的分项，**不算覆盖**，该项按漏项处理。

**例外：是非题的必答项就是那个判断本身。** 对「Does A …?」这类问题，把命题肯定地陈述一遍
就已经交付了判断，算 `complete`；对「哪家公司 / 多少 / 列出三项」这类问题，复述问题不算覆盖。

---

## verdict_reading：读者从回答文字里读到的结论

| 值 | 判据 |
| --- | --- |
| `yes` / `no` | **只根据回答文字**就能明确读出是 / 否 |
| `unclear` | 人也不能稳定读出 |
| `other_choice` | 是判断题，但答案不是是非（更大 / 更小、一致 / 不一致等）；在 `note` 写下答案选了哪一支 |
| `not_applicable` | 本身不是判断题（问名称、问数值、问清单） |

### 规则 7｜合取问句怎么读

问句形如「Does A suggest X, **while** B indicates Y?」：

- **全部分句都被肯定 → `yes`**；
- **任一分句被明确否定 → `no`**；
- **只处理了一部分分句 → completeness `partial`，verdict_reading `unclear`**。

注意这类回答里往往**根本不出现 yes / no 字样**——把判断读出来正是这个字段存在的理由。

### 规则 8｜非二元判断题用 `other_choice`

数据集里有「更大还是更小」「一致还是不一致」这类问句（金标写作 `Smaller` / `Consistent`），
`yes/no` 表达不了。**这同时是系统侧的缺口**：结论字段目前只能输出 `yes|no|unclear`，
也表达不了这类答案——请在 `note` 里记下来，这条会进 output protocol 的待办。

---

## v3 新增规则 11–14（2026-09-24）

> 来源：项目负责人对 v2 分歧的四项裁决。
> 规则 1–10 一字未改。**两轮全部样本按 v3 统一重裁**，不只作用于新样本。

### 规则 11｜合取问句的单支 claim：relevance 判 `answers`（决定 1）

问句形如「Does A suggest X, **while / and** B indicates Y?」，每个分句都是必答项；
只处理其中一支的 claim 判 `answers`。

**与规则 5 的边界（v3.1，按负责人先例改写）**：看问句是否要求**核实某个来源说了什么**。

- 问句里有「Does / Did A suggest / report / mention X」这类分句——即使同时还问两者是否相似
  （while / similar to / same as / compared to / before 等）——回答其中任一分句的 claim 判 `answers`；
- 问句只问关系（「A 与 B 之间的报道是否一致 / 有无变化 / 更大还是更小」），来源内容只作为前提描述
  出现，这时只陈述一侧、不下关系判断的 claim 仍按规则 5 判 `related`。

### 规则 12｜关系断言的 entailment（决定 2）

claim 断言两侧之间的关系（一致 / 不一致 / 有变化 / 相同 / 不同）：

- 两侧事实都在引用里，且足以推出该关系 → `supported`；
- 两侧都在，但推不出该关系 → `unsupported`；
- 任一侧不在引用里 → `unsupported`；
- 只有写成「A 说 X、B 说 Y，因此一致」这种**可拆开的多断言**，才按「部分断言成立」判 `partial`。

### 规则 13｜给了数值但没选分支（决定 3）

判断题的回答只罗列数值或事实、没有选定是 / 否或某一支 → completeness `partial`，
verdict_reading `unclear`。`other_choice` 只用于回答**明确选了**某个非是非分支。

### 规则 14｜wh 问句的 relevance（决定 4，取窄口径）

wh 问句（Who / Which / What …）的必答项是答案实体。

- claim 以答案实体为断言对象（主语，或「X 就是 …」），即使只确认了题干里的一个限定语 → `answers`；
- claim 的断言对象不是答案实体（例如以第三方的行为、意图为主语），答案实体只是顺带出现 → `related`。

例：问「哪位名人……？」，答案是某歌手。claim「该歌手曾在某球场演出」→ `answers`；claim「某球员有意邀请她观赛」主语是球员的意图 → `related`。

---

## 标注不看什么

看不到金标答案、看不到系统输出的结论字段、看不到当前指标是否判对。
系统结论是否正确**事后按 item_id 合并算出**，不作为标注项——让标注者给系统打分会引入锚定。

其它记录性约定：回答语言与问题语言不一致（英文问题、中文 claim）请在 `note` 写 `language-mismatch`，
它不改变 entailment / relevance 的判定。

---

## 标注顺序

行按分层轮转排列。**先标一个平衡前缀**（约 50 条 claim、30 条 answer、20 条 Yes-bias），
确认按 v2 判据不再频繁犹豫，再标完剩余部分。
**不要为了标满 176 条，把不稳定的标签变成伪 gold。**
