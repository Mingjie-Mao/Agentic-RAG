# P2 / P3 / P4 独立审核规范

审核输入是实际开发拆分、原文和已生成断言。助手编写的预期事实、旧机械标签和模型输出均不是人工金标。审核者应独立阅读来源，说明身份与理由；本轮尚未有人工审核者参与。

## 文件与标签

P2 最新使用 `.runtime/semantic-slot-v2/review-r4/pilot.json` 和 `expanded-distinct.json`。前者 260 条且全部 dev；后者 2,419 个不重复输入，去除了 1,309 条重复运行输入，并保留到原始输入的映射。原始文件不可改写，另存为 `pilot-reviewed.json` / `expanded-distinct-reviewed.json` 后只增加 `reviews`、必要时增加 `adjudications`。所有原始 `items` 与 `source_sha256` 必须保持相同。

可用项目根目录的 `semantic_review.html` 在本地选择审核包、逐项阅读原文并手动填写身份、独立性、标签与理由，然后导出 JSON。未填标签不自动补全，部分审核可保存续审；正式门槛仍拒绝不完整审核。该界面仅支持组件及影子分歧，P4 题目/答案使用下述结构。界面表单逻辑已检查，视觉预览因浏览器工具禁止 file URL 未完成。助手不得替实际审核者填写姓名、人工身份或独立性声明。

每个审核记录包含 `id`、`reviewer`、`reviewer_type`、`independent`、`label`、`reason`。`reviewer_type` 仅在真实独立人工审核后填写 `human`，`independent` 仅在该事实成立后填写 `true`。同一项目有不同标签时，另一个独立审核者在 `adjudications` 提供同样字段；不能用最后写入的标签覆盖争议。

| 输入 | 标签含义 |
|---|---|
| relevance | relevant：有助于回答这个实际槽位；irrelevant：仅主题相关或无关；unclear：输入不足以确认 |
| coverage | supported：全部必答内容有直接原文支持；partial：只支持复合要求的一部分；unsupported：未回答；contradicted：同主体、属性和适用范围有不相容内容；unclear：范围或要求不明确 |
| claim | supported：该断言被自己的引用完整支持；partial：复合断言仅部分支持；unsupported：缺支持但无明确反证；contradicted：引用有明确反证；unclear：原文或断言有歧义 |

数值随版本变化、不同主体的差异不自动构成冲突。单一原子断言错误不能按“部分正确”处理。裁判断线、JSON 错误、引用不存在是技术 unknown，不能当作事实缺失的金标。模型 confidence 不参与完成判断。

## 分区与校准

同一问题的变体和所有涉及的文档、版本家族保持同一分区，连通的干扰文档也不跨分区；三个组件之间也不能跨分区复用同一来源。中文、英文分别报告。原中文来源主要形成一个相连家族且无 dev；现在追加了 6 份助手编写的合成中文开发文档，18 次真实问答构建 118 relevance / 138 coverage / 25 claim，整体在生成前指定为 dev。新来源与旧文档和问题隔离，没有拆分原家族，旧 test 输入未变。合成来源的代表性与语义标签仍需人工核验，不是外部真实语料或已标金标。

重复输入不作为新增样本参与最低样本数和统计；旧 1,696 条断言只有 387 个不重复输入，新增中文来源后共有 412 个不重复断言。若同样输入得到不同标签，须复核，不能以多数条重复记录制造置信度。

只用 dev 选择切点，冻结后用 test 报告。保留 lexical、BGE-only、旧规则 B 与新槽位候选四个对照。precision/recall 使用 Wilson 区间，macro-F1 增益按文档家族 bootstrap。样本不足、缺少类别、区间过宽都不能自动过门槛。claim 四分类只用于离线/影子。

## P3 分歧

`.runtime/semantic-slot-v2/shadow-disagreements-final-20261002.json` 提供槽位、原始裁判输出和所见上下文。审核标签为 `lexical_early_stop`、`semantic_false_rejection`、`splitter_error`、`judge_context_omission`、`citation_mismatch`、`missing_evidence` 或 `unclear`。这些是待验证的归因，不是预设结论。

## P4 问题及答案审核

`fixtures/unseen_v2/review.json` 审核 70 道问题及 2 个受控安全场景，另存 `reviewed.json`。逐项确认题目、必答事实、必要证据、条件真值、版本范围与 scorer 公平性，在 `checks` 对应项记录判断。每个输入同时绑定完整语料字节哈希。标签必须全部为 approved 才能冻结和入库；改题或改原文后需重新审核。

隐含链接问题不能直接给出第二跳对象；一次检索获得全部必要证据时允许合法停止。证据里出现未执行分支的规则，不等于最终答案违反条件。撤权后的数据出现在对用户可见内容中才是 ACL 泄露。

运行后生成匿名的 `.answer-review.json`。另存审核文件，为每个 `required_slots` 填写 `slot_verdicts`，并审核 `citation_valid`、`condition_compliant`、`version_scope_valid`、`status_valid`；完整支持才可标 correct。可填写 `early_stop_error` 为 true/false，无法判定或不适用为 null，报告保留分母。单纯字符串命中不能通过正式质量验收。
