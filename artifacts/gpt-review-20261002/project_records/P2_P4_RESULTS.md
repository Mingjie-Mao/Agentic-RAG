# P2 / P3 / P4 实现与验证记录（2026-10-02）

工程候选、审核资产、开发烟测和验收拦截已实现；**独立人工校准及正式验收尚未完成**。后续修复及最新状态见文末：已补齐隔离的合成中文 dev 输入，独立人工标签仍为 0，语义控制默认关闭，新 70 题未冻结、未入库、未运行。没有提交、暂存或推送。

## P2：槽位评估与校准流程

复用现有 BGE、EvidenceJudgment、裁判传输和预算计费，在 `app/semantic_evidence.py` 增加 SlotEvidenceEvaluator。旧 v1 规则及结果保留，不把新行为重算成旧成绩。

- 四个事实状态与 evaluation_validity 分离，技术失败返回 unknown；统一 aggregate_slots 决定 complete/partial/missing/contradicted/unknown/inactive，忽略自报 confidence。
- 必须引用实际展示的原文；高相关门槛检查**实际引用**，不能借未引用的高分片段。重复/不存在的引用、非原文引文、遗漏槽位、范围不明都不能当作完成。
- 多槽位轮流分配上下文；同范围冲突需明确范围依据；claim 只能使用自己的引用。完整原文提供给裁判，不沿用旧 1,200 字截断；调用前按 UTF-8 字节上界检查 8,192 token 容量并预留生成/模板空间，超限返回 unknown，不让服务自动截断后误报完整。
- 请求内缓存绑定槽位、内容、版本、模型与评估器配置，缓存命中前再次授权；技术 unknown 不作为有效事实缓存。
- 可完整匹配的单属性事实请求复用 P1 唯一值绑定，跳过 judge；复合要求、断言忠实度、否定/歧义值不能走该短路。单独回归该优化。

试点 **100 relevance / 80 coverage / 80 claim**。扩展第二版 **1,058 / 663 / 1,696**，包括实际 S3/MultiHop 拆分及已使用 Hard 的生成证据/断言。数量是审核候选数量，不是金标或独立样本数。原始第一版包和报告保留。按问题、文档及版本家族连接分区后，中文来源形成一个家族且没有独立 dev；增加旧 Hard 输出仍不能解决该来源隔离问题。

已实现审核完整性/争议复核、dev-only 阈值选择、分语料指标、四路组件对照、Wilson 区间、家族 bootstrap 和高精度启用门槛。运行时门槛绑定人工审核、校准、代码、模型及原始预测；任一变化会拒绝使用旧门槛。

真实本机组件烟测使用 4 个已用中英文输入：1 个 supported、3 个 unknown；unknown 原因为冲突范围未证明、引文不匹配、未经校准的相关性切点。新候选 judge 3 次，旧规则 B judge 3 次，均成功返回结构化输出。这是工程/故障路径验证，**不是 accuracy/F1 或质量提升结论**。

## P3：隔离影子、分歧和成对回放

新影子在主任务提交、主预算退出后运行，使用自己的数据库会话、截止时间、2 次 judge 上限与本地 sidecar。它不写回主答案、工具轨迹、证据或预算；嵌套主预算内不启动影子。权限复核后才复用已有观察，模型/代码/证据变化使缓存失效。旧影子保持其历史实现，实验使用新开关。

两次真实开发任务回放验证答案、任务输入/主预算、全部事件及工具结果哈希不变。第一轮每题各 1 次 judge，40.2 / 53.9 秒裁判耗时；这是本机开发运行，含并发候选的影响，不是正式成本比。后续回放补齐模型指纹、裁判原始输出与上下文，保留前一轮原始报告。

分歧包按六类原因和 unclear 待独立审核。已准备旧 35 题的同快照 lexical/semantic 成对开发套件及 1 个撤权场景；仅在 P2 通过后才允许冻结/运行。成本门槛独立检查平均 judge≤0.5、P50≤1.25、P95≤1.30、token≤1.30、权限与终止失败为 0；缺遥测不能算通过。P3 正式成对回放未运行，尚无收益/成本过门槛结论。

## P4：新正式基准与验收流程

已准备 **70 题、7 类各 10 题、80 文档 / 90 版本**，加 **2 个拒绝/撤权场景**。这是一份助手编写、待独立审核的候选基准；10 组重复结构的模板限制多样性，需要审核者检验代表性与题目必要性。

- 默认三路 RAG / Workflow / Hybrid lexical；语义 arm 需 P2/P3 双门槛。所有方式共享 P1 契约开关。
- 隐含链接题面不直接给出第二跳单元；不要求人为增加 hop 或额外工具调用。
- scorer 把字符串/文档命中保留为 provisional 诊断；正式答案必须独立审核每个必答槽位、引用蕴含、条件与版本范围。
- 条件禁项检查最终断言，ACL 检查对用户可见输出；新汇总分别报告。
- 冻结覆盖任务、语料、审核、全部本地 Python、依赖锁文件、实际 BGE 权重/tokenizer、Ollama runtime/model digest、阈值、开关与预算。保留轮换执行、不可变快照、检查点及同快照恢复。
- 质量门槛用完整性与 task success 的配对 +5pp/正向区间，报告逐类指标、真实事实槽位覆盖、适用分母、成本和安全；区间跨 0 不放宽旧门槛。

实测未审核的 v2 在冻结前被拒绝，未生成 freeze.json。另用 1 个已使用开发题验证新三路运行器，三路全部成功，独立原始结果和检查点保留；这 3 个 smoke run 不计入新的 70 题，也不能当作正式成功率。

## 验证资产与后续操作

- 单元回归：406 passed、25 skipped；`artifacts/p2-p4-unit-final-20261002-v2.xml`。
- 真实服务：`artifacts/p2-p4-final-integration-20261002.xml`（7 项），后续影子指纹/缓存检查 2 passed，另存 `artifacts/p2-p4-shadow-cache-final-20261002.xml`。
- 组件：`artifacts/p2-slot-development-smoke-20261002.json`。
- 影子：`artifacts/p3-slot-shadow-isolation-20261002.json`、`p3-slot-shadow-isolation-final-20261002.json`。
- 三路 smoke：`artifacts/p2-p4-three-arm-smoke-20261002.json`，对应 `fixtures/p2_p4_runtime_smoke/freeze.json`。
- 状态：`artifacts/p2-p4-validation-status-20261002.json`。旧 unseen v1 原始 JSON 不改写。

审核操作详见 [独立审核规范](P2_P4_REVIEW_PROTOCOL.md)。后续顺序：增加独立中文 dev 来源 → 试点审核/争议复核 → 扩展独立标签 → dev 校准并冻结 → 全部 test 组件评估 → P2 通过后成对回放与 P3 成本/质量审核 → 审核新 70 题 → 冻结、入库、一次正式运行 → 匿名答案复核与最终门槛。任何一环失败，保持实验开关关闭，报告拒绝或证据不足，不冒充验收完成。

```sh
# 读取原始包，人工另存 reviewed 文件；目前不能省略这一步。
.venv/bin/python scripts/predict_semantic_slots.py --source .runtime/semantic-slot-v2/review-r4/expanded-distinct.json --relevance-only --output artifacts/p2-slot-raw-dev-scores.json
.venv/bin/python scripts/semantic_slot_eval.py calibrate --source .runtime/semantic-slot-v2/review-r4/expanded-distinct.json --reviewed .runtime/semantic-slot-v2/review-r4/expanded-distinct-reviewed.json --predictions artifacts/p2-slot-raw-dev-scores.json --output artifacts/p2-slot-calibration.json
.venv/bin/python scripts/predict_semantic_slots.py --source .runtime/semantic-slot-v2/review-r4/expanded-distinct.json --calibration artifacts/p2-slot-calibration.json --output artifacts/p2-slot-calibrated-predictions.json
.venv/bin/python scripts/semantic_slot_eval.py evaluate --source .runtime/semantic-slot-v2/review-r4/expanded-distinct.json --reviewed .runtime/semantic-slot-v2/review-r4/expanded-distinct-reviewed.json --predictions artifacts/p2-slot-calibrated-predictions.json --calibration artifacts/p2-slot-calibration.json --output artifacts/p2-slot-component-gate.json
# P2 通过后：成对开发套件 --freeze，再以相同参数运行。
.venv/bin/python scripts/run_unseen_benchmark.py --suite-dir fixtures/semantic_paired_development --freeze --output artifacts/p3-paired.json
# 新题独立审核通过后：--freeze、入库、运行；目前会被拦截。
.venv/bin/python scripts/run_unseen_benchmark.py --suite-dir fixtures/unseen_v2 --freeze --output artifacts/unseen-v2.json
.venv/bin/python scripts/setup_unseen_benchmark.py --suite-dir fixtures/unseen_v2
.venv/bin/python scripts/run_unseen_benchmark.py --suite-dir fixtures/unseen_v2 --output artifacts/unseen-v2.json
# 原始结果配套的匿名答案审核完成后才能做最终门槛。
.venv/bin/python scripts/finalize_semantic_gates.py p4 --results artifacts/unseen-v2.json --suite fixtures/unseen_v2/tasks.json --answer-review artifacts/unseen-v2.answer-reviewed.json --output artifacts/unseen-v2-final-gate.json
```

## 继续修复与验证（2026-10-02）

本节替代上文“中文 dev 缺失”的当前状态，保留前期烟测和原始实验结论。

- 新影子改为非阻塞后台诊断：单工作线程、最多一个等待任务、重复任务不排队，默认抽样率 0.1；总开关仍默认关闭。只传记录 ID，观察者自己打开数据库会话。失败不传播给已完成的请求，进程退出时可能丢失尚未完成的诊断。两路真实集成检查让裁判等待“主请求已返回”信号，确认主请求不等待影子。模型仍共享本机资源，异步与抽样不是正式 P3 成本过关的证据。
- QA 保存生成上下文中的全部 chunk ID；影子和 P2 审核导出读取实际上下文，不再把最终引用子集冒充完整输入。遗留 QA 缺上下文时记 unknown。R3 重新导出的覆盖候选为 693，R2 和旧结果均保留。
- 条件未判定的分支及未证明完整的历史版本链保持 unknown；主契约的未知分支也不再记录成 inactive。缓存命中恢复对应裁判上下文，字面短路清空旧诊断；短路不能绕过主体、版本或 allowed_refs。裁判/排序运行结束再次授权，期间撤权不接受或缓存支持判断。
- 组件预测逐条原子保存、进程锁与同快照续跑；中断尝试保留预算、标 unknown，不能悄悄重跑。完全相同的离线输入复用判定且不重复计模型成本；技术 unknown 不复用。校准前可仅计算 relevance，避免不必要的裁判调用。完整校准预测仍必须覆盖所有冻结输入。
- 对组件输入去重后计算统计和最低样本数；重复标签不一致要求独立复核。跨 relevance/coverage/claim 检查问题与文档家族，防止同来源伪装成不同 family 分别进入 dev/test。切点必须有限且 high≥keep。

中文开发来源为**助手编写的合成资料**，不是外部真实企业语料或人工金标。6 份文档在新租户 `slot-calibration-zh-dev` 通过正常解析、索引和发布链路入库，18 次问答导出 118 relevance / 138 coverage / 25 claim。整体来源在生成前指定为 dev；与旧来源没有文档或问题重叠，旧 test 行保留原样。其实际来源、代码和模型记录在 `.runtime/semantic-slot-v2/review-r4/runs.json`，文本配方在 `fixtures/semantic_slot_zh_development/source.json`。所有问答只用于开发输入构建，不据此报告准确率。

最新审核资产：

- `review-r4/expanded.json`：3,728 原始候选输入。
- `review-r4/expanded-distinct.json`：2,419 不重复输入，1,176 relevance / 831 coverage / 412 claim；保留 1,309 条重复输入到代表输入的映射，不生成任何标签。
- `review-r4/pilot.json`：100 / 80 / 80，共 260 条，全部 dev，避免试点为凑数读取 test。
- `semantic_review.html`：本地审核界面；原文以纯文本展示，标签没有默认值，实际审核者手动填写身份、独立性、标签和理由，导出保留原始 items/hash。未向该界面填入任何真实审核记录。通过 5 项静态表单逻辑检查；浏览器工具禁止 file URL，未做视觉预览，未采用绕过方式。

最终回归：**430 passed / 25 skipped / 9 deselected**（非 integration，`artifacts/p2-p4-followup-final-unit-20261002-v3.xml`）；真实服务 **9 passed**（`artifacts/p2-p4-followup-final-integration-20261002.xml`），其中新增完整/遗留 QA 上下文和实际裁判期间撤权。4 个真实 BGE 相关性预处理样本完成，judge 调用 0，仅为开发烟测。Ruff 与 diff 检查通过。期间自动审批曾因额度限制失败，之后按正常审批获批续跑和验证，没有绕过。

**未完成项仍明确保留：**独立人工标签 0；合成 dev 的代表性待审核；P2 正式校准与测试、P3 同快照成对质量/成本验收、P4 新 70 题和答案审核均未完成。旧语义影子每题约 30–37 秒不是已降低的正式成本，不能据异步处理声称成本或准确率达标。当前最新输入的 P2 拦截为 `independent review incomplete: 0/2419`；P4 仍拒绝未经审核冻结/入库。
