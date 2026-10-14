# AI-assisted 评测与正式冻结清单（2026-10-14）

本页执行用户新授权：独立人工金标准是可选未来增强项，不再作为 Core 启动或验收的强制条件。当前协议是 `evaluation-policy-v2-ai-assisted.json` 与 `scoring-rules-v3-ai-assisted.json`。旧 v2 规则、§20.39、原人工审核准备说明和历史成绩保留，以当时协议解读，不静默改写。AI-assisted 不是独立人工金标准；模型换名或隔离上下文不能证明统计独立性。

## 已完成的 External 审核

150/150 题先做源文参考判断；609 篇原始文章和全语料锚点检索用于核验。初始参考冻结后才读取原 gold 和系统答案。候选文章 ID 继承上游筛选，存在选择偏差。全文补查导致 3 个参考判断在答案访问后修订，原记录、理由和原文定位完整保留，**最终参考并非完全盲审**。最终 112 题可回答、16 题支持语料范围拒答、22 题 unresolved。

Codex 核对全部 300 个历史 RAG/Workflow 位置：296 个唯一元数据匹配；4 个原始运行身份不唯一，其全部 10 个匹配候选均已核对，语义、完整性、严格结果及 Token 总和一致，按结果不变项汇总，未挑选最佳候选。历史产物没有答案哈希，因此恢复身份属于元数据推定，不能声称原运行密码学绑定。

23 个初始参考争议使用新的 source-only 上下文交叉审核；4 份引文不满足原文精确匹配，未采纳。另对 8 个答案不确定项使用隔离上下文交叉检查。低参数模型错误接受任职归属、年龄等推断，Codex 按真实原文拒绝采纳，保留分歧与 unknown。先前超时、截断 JSON 和中断调用均保留，没有把失败意见补成正确标签。

主审核实际身份：Codex，会话指令标识 GPT-6 家族，精确 serving variant/digest 不可用。交叉审核实际为 qwen2.5:3b，digest `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`，Ollama 0.11.10。qwen3.5:9b 的三次答案审核输出截断，未采用。全部 `human=false / independent=false`。

| AI-assisted 指标 | RAG | Workflow |
| --- | ---: | ---: |
| 语义正确 | 77/126，61.1%；24 未决 | 80/124，64.5%；26 未决 |
| 必要事实完整 | 77/133，57.9%；17 未决 | 80/134，59.7%；16 未决 |
| Claim 自身引用链完整支持 | 57/182，31.3% | 55/190，28.9%；1 未决 |
| 严格成功 | 29/143，20.3%；7 未决 | 25/143，17.5%；7 未决 |
| 全 150 严格成功范围 | 19.3%–24.0% | 16.7%–21.3% |
| Yes/No 方向 | 31/55，56.4% | 30/55，54.5% |
| 相同 55 题多数类基线 | 35/55，63.6% | 35/55，63.6% |

历史 binary 题集合仍是 77 题；55 题有可用的 AI 二值参考，其余 22 个参考未决或不适合强制二值，完整 ID 清单保留。无结论、拒答或 unclear 在已明确二值参考题中不计为正确。不能将这组 55 题与原 77 题旧准确率直接比较。22 个源文未决与语义/严格未决不是同一分母：确定的错误引用或其他失败可以使未知参考题严格失败；必要项 unknown 且无明确失败才保持 null。范围是未知项全部失败/全部成功的边界，不是置信区间。

失败标签可重叠：引用支持失败 RAG81/Workflow89题、事实遗漏32/30、指定来源27/36、错误拒答21/18、日期范围17/13、错误 Yes/No17/15。保存原因和主体、时间、数值单位、条件、模态逐项判断。正确实体名也要有完整来源约束证明；原文实质标题和发布日期可支持其自身事实，泛主题标题不能替代事实证据。引用空白格式归一化后无真实原文不匹配；不能把换行格式差异误判为伪造引文。

旧字面 true→AI语义 false 分别14/13题，旧字面 false→AI语义 true分别3/6题。原字面101/150不重写，也不能当作严格成功。Dev188位置用原 v2 投影及原混合模型审核重放，严格结果零变化。

## 评分及成本口径

v3 在审核前冻结，SHA-256 `04435dbdd52e0072bc5a873460c8977ac8b7a610d21b5cb828f0119e30a00326`。严格成功同时要求必要事实、断言与引用支持、无关键错误、正确条件/版本/拒答、合法执行及规定安全事件；明确失败优先于 unknown。保留原 gold，只在 AI sidecar 使用新判断。没有模型多数表决自动覆盖 gold。

复用已有 `assess()`；v3 机械校验保留引用 ID、数量及字面引文绑定，数值/日期等语义规则由保存的审核判断决定，避免当前生产验证器无记录地否决历史冻结答案。v1/v2 评分和生产验证器不改。审核投影去除固定 Agent 诊断消息和实现标题后缀，原始答案保留。

历史答案 Token：RAG 测得360,901（150/150），Workflow已知小计387,077（149/150），总量null。仅覆盖已保存的生成/判断计数，embedding/rerank未计量；完整历史模型调用数不能全部恢复，unknown。历史墙钟 P50/P95：RAG53.46/77.30秒，Workflow53.98/77.35秒，n=150，P50中位数/P95 nearest-rank，不是本次审核耗时或重跑速度。

本地审核保存38个请求响应，其中37个具有完整Token：44,450输入/10,139输出；另有一个保存的超时与两个中断在途调用缺计数，完整审核Token总量null。两次玩具速度探针另外保留。Codex本会话逐题Token/延迟不可用，null。

## Core readiness：仍未满足

最新 [readiness v8](../artifacts/benchmark-freeze-readiness-20261014-v8-ai-assisted.json) 没有人工标注阻塞。规则冻结和150题AI审核已通过；原始/候选答案、修订、交叉分歧与未决清单通过哈希链校验。Core/Security 原有题目标注批准保持模型身份；External 新参考使用独立保存的 AI sidecar，不覆盖旧 gold。

| 冻结项 | 当前状态与具体剩余条件 |
| --- | --- |
| 数据版本/SHA-256 | enterprise_rag/v1；清单逐文件列 manifest、tasks、documents、review及上游指纹，第三方原文仅本地 |
| 数据隔离 | Dev47可调参；Core70全部留作正式测试；Security16单报；External150已用固定比较，无新增Core答案/结果访问 |
| 评分版本 | v3正式冻结；旧v2保留；AI-assisted身份、原文依据、unknown口径明确 |
| 模型与依赖 | 真实 metadata 检查取得生成/embedding/policy/judge digest、Ollama版本、重排权重及库版本；还需绑定最终冻结快照 |
| 索引/ACL | External609文档/24,911chunks可读，索引SHA已保存；Core首个所需文档及Security首个所需文档未发布，尚不能验证完整索引/权限快照 |
| Legacy配置 | 原默认保持；温度0/seed42不是确定性保证；实际请求参数、top_k变化、context/预算随最终代码冻结 |
| 随机/重试/失败 | 现有调用预算、one-shot ledger、CheckpointStore续接；失败尝试计成本，同快照同输出恢复，不通过重采样挑答案 |
| 正式快照 | Core/Security/External尚无当前协议环境freeze；需选定不可变代码、数据、索引、模型、参数及原八路矩阵，运行前再次校验 |
| 安全验收 | 原跨租户、组权限、检索后撤权与版本要求保持；针对性逻辑回归不替代最终冻结环境的正式安全验收 |

本轮只读取模型元数据和已有索引，`model_generation_calls=0 / index_mutations=0` 指环境预检，不包含上述审核模型请求。没有正式Core运行或写入其运行ledger。

复核命令：

```bash
.venv/bin/python -m pytest -q tests/test_external_ai_audit.py tests/test_evaluation_trust.py tests/test_benchmark_package.py tests/test_external_answer_review.py tests/test_external_independent_annotation.py tests/test_execution_trace.py tests/test_external_validation.py tests/test_external_repair_runner.py
.venv/bin/python scripts/benchmark_package.py readiness --output .runtime/readiness-new.json
.venv/bin/python scripts/benchmark_package.py readiness --verify-environment --output .runtime/readiness-env-new.json
```

环境预检需要本地配置的服务启动。保存的审核可用本地 `compile_final_audit.py` 审阅/重放计算；因历史产物不可覆盖，重新汇总时先选新版本输出名。复用 `run_unseen_benchmark.inputs/configuration`、`indexed_corpus` 和现有冻结/运行ledger完成后续快照；正式执行命令仅在明确授权与全部门槛通过后使用，本轮不启动。

原文、问题、答案、原gold和完整判断位于被忽略的 `.runtime/external-ai-review-20261014/v1`，公开[审核汇总](../artifacts/external-ai-assisted-review-20261014-v1.json)只含ID、指标、哈希与局限。历史、负结果、失败日志及修改前后快照均保留。Evidence Delivery技术债继续按[原独立工作项](evidence-delivery-debt-20261014.md)推进，不开发新生成补丁。
