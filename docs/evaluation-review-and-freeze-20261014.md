# 评分审核、External 独立标注与冻结准备

更新：2026-10-14。权威结论见 PROJECT_REPORT §20.39。当前是工具和交付准备完成，人工标注与正式冻结未完成；不运行 Core。

## 评分口径

新规范为 `benchmarks/enterprise_rag/v1/scoring-rules-v2.json`，协议为 `enterprise-benchmark-review-v2`。规范文件进入既有 `run_unseen_benchmark.inputs()` 的冻结哈希；旧 manifest、题目、标注、答案及审核包不改写。

| 要核对的事项 | 正确评分依据 |
| --- | --- |
| 同义表达、格式差异 | 比较语义和对应原文，合法同义答案可通过；字面匹配只作筛查 |
| Yes / No / Unclear | 单独核对最终结论、正文和题目命题；正文列出正确事实不能救回错误结论 |
| 事实完整性 | 每项必要事实都在最终答复中；仅出现在上下文中不算已回答 |
| 引用支持 | 每条断言绑定的实际引文支持主体、属性和关系；逐字出现不等于支持 |
| 日期、数值、单位 | 核对事件与发表时间、范围、数量级、单位及比较所需的全部前提 |
| 人物、来源、条件 | 保留说话者与观点/指控限定；指定来源和适用分支必须正确 |
| 验证器拒绝/接受 | 分别重算机械代码、检查原文语义；机械接受不认证语义，拒答需要具体依据 |
| 不确定标注或审核 | 无明确失败证据时严格成绩为 null；明确失败为 false；全项成立才为 true |

`assess()` 复用既有事实/断言/引用及五项严格检查。`no_critical_error` 包括最终方向、完整回答、单位及主体等重要错误。新工具允许严格检查显式为 null；缺少检查字段仍拒绝。若一个必要项已明确失败，即使另一个项不确定，该答案仍严格失败；没有确定失败而存在必要项 unknown 时保持 null。

`report()` 的 Token `sum` 只有全部位置计量完整时才填写；`observed_sum` 是实测位置小计，必须带 observed/total。显式测得的 0 保留。P50 为中位数，P95 为 nearest-rank，统计实际记录的任务墙钟时间并披露分母、失败与背景负载。尚未记录的成本不能从延迟或成功调用数推断。

主表还要求完整 Core 70、可验证冻结登记、冻结语料/评分规范、当前审核协议及协议允许的 GPT/人工署名审核。匿名、换模型或工具校验通过不证明独立性。旧 v1 审核包可按原投影校验历史输入，但不能直接取得新协议主表资格。

## 本轮逐题材料

本地 `.runtime/evaluation-trust-20261014/v1/per-case-scoring-audit-v2.json` 保存原始答案、参考标准、字面与旧模型评分、建议评分、原始引用、机械检查及理由：308 个历史诊断位置，另含 §20.37–20.38 的 48 个位置。§20.36 的 28 个判断、4 个修复及4个拒答位置另存原始审计副本。位置有重合和重复答案，不当作独立题目数。

[公开派生汇总](../artifacts/evaluation-trust-scoring-audit-20261014-v3.json) 只含 ID、哈希及诊断。30 格字面/模型分歧不能当作人工确认错误；三格有源文依据的建议重判仍为模型辅助、human_confirmation=pending。原七道 Dev 标注 pending 不受旧模型接受影响。旧文件、旧负结果、初版哈希及署名范围纠正记录全部保留。

Dev v6 同一原始答案与审核的 188 个位置，修改前后严格判定零变化，与历史报告也一致。RAG 原 77,362 Token 是 46/47 位置的小计，新完整总量为 null；其余三方法完整 Token 总量不变。全量逐条元数据为125份GPT、63份Claude，均非独立，不能按历史文件名将混合来源统称一个模型；实际来源仍需追溯，不据本轮复算宣称补做了整套GPT审核。

## External 独立审核交付

两份包在本地 `.runtime/evaluation-trust-20261014/v1/external-independent-review/reviewer-1/packet.json` 与 `reviewer-2/packet.json`。每份包含150题和609份实际源文，opaque 随机题目/来源 ID，不含系统答案、旧参考答案、方法、组别或预期实验结论。各题 candidate_source_ids 继承上游引用选择，不独立认证这项选择；完整源文目录可供寻找替代证据。

只交付相应 reviewer 文件夹。`coordinator/` 中的 ID 映射、原标注、分歧表和审计不得交给首轮审核人员。准备不是发送，也不代表已经有审核者或已收到结果。当前独立人工标签为 **0**。

每位真实审核者独立填写 reference_answer、answerable、必要事实及其 source_id/逐字原文引文、条件、时间范围、来源要求和理由；不适用字段明确写 none，未解决项写 unclear。完成时记录姓名、时间、reviewer_type=human、independent=true，并声明没有看过系统输出、已有标注或另一位审核者结果。

回收只读验证，不覆盖原包：

```bash
.venv/bin/python scripts/prepare_source_human_review.py \
  --original .runtime/evaluation-trust-20261014/v1/external-independent-review/reviewer-1/packet.json \
  --completed PATH_TO_NEW_COMPLETED_LOCAL_PACKET.json
```

`validate_external_review()` 验证完整题目覆盖、未改变的源文/题面、独立性声明、显式范围及事实引文。`compare_external_reviews(first, second, key, originals=...)` 校验两份原包哈希和两位不同审核者，产出字段分歧；字符串分歧只作筛查，语义是否等价交给真实裁决者。即使一致也不自动写入 gold。

协调者另存完成包、两份完成包 SHA-256、分歧、真实裁决者身份/时间/理由、原标注→建议标注的逐项差异、接受或拒绝决定。裁决模板留在 coordinator；未裁决不修改现有 suite，不写 reviewed.json。需要修订时另建版本和登记，不覆盖旧任务/标注/审核。确定标注后才能用盲化的系统答案做事实、引用和严格成功审核；首轮标注人员仍不得看系统输出。

## 冻结清单与运行边界

[最终只读清单](../artifacts/benchmark-freeze-readiness-20261014-v6.json) 保存数据原始 SHA-256、当前代码、规范、依赖锁、配置与已安装库版本。它没有向数据库/搜索/模型服务发请求，没有写 freeze.json，也没有启动评测。

| 项目 | 当前状态 / 正式冻结所需 |
| --- | --- |
| 数据与用途 | Dev47 可开发；Core70 全部保留测试；Security16 单列；External150 不调参，已历史使用，重复运行不是首次 unseen |
| 数据版本 | manifest、各 split tasks/documents/review、源码及 requirements.lock 哈希已记录；真实索引、版本、ACL 与向量尚需既有 indexed_corpus 入口核验 |
| Legacy | 7B、hybrid/rule、task contract 开启；配置 top_k=4，既有 QA 多来源取8块、短问取6块，检索上限8块/5000 token，不能把历史八块上下文写成所有题的默认。生产配置未改。生成 temperature=0、seed=42、num_ctx=8192、默认700输出 token；Legacy 判定4096上下文/60输出 token，仍须冻结实际逐请求参数 |
| 矩阵 | 已注册八路矩阵；正式首次运行前冻结完整矩阵，不看结果增补消融 |
| 审核 | Core/Security 原模型题目标注有批准记录、非独立；External 新协议及独立人工标注/裁决未完成，历史答案来源冲突与 GPT 审核缺口未解决 |
| 预算与失败 | 当前任务600秒、模型传输180秒；policy/judge/generation 累计调用上限、judge token 预算见清单。有限修复/重试由源码绑定，不重抽失败答案 |
| 可复现环境 | Python、库版本、锁文件已记录；实际模型权重/runtime、交叉编码器文件、索引及租户权限需既有 configuration()/indexed_corpus() 正式冻结 |
| 中断续跑 | 既有 ledger/CheckpointStore，仅同冻结状态、同输出 --resume；未观察到的答案/Token 保持 null，失败记录保留 |
| 当前阻塞 | External 审核与裁决、评分规范正式冻结、环境与索引/权重验证、最终冻结快照选择、历史审核身份追溯 |

只读清单可以使用新输出名复查：

```bash
.venv/bin/python scripts/benchmark_package.py readiness --output NEW_READINESS_PATH.json
```

正式入库、freeze、运行和 report 的现有命令仍在包 README；这些是以后满足门槛后的操作，不是本轮执行。当前 formal_core_ready=false，不自动启动 Core、不创建 Git 提交、不发布。
