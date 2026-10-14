# Evidence Delivery 后续技术债

更新：2026-10-14。来源为 §20.38 及其原始产物。本轮只登记，不改生成、判断、检索或校验生产函数。原文、答案与第三方题面只在本地。

| 工作项 | 已确认依据 / 边界 | 受影响的真实函数 | 可验证验收条件 |
| --- | --- | --- | --- |
| ED-01：SF-64 选中内容未交付 | 最终 `relevant_doc` 的必要段被 `top_k_full` 排除；更早 document_quota 不是最终阶段的完整解释 | `app.evidence_selection.select_relevant_evidence()`、`_Pool.assign()`、`app.qa._answer_question()`、`app.passages.prepare_passages()` | 在同一候选池/权限/版本/预算下可追溯每块候选→选中→模型输入；确认必要段是否交付，既有正确题与必要事实不丢失；不硬编码题目或 ID |
| ED-02：DD14 名录未交付 | 名录在原语料，最终上下文缺失；旧候选未保存，检索/选择阶段 unknown | `app.retrieval.retrieve_authorized()`、`app.evidence_selection.build_lanes()`、`select_relevant_evidence()`、`app.qa._answer_question()` | 新独立工作先保存原候选、排除理由及实际请求上下文；定位真实丢失阶段，unknown 不推成检索失败；重核权限和活动版本 |
| ED-03：补证后事实遗漏与单位错误 | A/B 证据完整性改善，无严格净收益；源文已给齐仍有遗漏和数量级错误 | `app.clients.Models.generate()`、`app.answer_contract.quantitative_support_issues()`、`app.qa.claim_validation_issues()` | 逐项核对必要事实、数值/单位角色和自己的引文；引用存在不能算语义支持。将 DD24 类机械接受但数字未获自身引文支持的边界单独复核；不据单题放宽校验或删事实 |
| ED-04：来源、条件及最终方向 | DD14/DD17 引错适用分支；SF-4345 对象/来源错误；直呈暴露错误 Yes | `app.answer_contract.publication_binding_issues()`、`verdict_proof_groups()`、`app.qa.answer_verdict()`、`app.clients.Models.decide_verdict()` | 实际条件用已引用前提链复算；对象、来源、观点主体与结论逐项一致；不能借未引用正确分支补救；方向正确但事实缺失仍失败 |
| ED-05：直呈删除关系 / 草稿缺测 | SF-66 从原正确 No 退为 unclear；SF-4345 provider 成功后预算失败，草稿未保存 | `app.answer_quality.bind_source_text()`、`app.qa.answer_verdict()`、`app.execution_budget.bounded_model()` | 关系结论、原必要事实及正确引用无损失；明确区分 provider 成功与业务完成。保存真实返回/预算事件，不重建缺失草稿；缺测成本不补0 |

## 复现材料与步骤

1. 固定 `.runtime/evidence-delivery-20261014/v1/inputs.json`、`delivery/raw.json`、`support/raw.json` 和两个注册；逐字 SHA-256 见 `artifacts/dev-evidence-delivery-20261014-v1.audit.json`。复用本地 `failure-attribution.md` 与 `counterfactual-and-candidate-review.md` 的原文、ID、条件核验和观察限制。
2. 只读复查 SF-64 / DD14 在输入中 current/additions、实际 raw.sources、drafts、payload 和候选/最终选择轨迹。候选来源和原文证明见 `injection-source-checks.json`，SF-64 最终排除链见公开归因产物；DD14 旧候选缺失继续 unknown。
3. 以 `.runtime/evaluation-trust-20261014/v1/snapshot-before` 的函数字节及注册哈希保存旧执行语义。本轮评分工具修复后，当前源码整体哈希变化是预期的；不能用当前代码冒充旧冻结运行，也不能为“复现”重跑旧失败位置。
4. 仅对保存的草稿重新调用 `GeneratedAnswer.model_validate()` 与 `claim_validation_issues()`；引用逐字、数值/条件、语义支持三项分别记录。真实网络复现若另立工作项，必须新注册/新文件、只用 Dev，不能改原运行。

## 开发前置条件与停止条件

先完成可信参考标准与归因；提出有一般适用性的最小改变后再预注册独立 Dev 对照。沿用当前业务权限、版本、累计预算、事实完整性与安全边界，不新建图引擎。验收至少要求严格净收益、零原正确题退化、零必要事实或支持损失，且无新增数值/单位/主体/时间/条件错误；预算、成本与失败口径完整。

未达门槛即保留负结果，不继续对少量案例重复 Prompt / Structured Verdict / Grounding 调参。上述工作项未实施、正式 Core 未运行；本轮工具准备完成不意味着这些答案质量问题已经解决。
