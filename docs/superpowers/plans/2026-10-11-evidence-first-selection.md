# Evidence-first selection implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for implementation, spec review and then quality review. Current-directory execution is already authorized. Do not commit, push, deploy or modify historical artifacts.

**Goal:** First repair evidence selection without mechanically reserving unverified background; then proceed, in the user's order, to slot-complete generation, source entailment verification and finally Agent-policy ablations.

**Architecture:** Reuse SupplementContract, assess_slot_presence and the existing capped allocator. Keep the historical rank-floor selector as an unchanged default/control. Add an explicit guarded experimental policy: reserve only coherent literal witnesses before relevance fill, mark unknown aspects as unverified, and prevent exact duplicate content from the same document/version consuming extra slots. Preserve mandatory IDs, lane feasibility, permission/version/source identity, contradiction visibility and conditional witnesses. This stage adds no model calls or orchestration framework.

**Tech Stack:** Python3.13, existing Pydantic contracts and token accounting, BGE reranker, PostgreSQL/OpenSearch readonly; official LangGraph remains the Agent executor.

## Stage1: Evidence Selection

Files: `app/evidence_focus.py`, `tests/test_evidence_focus.py`; a new separately registered Dev replay runner after implementation passes review.

- [x] Write failing synthetic tests: a numeric requested fact ranks behind irrelevant background, two explicit subquestions share duplicate background, unrelated unknown aspects do not receive mandatory reservations, and duplicate body text in different sources/versions stays distinct. Use the real selector and allocator with a deterministic ranking fixture; do not use Dev gold in runtime logic.

```python
contract = build_supplement_contract('白泽账务的超时阈值和青鸾网关的超时阈值分别是多少秒？', [])
chosen = select_aspect_evidence(contract, candidates, ranker=ranker, limit=2,
                               selection_policy='guarded')
assert {r['chunk_id'] for r in chosen.evidence} == {'ledger-value', 'gateway-value'}
```

- [x] Run `.venv/bin/pytest -q tests/test_evidence_focus.py`; observe the new policy tests fail before implementation.
- [x] Extend `select_aspect_evidence(..., selection_policy='rank_floor')`; validate the policy before dispatch. `rank_floor` retains existing behavior and telemetry. `guarded` uses existing structural/literal assessment to prioritize complete supported witnesses, verifies their actual selected-subset support, and does not represent ranked generic prose as proof. Missing/unknown/conflicting facts do not receive false support labels.
- [x] In guarded reservation/fill, avoid adding another exact whole-text duplicate from the same document/version/source identity. Do not merge source IDs, clip text, erase contradictory values or discard caller-mandatory IDs. Keep the original eligible rows and rejection diagnostics, including which duplicate was skipped. If a candidate's lane is infeasible, another recorded legal occurrence/candidate remains usable.
- [x] Verify two-question coverage, shared witnesses, infeasible caps, a rejected early unrelated aspect, source quota alternatives, table header/value dependencies, conditions/negation, stale ACL/version/SHA, distinct-source identical text, and mandatory duplicates. Run focused regressions and then fresh spec→quality review.
- [x] Preregister immutable Dev47 replay, baseline contexts and both selection policies with fixed8chunks/5000tokens; freeze input hashes, code, reranker observations and source identities. First compare candidates→context delivery under original/revised references, without additional search or gold-aware routing. Actual semantic/answer scores require named method-blind reviews. Failed/incomplete observations stay null and outputs use new names.

## Ordered follow-on stages

These are separate subsystem plans, prepared and executed after Stage1's reviews; they must not be implemented concurrently with it.

1. **Answer Generation:** First correct the evaluation entry: the previous context replay called `Models.generate` directly, while `app.qa` uses shared `contract_generate` when the task contract is enabled. Register separate direct-model replay and shared-contract controls so an evaluation-path repair is not credited to a new algorithm. Reuse `acceptance_items_override`, `build_contract`/`evaluate_condition`, server-owned source aliases and existing structured response APIs. Batch unsupported question slots in one budgeted call where feasible; preserve unknown conditions, period/source roles and explicit missing-slot status. Regression cases include a second question omitted, an active branch omitted and a current value used for historical scope. Fixed-active-context replay does not exercise production historical document fetching; report that limitation. Do not generate a hidden answer from benchmark facts.
2. **Citation Verification:** Reuse existing `BoundAnswerAudit` and source-entailment rubric; citation existence/ACL/version checks remain mandatory. Each assertion is reviewed only against its bound quotes and legitimate source metadata. Contributing premises may jointly support a relation; irrelevant SDK/background quotes may not. Register model identity, calibration, uncertainty and all physical/business model costs before controlled Dev tests. An uncalibrated shadow score must not silently become a production gate.
3. **Agent Policy:** Only after selection/generation/support verification, compare no-supplement control against fixed-query and intermediate-result supplement arms through official LangGraph. Register the complete ablation matrix and shared model/search/context budgets before execution. Evaluate semantic completeness, gain/loss, refusal, cost and stopping; do not require one expected tool path.

## Completion evidence

- [ ] Keep all previous local changes and artifacts; new third-party questions/source/review text stay in ignored `.runtime`.
- [ ] Run appropriate focused tests, final `make test`, docs/diff checks and historical byte invariants. Append report with actual measured results and exact experimental status.
- [ ] Adopt a production strategy or freeze Core only after the registered Dev adoption gate passes. Core70 remains unseen during tuning; public main accuracy remains pending until frozen and answer-reviewed Core.

## Checkpoint

Initial snapshot: `.runtime/evidence-first-20261011/initial-files.json`. Prior Dev primary strict24/40→25/40, original fact retrieval52/58 unchanged, revised external15/19 unchanged. The existing selector's aspect floor reserves a highest-ranked feasible row even when the aspect is unknown; relevance is not entailment. Existing source-entailment, answer contract and conditional primitives must be reused before adding anything new.

2026-10-11 continuation: latest saved selector regression 289 passed; independent spec review still requires conservative table/heading qualification handling. First implementer hit its usage limit; a fresh repair agent is completing this fix. No Dev replay or later-stage inference has run in this continuation. Historical snapshot check: 0 removed files, only selector/test bytes changed among 1098 initial files, no historical artifact changes.

2026-10-11 selection code gate: independent spec PASS and quality APPROVE after qualification/context/table-target fixes; focused regression 316 passed. Registered replay runner implementation is in progress; no new Dev delivery/accuracy claim yet. Default rank_floor and production configuration remain unchanged.

2026-10-11 Stage1 actual gate complete: valid full47 v2, all drift false. Baseline native45/45, accepted external15/19; guarded45/45 and14/19, so baseline retained. No accuracy/semantic-gain claim. Stage2 ordinary-slot opt-in implementation now begins after Stage1 reviews and completed replay.

2026-10-11 Stage2 actual gate complete: v1 literal_slots lost correct tasks and required facts; v2 literal_fallback finished94/94/no errors/all eight drifts false. Fresh method-blind62-answer review with exactly reused47 canonical evidence grades gives shared_contract29/40 versus candidate30/40, but one required-fact loss and52 versus53 generation calls. Frozen adoption false; retain shared_contract and stop Stage2 candidate tuning. Retrieval unchanged52/58. Results are named model reviews, not human labels or Core accuracy. See PROJECT_REPORT.md §23.2 and dev-slot-generation-review-20261011-v2.json.

Stage3 component now reuses the existing bound-audit API with explicit pre/post source reauthorization and one prepaid judge call. SPEC passed167 focused tests; QUALITY found JSON type drift hidden by Python equality. Same implementer is fixing that boundary before re-SPEC/re-QUALITY. No Stage3 actual inference or production audit gate activation yet. Stage4 remains pending in the user's order.

2026-10-12 Stage3 component gate complete: same SPEC PASS191 focused tests/86component and QUALITY APPROVE after type-sensitive JSON record comparisons. Original payload and judgment remain unchanged; pre/post scope failures retain costs, provider counters unknown remain null with distinct conservative budget bookkeeping. Approved three files archived under ignored citation-component-code-v1 before runner work. Separately reviewed preregistration/replay runner is now being implemented; no actual judge inference, production audit gate or Core activation yet.

2026-10-12 Stage3 actual gate complete:118/118, valid and all drift false.71 judge dispatches,66,302 provider tokens; authored false accept1/11 and false reject2/12,1 mechanical negative.94 Dev positions66 observed/4 mechanical/24 not applicable; primary strict model-reference agreement23/27 per arm with13/40 unavailable, not human accuracy. No answer rewriting or production gate, no retrieval/answer correctness gain. Source hold released; Stage4 experimental focus propagation, official one-active-arm graph and transport budget component now starts through a fresh implementer and independent reviews.
