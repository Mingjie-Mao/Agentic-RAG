# Bound citation validation implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development and test-driven-development. Execute after slot generation. Current directory is authorized; no commit, push, Core access, default activation or historical artifact changes.

**Goal:** Measure whether an existing source-bound semantic verifier detects true assertion/citation mismatch and omission, and account for its errors and cost.

**Architecture:** Reuse `Models.audit_bound_answer`, `BoundAnswerAudit` and the existing source-entailment rubric. Mechanical quote existence, original source identity, ACL and version checks precede any model audit. Semantic results are a separate experimental/shadow observation, not automatic production truth; preserve the existing production calibration gate.

**Tech Stack:** Existing answer models, ExecutionBudget, model transport telemetry, readonly source reauthorization.

## Task 1: Controlled bound verifier

Files: create `scripts/run_bound_citation_dev.py`, `tests/test_bound_citation_dev.py`; fixtures under `fixtures/source_contract` for authored contrast cases, actual third-party answer/source observations under ignored `.runtime`.

- [x] Add RED tests that a nonexistent quote or mismatched ID does not reach the semantic model; each valid assertion passes only its own original quotes and legitimate source metadata. Audit cannot borrow another claim's quote or source index.
- [x] Add tests for unsupported and incomplete verdicts, dependency/format/budget failure remaining null, and preserving original answer payload byte hashes. Reuse existing validation APIs, not a second entailment implementation.
- [x] Add a narrow `audit_answer(question, payload, evidence, models, cfg)` helper. Reconstruct short aliases from server-validated source IDs; reject malformed bindings before dispatch. One judge call, registered output/elapsed/judge-token limits, actual calls/failed attempts/tokens/latency. Audit result is shadow only and no assertion is rewritten into a success. Conditional and temporal checks remain required in external task review.
- [x] Preregister authored paired contrasts covering exact support, wrong value/unit/subject, background, modality, attribution, incorrect time scope, omission and legitimately jointly supported comparison. Freeze case bytes, rubric/code/model digest and thresholds before inference. Labels are assistant-authored examples, not human gold or representative Dev accuracy.
- [x] Run actual component inference and then the same verifier over every applicable generated Dev answer from Stage2. Report false accept/reject/unknown/operational failures separately, all model costs, and method-blind task-review agreement where available. A component contrast success alone cannot qualify a headline or deployment.
- [x] Run `.venv/bin/pytest -q tests/test_bound_citation_dev.py tests/test_answer_quality.py tests/test_external_answer_generation.py`; spec then quality review. Preserve `require_audit_gate` and semantic defaults; do not manufacture a production gate from self-reviewed examples.

## Output policy

Fresh registration and result names under `benchmarks/enterprise_rag/v1/dev/evidence-first-20261011` and `artifacts`, respectively. Public output is IDs/hashes/counts only. Complete failures/interrupted calls stay visible with null scores; private source/reasons stay local. Semantic calls are additional measured costs and are not presented as the old zero-judge budget.

## Execution gates

Finish the Stage2 matched answer review before changing registered application or runner sources. Implement and review the narrow binding/budget helper first, using synthetic transports and assistant-authored contrasts only. Then implement and separately review preregistration, readonly reauthorization, complete Dev replay, immutable publishing and comparison with the frozen answer review. Actual inference starts only after both SPEC and QUALITY gates pass for that runner. This separates implementation checks from model-verifier performance; neither synthetic tests nor authored contrast labels count as reviewed Dev accuracy.

The registered audit is shadow only. It preserves the original answer and verdict, records completeness and assertion support separately, and cannot silently replace a failed answer with the verifier's conclusion. Exact duplicate inputs may share one physical audit only when registered before inference; disclose shared costs and retain every mapped answer position.

The caller supplies an explicit readonly reauthorization callback; do not attach executable scope authority to serializable Settings or implement another permission store. Check the complete trusted source records before and after dispatch. Token preflight uses the repository's cl100k_base estimate, including schema and framing; this is not the Qwen provider's exact context count. Keep conservative numeric budget bookkeeping separate from observed provider token counters, which remain null when missing. A successful schema-valid audit with missing telemetry is not a fabricated failed request; detected ceiling violations and actual transport/schema failures retain their consumed cost and unavailable outcome.

2026-10-12 component gate: SPEC PASS and QUALITY APPROVE after JSON type-drift regression repair,191 related tests/86 component cases. Approved component/test/fixture bytes preserved under ignored citation-component-code-v1. The separately reviewed register/execute runner is in progress; no actual model inference or default gate activation has occurred. Detailed runner requirements live in ignored citation-runner-spec-v1.md for the fresh implementer/reviewers.

2026-10-12 workspace rename: user confirmed the new root `/Users/mingjie/Desktop/Agentic-RAG`. A legacy path alias preserves access to historical absolute references. Historical bytes remain unchanged; the new runner must verify canonical file identity plus the complete original keyset and hashes before accepting either path spelling. Actual inference remains pending runner SPEC and QUALITY review.

2026-10-12 runner gates: SPEC PASS (363 related synthetic tests) and QUALITY APPROVE after strict authored-label/public-type and cached-reauthorization interruption fixes. Mandatory suite: 268 passed; the re-review added 9 independent synthetic probes. Metadata attestation prepared without model calls, then citation-input-20261012-v1 froze 206 source files, 94 Dev +24 authored positions, common budgets, actual judge identity and complete input byte bindings. A new 210-file ignored code/input snapshot was saved. Actual shadow replay started as ca934792b6f44ce7b7d52e65a526f96f; performance and agreement are pending completion. Source, fixture and settings remain frozen throughout replay and comparison.

2026-10-12 actual replay complete:118/118, valid, all five drift checks false. Dev66 semantic observations /4 mechanical /24 not applicable; authored1/11 false accept,2/12 false reject,1 mechanically blocked negative. Strict model-reference agreement23/27 per arm,13/40 unavailable per arm, not human accuracy.71 physical judge calls,66,302 provider tokens; no production gate activation or answer/retrieval gain. Stage3 is complete and source hold released for the registered Stage4 work.
