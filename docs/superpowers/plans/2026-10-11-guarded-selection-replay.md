# Guarded selection replay implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development and test-driven-development. Current directory is authorized; no commit, push, Core access or historical artifact mutation.

**Goal:** Measure the new selector alone on all Dev47 before changing generation or Agent policy.

**Architecture:** Reauthorize the existing depth50 candidate pool and replay actual same-query BGE scores from the frozen October10 run. Compare recorded relevant_doc, reproduced rank_floor and guarded using identical pools, caps and mandatory source backbone. Unknown cached pairs fail; this stage performs no new search or model inference.

**Tech Stack:** Existing Python replay/scoring helpers, PostgreSQL readonly, OpenSearch identity ledger.

## Task 1: Reproducible score replay

Files: create `scripts/run_guarded_selection_dev.py`, `tests/test_guarded_selection_replay.py`.

- [ ] Add RED tests for query/identity/text mismatches, missing/nonfinite/duplicate scores, and replay preserving actual ranking. Cache keys use the existing `cached_rank` identity tuple. A changed query or row must raise rather than fabricate or reuse another score.

```python
with pytest.raises(ValueError):
    replay_rank('changed query', original_rows, observations)
```

- [ ] Implement narrow cache extraction from the original aspect ranking trace and matching private pool rows. Freeze public parent, registration and each private context file byte hash before execution. Require complete valid Dev47 and the exact task order.
- [ ] Compare `baseline`, `rank_floor`, `guarded`. Runtime takes question, authorized rows and contract only. Required IDs come from the existing backbone and original trace, never labels. Validate reproduced rank_floor context IDs/order against the stored aspect context before evaluating guarded. Retain unresolved/history/unsupported decomposition fallback reasons.
- [ ] Reuse `validate_suite`, `validate_parent`, `validate_annotation`, `contract_for`, `policy_caps`, `rebind_rows`, `rescore`, `diagnose`, `summarize_revised`, `reserve_output`, `_publish`. Set document_quota=None as the original experiment did; retain source quotas, eight chunks and 5000 tokens. DB transaction is readonly; check corpus/index/source, input bytes and code drift before and after.
- [ ] Private outputs contain whole original rows and traces. Public outputs contain only existing safe telemetry, IDs/hashes/counts; no questions, source prose or reviewer reasons. Persist failed/interrupted results with null observations, preserve original artifacts. New physical search/model calls are explicitly zero; previous score acquisition costs are provenance, not new calls.
- [ ] Run `.venv/bin/pytest -q tests/test_guarded_selection_replay.py`, Ruff, spec review then quality review. Execute only after both reviews pass and the registration is saved. Report per-reference native losses and external delivery differences; selector deployment requires no native losses and a demonstrated external benefit, not an aggregate trade.

## Execution inputs

Suite `.runtime/benchmark-package/v1/dev/tasks.json`; selection `benchmarks/enterprise_rag/v1/dev/selection.json`; parent `artifacts/retrieval-quality-dev-20261009-v4.json`; original relevant_doc `artifacts/dev-evidence-relevance-20261009-v1.json`; original repair `artifacts/dev-external-repair-20261010-v3.json`; original private contexts `.runtime/external-repair-20261010/frozen-code-v5/.runtime/repaired-contexts/4e0967aa58cd4dd1bd8c5da092e5c589`; overlay `.runtime/dev-evidence-repair-20261009/accepted-annotation-v1.json`; manifest `benchmarks/enterprise_rag/v1/dev/evidence-20261009-v2/manifest.json`.

New registration `benchmarks/enterprise_rag/v1/dev/evidence-first-20261011/selection-input-v1.json`; output `artifacts/dev-guarded-selection-20261011-v1.json`; private output root `.runtime/evidence-first-20261011/selection`.
