# External evidence repair and answer validation implementation plan

> **For agentic workers:** Use superpowers:executing-plans, or subagent-driven-development when an independent worker is available. Track actual checks below; an unavailable review is not a passed review.

**Goal:** Repair query-derived source/date and aspect selection, compare all47 Dev tasks with unchanged legacy word-fourgram scoring plus sealed revised reference scoring and separate semantic scoring and generated-answer review, then freeze/run Core only after demonstrated Dev benefit.

**Architecture:** Keep official LangGraph and existing permission/version, ranker and answer-validation boundaries. Date interpretation and content aspects come only from the original question/current authorized metadata. Rerank and reserve evidence for question aspects within8 chunks/5000 tokens, with no gold/expected paths in runtime. Scoring/reviews run after execution using separate immutable local inputs. Historical files and unrelated changes remain intact.

**Tech stack:** Python3.13, Pydantic, existing BGE reranker, FastAPI answer validation, LangGraph, readonly PostgreSQL/OpenSearch, local Ollama; existing benchmark package freeze/scoring.

**Authorized execution:** Current directory; no new worktree. No Git integration or deployment. Existing user authorization to complete tests includes required benchmark setup at the formal stage; Dev repair/replay remains readonly. Code/source snapshots isolate concurrent changes. Failure outputs use new names. Do not manufacture the requested improvement: if comparisons do not establish it, report the measured result and the unmet condition.

## Task1 — Publication dates and source-clause attachment

Files: `app/supplement_contract.py`, `tests/test_supplement_contract.py`.

- [x] Add a failing synthetic test for one publisher with two coordinated dated reports, preserving two distinct clauses and date/document bindings:

```python
q = "Did Publisher P report on January 3, 2026, that Acme paid billions, and then on February 8, 2026, reported a lawsuit?"
c = build_supplement_contract(q, [document('jan'), document('feb', published='2026-02-08')])
assert [r.document_ids for r in c.source_requests] == [('jan',), ('feb',)]
assert 'billions' in c.slots[0].query and 'lawsuit' not in c.slots[0].query
assert 'lawsuit' in c.slots[1].query and 'billions' not in c.slots[1].query
```

- [x] Run `.venv/bin/pytest -q tests/test_supplement_contract.py`; verify the new behavior fails before editing implementation.
- [x] Implement a narrow coordinated-report parser in `_constraints`/request construction: explicit publication verb before the first date, `and then on DATE ... reported` inherits only the immediately preceding explicit publisher. Split clauses at the coordinated marker. Preserve strict before/after, occurrence identity and literal offsets. Unattached event dates, ambiguous alternatives/shared publishers, malformed metadata and explicit times remain unknown; never broaden scopes.
- [x] Add adversarial/date negatives; run contract and legacy retrieval tests. Root spec/quality checks or fresh reviewers must finish before Task2.

## Task2 — Question aspects and bounded complete extraction

Files: new `app/evidence_focus.py`, `tests/test_evidence_focus.py`; extend `agent/supplement_proposal.py`, `agent/supplement_repair.py` and tests only as needed.

- [x] TDD `question_aspects(contract)` returning ordered source-bound aspect queries from actual question clauses, with no gold inputs. Remove only publisher/publication scaffolding; preserve negation, units, actor names and all material topic phrases. A plain report clause and coordinated dated report remain separate; comma/conjunction lists retain the final requested aspect. Chinese recognized task slots reuse existing SlotSpec. Queries are bounded deliberately; no clipping of evidence text.
- [x] TDD `select_aspect_evidence(contract, candidates, ranker, caps)` using the existing BGE reranker per bounded aspect over compatible current-source candidates, deterministic ties, mandatory source witnesses and exact existing allocator. Relevance ranking is not semantic support. No added model judge, outside corpus evidence, scores compared across queries, or hidden model costs.
- [x] TDD complete source-bound extraction: make the legacy extractor's input clipping optional with default unchanged, bounded whole-passage payload preflight for the experimental branch; verify an entity after character700 can be extracted and must still bind to actual authorized bytes. Missing literal values/refreshed ACL/version mismatches remain rejections. Budget stays2 policy operations,1 extra search/arm,180 seconds,8 chunks/5000 tokens.
- [x] Verify selection preservation for contradictory numeric assertions, repeated publishers/date requests, background high ranks, multi-aspect single-source articles, stale permissions, infeasible caps and private/public trace projection. Run focused and legacy regressions before registered live work.

## Task3 — Registered Dev retrieval and generation comparison

Files: new `scripts/run_external_repair_dev.py`, tests; reuse existing immutable parent, baseline, overlay, manifest and metering helpers.

- [ ] Register before execution: fixed all47 IDs, baseline relevant_doc, aspect selection over cached candidates plus at most1 bounded question-derived supplementary query per arm, frozen union rule and every physical ranker/policy/search/generation cost. No task-ID routing. A baseline/candidate profile may be cached only after exact bytes/source/corpus reproduction.
- [ ] Execute readonly from hashed copied code/Dev source snapshots. Before changing any outcome claim verify source hashes, corpus/index, actual models, configuration and end drift. Store raw questions, contexts and answers in ignored runtime only; publish derived IDs/hashes/counts. Keep failures and missing results with nulls.
- [ ] Generate final answers for baseline and improved contexts using the same `Models.generate`/answer binding validation and explicit equal generation budgets; do not infer answer correctness from retrieval proxies. Retain all47 including history/no-reference tasks. If active-only retrieval cannot answer history, record actual output and score it; no exclusions.
- [ ] Create method-blind review packets using `benchmark_package_scoring.answer_review_packet`. Score the original26 upstream references with the unchanged multihop_retrieval_eval.fact_delivered threshold0.5 and document binding, separately from revised19. Also grade every original required fact (including retrieved-fact verdicts) with the existing semantic rubric and actual reviewer identity/provenance. Revised source45/accepted-upstream19 proxy scoring stays separate. Pending/unclear annotation cases must not become approved just to increase accuracy.
- [ ] Report the same original legacy reference delivery, semantic required-fact recall, revised reference delivery, generated-answer strict success, gain/loss task identities, physical costs and paired differences/intervals. Historical0.368/0.442 are not new-run baselines unless all original protocol/inputs are reproduced. No independent/human review claim unless actually obtained.

## Task4 — Conditional common-runtime integration and Core freeze

Files: shared retrieval configuration/hook and tests only if Dev demonstrates benefit; existing `scripts/run_unseen_benchmark.py`, `scripts/benchmark_package_runtime.py`, package registration/freeze metadata.

- [ ] Require positive audited Dev answer/retrieval evidence with no fabricated/null scores before adopting the treatment. If no benefit, preserve results, diagnose on Dev, make a separately registered revision; never inspect Core to tune.
- [ ] Before reading Core outcomes, freeze unchanged baseline control, shared improved RAG/Workflow/Dynamic/Hybrid and every ablation in one full method matrix. Extend only the narrow configuration override allowlist needed for the new shared strategy. Freeze data/annotations/code/dependencies/models/configuration/indexed corpus and source/permission identities using existing machinery. The existing unfrozen matrix is inspectable protocol metadata; Core questions must not be used during implementation.
- [ ] Run the registered complete Core70 matrix once from the verified immutable worker snapshot; use explicit resume only for the same interrupted registration. No post-test new arm or threshold. Separately report Security/External if run; do not mix their denominators.
- [ ] Review actual Core answers with explicit model/human identity and provenance, then compute Core Strict Task Success Rate and paired comparison. Claim only the measured magnitude/scope; an interval spanning zero cannot support a significant/generic improvement claim. Core failures or no gain are reported honestly, with no test-driven retuning.

## Task5 — Documentation and final evidence

Files: append new `PROJECT_REPORT.md` section; change README/site headline only if verified Core results qualify.

- [ ] Save all new registrations/artifacts and reviewer rationale, run `make test`, docs check, diff check and historical-byte/HEAD invariants. Record unavailable independent reviews distinctly.
- [ ] If audited frozen Core shows improvement, update README/site with comparator, sample size, strict metric, reviewer basis and exact result artifact links; otherwise retain pending/main table and report why the requested claim is unsupported. User authorization covers this conditional documentation update; no automatic deployment/push.
- [ ] Finish with a self-contained Chinese report covering actual fixes, three Dev metrics, Core status, tests, costs and remaining limitations.

## Checkpoint log

Initial files/HEAD hashes saved at `.runtime/external-repair-20261010/initial-files.json`. No Core freeze/registered-run exists at start. Anthropic/OpenAI API keys are absent; any review provider must be actually available and honestly recorded. Historical0.368/0.442 belongs to264 MultiHop questions and multihop_retrieval_eval.word-fourgram micro recall; Dev47 uses only10 of those questions and must not be numerically presented as that264-task rerun. The benchmark semantic scorer is a separate additional measure. Legacy Dev original26 refs and revised19 keep separate denominators. Previous v3 has13 native-only pairs,45/45 source atoms and15/19 accepted upstream references; no external paired dispatch.

Task1 complete: coordinated publication dates and named `that` complementizers fixed with red→green tests. Spec and quality reviewers both passed;106 scoped tests passed. Conservative event-date/time/ambiguity behavior remains. Pending source-only audit checked11 original required facts against19 complete sources (8 supported,3 unclear); this is not an answer audit. Task2 spec review found source-tail and governed-negation regressions; implementation fixed both with301 scoped tests, re-review underway. Separate comparator/pending-Dev scoring tests7 passed; missing annotation remains null and cannot filter Core.

Task2 complete: spec re-review passed64 tests; quality re-review passed79 focused tests and masked-conflict/contraction synthetic repros. Condition deactivation now preserves a feasible literal witness, retains all aspects on candidate ambiguity, and verifies final context. Final full make test:1371 passed,36 skipped; Ruff and docs checks passed. Code snapshot v1 contains202 code/config instruction files plus49 Dev source files, no Core input. Registered readonly Dev47 v1 is running; one native task encountered a deadline failure, whose model/search/ranker telemetry and missing arm results are retained. No accuracy claim or Core freeze yet.
