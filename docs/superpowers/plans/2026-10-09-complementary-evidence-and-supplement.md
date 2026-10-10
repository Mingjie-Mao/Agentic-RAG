# Complementary Evidence and Controlled Supplement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sequentially verify complementary evidence selection, quota-aware replacement, and one grounded supplemental retrieval on frozen Dev, with the same final eight-evidence / 5000-token budget.

**Architecture:** Replay the valid `depth50_rerank` candidate pool from the previous immutable Dev run, reauthorizing its chunks and verifying source/index identity before use. Selection and replacement are bounded business policies using existing question decomposition and lexical signals; no gold enters either. The final experiment uses official LangGraph nodes and existing literal bridge validation, model accounting and authorized retrieval for one additional search per paired arm. Production defaults and the existing Agent graph remain untouched.

**Tech Stack:** Python 3.13, existing PostgreSQL/OpenSearch/Ollama, Pydantic, LangGraph StateGraph, existing ExecutionBudget and benchmark utilities, pytest.

---

## Boundaries and registered method

- The user already selected the current checkout. Preserve its existing Agent, README/site, experimental and retrieval changes. Start hashes are `.runtime/retrieval-selection-20261009/initial-files.json` (39 files). No staging, commits, push, production reindex or document updates.
- Use the complete frozen Dev 47 selection only. Eligibility remains 26 native tasks / 41 source spans and 10 MultiHop tasks / 26 upstream facts; 5 historical tasks and 6 no-reference tasks remain in accounting. Never route supplementation from gold or the prior evaluator's `candidate_missing` label. Report that subgroup only after execution.
- Parent: `artifacts/retrieval-quality-dev-20261009-v4.json`, profile `depth50_rerank`, complete/valid, 47 tasks, zero errors. Parent candidate scores are cached measurements. Replaying them isolates selection, and its timing is NOT fresh end-to-end reranking latency.
- Validate parent bytes, selected IDs, frozen suite/selection, all source bindings and actual readable corpus/index against parent identity. Re-read and check identity at the end; retain invalid/error/interrupted artifacts and nulls. No parent overwrite or cached raw text in public outputs.
- All final contexts use `token_count(text) + token_count(title) + 100` per chunk, at most 8 unique IDs and at most 5000 tokens. Never truncate text to disguise over-budget evidence. Hard exclusions remain: authorization/version failures, `boilerplate_only`, `below_min_similarity`, `outside_requested_publication_dates` and tenant scope.
- Reuse `agent.planner.subgoals`, `app.verdict.question_slots`, `agent.planner.evidence_coverage`, `_grams`, `_EN_STOP` and `keywords`. Choose the decomposition with more items, tie preferring `subgoals`; cap six. The signals remain lexical candidate proxies, not semantic proofs. Freeze the actual global coverage-affecting configuration as well as explicit safe retrieval settings.
- Preserve every duplicate chunk's eligible lane alternatives; do not arbitrarily discard a global alternative or count a chunk twice. Source quota uses the original lane identity (`lane_sha256`) and group count re-derived with existing `route_sources(question, readable_documents)` (including groups with no returned hits), and original formula `max(2, (8-2)//source_group_count)`. Document quota 2 applies only when the original question is diversified (source groups, `needs_document_diversity`, or `multi_source_intent`). Each selected chunk uses one feasible lane, preferring earlier parent rank. Quota relaxation NEVER changes ACL or date/version filters.

## File map

| File | Responsibility |
|---|---|
| `app/evidence_selection.py` | Append pure bounded complementary selection/replacement APIs; preserve existing sentence selection and service helpers |
| `tests/test_evidence_selection.py` | Complementarity, quota variants, redundancy, lane alternatives, cost and identity behavior |
| `scripts/run_evidence_selection_dev.py` | Dev-only parent replay, registration, reauthorization, phase scoring and immutable selection/quota reports |
| `tests/test_evidence_selection_dev.py` | Parent/Dev guards, no gold in policy calls, drift/failure retention and reproduction |
| `agent/evidence_supplement.py` | Experiment-only LangGraph for runtime gate, two-step plan, literal bridge and paired controlled retrieval |
| `tests/test_evidence_supplement.py` | Graph gating, grounded values, equal search budgets, query constraints and failure/budget behavior |
| `scripts/run_evidence_supplement_dev.py` | Frozen paired supplemental Dev run with active policy model identity and derived telemetry |
| `tests/test_evidence_supplement_dev.py` | Matched controls, all-task accounting, model identity, no gold/original text export |
| `PROJECT_REPORT.md` | Actual sequential results, rejected methods, costs and pending answer/Core acceptance |

## Task 1: Pure selection and replacement policy

- [x] Write tests before implementation. Public APIs: `selection_facets(question)`, `select_complementary(question, candidates, *, limit=8, token_budget=5000, document_quota=2, source_quota=None, required_ids=())`, `replace_weak_evidence(question, candidates, seed, *, limit=8, token_budget=5000, document_quota=2, source_quota=None, max_replacements=8, required_ids=())`. Return a `SelectionResult` with `evidence`, `context_tokens`, and text-free `trace`. `required_ids` reserves real source chunks within the same budget for the later bridge experiment; infeasible/missing required evidence raises a sanitized `ValueError`, never silently falls out of the context.

```python
def row(cid, text, rank, doc="a"):
    return {"chunk_id": cid, "document_id": doc, "version_id": doc + "-v1",
            "source_sha256": doc + "-sha", "title": "Alpha", "text": text,
            "rank": rank, "lane_type": "global", "lane_sha256": "global"}

def test_two_facets_beat_repeated_background():
    pool = [row("a1", "The timeout is 30 seconds.", 1),
            row("a2", "The timeout is 30 seconds.", 2),
            row("b1", "The retry limit is 3 attempts.", 3, "b")]
    got = select_complementary("What is the timeout; what is the retry limit?", pool, limit=2)
    assert {r["chunk_id"] for r in got.evidence} == {"a1", "b1"}
```

- [x] Confirm the tests fail on the missing module, then implement the minimal pure policy. Precompute per-chunk lexical material and facet coverage; exact chunk IDs deduplicate but differing document/version identities are not merged. Greedy priority: newly covered facets, newly covered meaningful question terms, lower maximum content Jaccard redundancy, then earlier parent rank and stable chunk ID. Empty lexical signals fall back to rank. Evaluate feasibility before admission and keep counts/costs exact.
- [x] Implement bounded replacement from the same strict seed: evaluate each one-for-one swap (or append if below eight); accept only a strict lexicographic improvement of total covered facets, covered meaningful question terms, lower summed pairwise content Jaccard redundancy, then lower summed parent rank. Stop when no improving feasible change exists or eight accepted changes. Always re-evaluate token/lane/document constraints. Trace added/removed IDs, objectives/counts and cap flags, never facet text or source text.
- [x] Test third relevant block blocked by strict document quota and available when relaxed; source quota independently; matching global alternative; oversized blocks; stable ties; duplicate same ID; conflicting source/version identity rejected; no mutation of inputs; bounded replacement and no cycling; required evidence retained by selection/replacement and a required source that cannot fit rejected explicitly. Run `.venv/bin/pytest -q tests/test_evidence_selection.py` and scoped Ruff, then independent spec and quality reviews.

**Task 1 verification:** 52 tests passed; independent spec review compliant and quality re-review approved. Repeated lane-option blowup fixed without policy changes; 1000 seeded assignments matched exhaustive earliest-feasible assignment. Existing sentence/service bodies preserved. Unsupported goal lanes/unknown exclusions remain outside the registered replay scope.

## Task 2: Frozen Dev replay runner

- [x] Write guard tests for non-Dev/mismatched selection, failed/partial/wrong-profile parent, duplicate task IDs, changed parent/source SHA, unauthorized or inactive chunks, existing output and missing telemetry. Verify policy receives only question plus authorized candidate rows and budgets, never the task object/gold/answer/reference IDs.
- [x] Implement CLI `--stage selection|quotas --parent <file> --output <fresh-file>`. Require all 47 tasks. Reuse `validate_suite`, `_bound`, `_telemetry`, code/model fingerprints, exclusive reservation/publication, `active_scope`, `index_ledger`, `CheckpointStore`, `atomic_json`, and `checkpoint_lock` from the existing modules; do not introduce another checkpoint engine.
- [x] Parent replay must reconstruct `legacy_rerank` from its recorded context chunk IDs, rebind real text/version SHA and reproduce all parent phase scores exactly before comparison. New candidate choices cannot read parent gold phases, matched IDs or cause labels. Freeze code, config, parent bytes, source/index and profile declarations before selection; recheck after all tasks. Preserve null/errors/all-task denominators and hash-only lane/facet/query identities.
- [x] Register the following exact stages; quota profiles share the same strict greedy seed and replacement objective, isolating each cap:

```python
SELECTION_PROFILES = ("legacy_rerank", "complementary_strict")
QUOTA_PROFILES = ("complementary_strict", "replace_strict", "replace_doc",
                  "replace_source", "replace_both")
# complementary_strict and replace_strict keep original caps.
# replace_doc relaxes only the document cap; replace_source only the source cap.
# replace_both relaxes both. All retain <=8 chunks and <=5000 tokens.
```

- [x] Pair selection against `legacy_rerank`, quotas against `replace_strict` (also retain greedy control comparison). Summaries use existing native/upstream proxy bases, applicability, losses and null final-quality metrics. Report selection-only time, final token/count usage, duplicate/redundancy observations and derived replacements. No answer generation or new model calls in these two stages.
- [x] Run `.venv/bin/pytest -q tests/test_evidence_selection_dev.py tests/test_evidence_selection.py tests/test_retrieval_quality_metrics.py` and Ruff; perform independent spec then quality review.

**Task 2 verification:** Independent spec approval; quality re-review approved after typed telemetry/model provenance guards. Final 113 runner tests; independent related suite 199 passed; scoped Ruff clean. Real parent accepts all47/36eligible/11N/A and cached model provenance projects exactly. No live services were used in implementation/reviews.

## Task 3: Sequential real selection and quota verification

- [x] Run stage selection and inspect valid/all-task/identity/error/accounting checks before the next stage:

```bash
.venv/bin/python scripts/run_evidence_selection_dev.py --stage selection \
  --parent artifacts/retrieval-quality-dev-20261009-v4.json \
  --output artifacts/evidence-selection-dev-20261009-v1.json
```

**Selection real run:** `evidence-selection-dev-20261009-v1.json` complete/valid, zeroerrors/no drift/all47 perarm. Exact legacy33/41 native and17/26 upstream reproduced. Strict complement9/41 and9/26; all-upstream-facts complete4/10 ->0/10. All47 mean pairJaccard0.0511904 ->0.0210892, tokens1370.43 ->1333.02, eight chunks inboth. Reject this policy: lower lexical redundancy did not preserve factual relevance.

- [x] Run stage quotas with the same parent and frozen method:

```bash
.venv/bin/python scripts/run_evidence_selection_dev.py --stage quotas \
  --parent artifacts/retrieval-quality-dev-20261009-v4.json \
  --output artifacts/evidence-quotas-dev-20261009-v1.json
```

- [x] Select the supplemental seed on Dev only, and freeze its profile/output hashes before supplement: require all 36 eligible tasks measured, no drift/errors, native final coverage >= legacy 33/41; maximize MultiHop delivered facts, then native delivered spans, then fewer relaxed caps. If no candidate qualifies, use `legacy_rerank`. Retain all negative results; this is Dev selection and not unseen validation.

**Quota real run:** `evidence-quotas-dev-20261009-v1.json` complete/valid, zeroerrors/no drift/all47 each of5arms; strict greedy contexts/phases match selection run. All four replacement arms9/41 native and7/26 upstream (1/10 all facts), versus strict greedy9/41 and9/26 (0/10). No native guard qualifier. These are negative results for this lexical objective, not proof that quota limits never matter.

**Frozen supplement seed:** `legacy_rerank` from `artifacts/evidence-selection-dev-20261009-v1.json`; bytes SHA256 `3d0f0f1c43766723863e93da4845985800e0ee6d870e04237c2283b2f2840da9`. Parent bytes SHA256 `3ebdd372139b0e24a7237b9db99475a5306a717d4b114236f9b67b905a7c74e6`; quota report SHA256 `61190c485c2d7b0846304e31ea0ec5edb7a8d0f109872b880d2c412b79166ea6`. Registered local manifest `.runtime/retrieval-selection-20261009/supplement-seed-registration.json` before graph/model implementation and dispatch.

## Task 4: Grounded one-search supplement with LangGraph

- [x] Write toy tests before implementation. New helper is experiment-only and does not change existing `agent/graph.py`, `agent/controller.py`, `agent/adaptive.py` or defaults. Use official `StateGraph` nodes/conditional edges; compile without durable production writes. This experiment claims no crash-resume capability. Reuse existing budget persistence for call accounting.
- [x] Runtime gate: nonempty authorized seed and either missing lexical facets, `multi_source_intent(question)` or `needs_document_diversity(question)`. No gold, prior loss labels, task category or expected behavior is available to the gate. History-only component tasks remain N/A without model dispatch. Record every gate outcome.
- [x] Reuse existing `agent.planned.Plan`, `PlanStep`, `Extract`, `wire_schema`, `_call`, `validate_plan`, `extract_value`, `literal_values` and subject-grounding helpers. Generate a bounded two-step search-only plan with one entity extraction and one dependent query (no list fan-out or historical operation). One planning call capped at 300 output tokens plus one existing short extraction call; restrict the model schema to exactly two steps (rather than asking for unrestricted five-step plans and truncating them); use `@bounded_model("policy")`, maximum two policy calls, 180-second per-task total deadline, no automatic replan/retry. Reject unsupported/incomplete plans as explicit no-valid-bridge outcomes.
- [x] Reauthorize source chunks before each model read and after literal validation. Use existing `focus_document` and `ungrounded_terms` on the authorized seed to focus first-step extraction, then existing `literal_values` row anchoring. A bridge must occur verbatim in an authorized current-version chunk and pass that subject anchoring; invented values, unauthorized/mismatched versions, unresolved placeholders, empty/numeric-only bridge, oversized query or a treatment indistinguishable from control cannot dispatch a search. A deliberately unchanged original-question control is allowed as the matched extra-call control. This lexical binding is not a semantic proof.
- [x] Build a matched pair from the same accepted plan/extraction: the control query removes the bridge placeholder, the treatment substitutes the verified entity. Both retain the full original question plus a bounded dependent query within SearchArgs' 500-character limit, or skip both explicitly if that cannot fit. Before dispatch, re-derive source-group document/version keys for both completed queries and require the same group scopes as the original question; additional source/date scopes reject the pair. Both use the same source/date scope (`routing_goal(question, fuse=False)`), depth50 authorized retrieval, no supplemental cross-encoder, and exactly one supplemental search maximum per arm. Initial frozen retrieval counts as one prior search in each arm; added calls are reported rather than described as free. One retrieval-tool attempt may fan out into multiple backend searches/embedding requests; record the actual backend request and embedding call counts/latency per arm, with only derived request hashes, not raw queries.
- [x] Graph nodes gate -> plan -> extract -> validate -> first search -> second search -> select -> END, with conditional edges and early END for legitimate skips. Task-index parity determines the two arm labels before dispatch; each arm has a single search attempt, with no retry or custom executor. Budget limits count two policy calls and two authorized retrieval-tool attempts for the paired experiment, i.e. one per arm. Each search node can fail independently; no silent scoring as success. Final selector uses the chosen frozen seed method and same <=8/5000 constraints on the union of old and new authorized candidates. Within each query, keep recorded candidate order; assign union `parent_rank` by deterministic old/new round-robin (old occurrence n = 2n-1, new occurrence n = 2n), with old recorded rank and per-query origin retained as derived telemetry. Never compare old cross-encoder scores against new fusion scores. Both arms use the same merge. A legacy fallback uses this rank order via rank-only selection; complementary/replacement seeds re-run their frozen policy on the union with required evidence reserved. Reserve the same verified bridge source chunk for both arms through `required_ids`, within the budget; no hidden ninth context chunk. Infeasible source retention is an explicit failed/unsupported treatment, not successful ungrounded coverage.
- [x] Public telemetry contains gate/status, policy/tool attempts and costs, IDs and SHA hashes of plans/queries/values/quotes, replacements and phase counts only. Raw states stay in memory/local `.runtime`. Tests cover gate skip, missing facets, valid entity-dependent query, fabricated entity, source revoked after extraction, version mismatch, preserved original constraints, same one-search cap per arm, budget exhausted, model/search failures and source-evidence retention. Run scoped pytest/Ruff; independent spec and quality review.

**Task 4 verification:** Fresh independent spec re-review approved66tests/Ruff after seed-bound and trace corrections. Quality subagent dispatch failed because account usage was exhausted; root (independent of the implementation author) completed code review, real LangGraph no-service thread/context/budget probe,143related tests and scoped Ruff. Probe confirmed same-main-thread callback execution and current budget propagation. Remaining implementation/review proceeds via executing-plans rather than waiting for unavailable subagents; final review provenance will state this limitation explicitly.

## Task 5: Supplemental Dev integration and real verification

- [x] Implement `scripts/run_evidence_supplement_dev.py --parent <v4> --seed-report <selection-or-quota-report> --seed-profile <chosen> --output <fresh>`, with tests. Reuse replay and benchmark utilities, validate seed task/reference/input hashes and registered method, freeze the chosen seed and active policy/embedding model identities before model dispatch. No policy is chosen inside runtime by gold.
- [x] Run all 47 IDs, rotating control/treatment search order by task index while using one shared plan/extraction per task. Keep operational failures, history/no-reference applicability and complete/missing pair counts explicit. Compare baseline seed, matched no-bridge control, and verified-bridge treatment; measure newly delivered gold only in evaluator. Report previously candidate-missing subgroup separately as post-hoc diagnosis, not a routing rule. Strict success/semantic recall remain null.
- [x] Run scoped pytest/Ruff and spec then quality review before the real run. Register exact seed profile and hashes in the plan/runtime before running:

```bash
.venv/bin/python scripts/run_evidence_supplement_dev.py \
  --parent artifacts/retrieval-quality-dev-20261009-v4.json \
  --seed-report artifacts/evidence-selection-dev-20261009-v1.json \
  --seed-profile legacy_rerank \
  --output artifacts/evidence-supplement-dev-20261009-v2.json
```

Preserve pre-existing `evidence-supplement-dev-20261009-v1.json` unchanged (older runner skipped all11 N/A tasks and lacks the completed active-policy/cost protocol). The corrected runner uses fresh v2; seed, graph policy, budgets and method remain frozen.

**Task 5 pre-run verification:** Root42runner tests/Ruff passed. Fresh independent spec review approved after interruption cost retention and policy event fixes; fresh quality review identified the backend-interface gap, then approved the existing Search implementation inheritance with real-authorized-retrieval regression tests. Independent graph/runner joint suite108passed, scoped Ruff and diff check passed.

Task3 selected the registered `legacy_rerank` fallback; the command above is the actual frozen command, with seed hashes recorded above. No alternative seed may be substituted after inspecting the supplemental results.

## Task 6: Report and final verification

- [x] Append a new dated section to `PROJECT_REPORT.md`; retain section20.30 and all old artifacts. Report each stage's baseline/denominators, native/upstream context coverage, all-facts-complete tasks, error/missing/gate counts, selection and supplemental costs, actual replacements and adoption decisions. No README/site headline accuracy update or production default switch from proxy-only data.
- [x] Run `make test` and `git diff --check`; compare initial file hashes (only documented new task hunks/append may differ). Independent final review must validate actual artifact/report arithmetic, gold isolation, authorization, same budgets and source-evidence binding.
- [x] Mark completed steps only after their checks/reviews pass and provide a self-contained Chinese completion report. Preserve current checkout and all user work; no integration menu, unrequested commit or push.

Framework reference checked before implementation: [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) and [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence). Selection policies are enterprise business semantics; scheduling/state transitions use LangGraph rather than a custom executor.

**Supplement real run v2:** complete/valid, all47/36eligible measured, 0errors/no drift; gate_false26/no_valid_bridge16/historyNA5. Active policyHTTP16success, prompt21214/output2815; graph budgetpolicy16attempt/15success/1failure; no plan accepted, extraction/search/embed/backend0. Three finalcontext arms identicalseed33/41native and17/26upstream; no genuinely dispatched matched searchpairs, treatment effect pending. Posthoc3candidate-missing tasks stay4/10facts, two plan rejected/one gatefalse. Realv2 artifact preserved; pre-existingv1 unchanged. Fullmake test1079passed36skipped and Ruff/docs passed; final report §20.31 inserted, final artifact/report review pending.

**Independent final review:** Approved actual §20.31 arithmetic/privacy/causal limits and immutable v2;108joint tests passed independently. Clarified final-context budget wording. Historical report bytes restore exactly after removing only §20.31. All eight historical evidence/retrieval artifacts match pre-final hashes; concurrent unrelated app/site/web/probe edits preserved. Full suite re-run underway because three additional tests were added by concurrent work after the prior1079/36 result.

**Final verification:** Latest full `make test`:1082passed36skipped (1118collected), Ruff and docs passed after concurrent tests were added. Final report updated to latest count; independent final review approved. Original report body byte-for-byte preserved outside new §20.31; original evidence-selection ASTbodies preserved; all historical evidence/retrieval snapshot artifacts unchanged. All tasks completed; no production default change/Core run/Git write.
