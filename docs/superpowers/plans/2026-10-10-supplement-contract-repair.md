# Supplement task routing, generation contract and evidence slots

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Execute independent supplementation without fabricating entity dependencies, align model decisions with program-owned execution fields, gate by source-bound structural slots, and separately measure candidate absence versus selection loss.

**Architecture:** Reuse official LangGraph, Pydantic, existing source alias/date parsing, question slots, task contracts and literal extraction. New business contracts are experiment-only; the old supplement runner/default behavior and production source routing remain compatible. Query-only/authorized-metadata-only routing and selection receive no benchmark IDs, gold, expected paths or answer labels. Semantic uncertainty remains explicit.

**Tech Stack:** Python, existing Pydantic/LangGraph dependencies, pytest/Ruff, read-only PostgreSQL/OpenSearch, existing metered local Ollama policy/embedding calls.

**Workspace:** Current directory as already authorized; `.runtime/supplement-repair-20261010/initial-files.json` freezes 1696 initial file hashes and HEAD. Preserve existing Agent/frontend/site work and all historical artifacts. No database/index writes, Git mutation, worktree changes or deployment. Skill commit steps use file/hash checkpoints under this workflow. Use fresh spec and quality reviewers between stages, with honest fallback disclosure if reviewers become unavailable. No confirmation menu: execution is already authorized.

## Task 1 — Query-only routing and distinct source requests

**Files:** Create `app/supplement_contract.py`, `tests/test_supplement_contract.py`; minimally factor alias mention parsing in `app/retrieval.py` with regression tests. Do not implement gate scoring or dispatch until this pure routing contract passes review.

- [x] Write failing tests, run them red, implement and run green. Commands: `.venv/bin/python -m pytest tests/test_supplement_contract.py tests/test_retrieval_candidate_depth.py -q`; `.venv/bin/ruff check app/supplement_contract.py app/retrieval.py tests/test_supplement_contract.py`.
- [x] Factor the already-existing alias matching algorithm into `source_mentions(question, authorized_documents)`, preserving exact production `route_sources` outputs. Each occurrence retains original start/end, canonical mention and matching authorized documents; repeated publisher mentions are not deduplicated here. Production deduplication/date fallback remains unchanged in `route_sources`.
- [x] Build frozen dataclasses for `SourceRequest`, `SupplementSlot`, `SupplementContract`; pure `build_supplement_contract(question, authorized_documents)` creates program-owned slot IDs. Source requests preserve publication date constraints and source/document/version identities. Explicit publication date with no matches never silently widens; incomplete metadata or unsupported/ambiguous date attachment stays unknown. Distinguish publication dates from event/version dates and use strict before/after semantics for the new contract, without changing historical routing.
- [x] Reuse `question_slots`/`build_contract` for explicit independent asks/comparisons. A recognized independent multi-source request does not establish an entity bridge. A clearly nested entity dependency is a bridge proposal; otherwise retain unknown for bounded model routing in Task2. No carrier-name-specific rule or task-ID routing.

```python

def test_repeated_source_dates_are_distinct():
    documents = [document('jan', 'Publisher P', '2026-01-03'),
                 document('feb', 'Publisher P', '2026-02-08')]
    c = build_supplement_contract(
        "Compare Publisher P's report published January 3, 2026 with Publisher P's report published February 8, 2026.", documents)
    assert c.mode == 'independent'
    assert len(c.slots) == 2
    assert c.slots[0].source.document_ids == ('jan',)
    assert c.slots[1].source.document_ids == ('feb',)


def test_unknown_publication_date_does_not_widen():
    c = build_supplement_contract(
        "What does Publisher P's article published February 8, 2026 say?",
        [document('jan', 'Publisher P', '2026-01-03')])
    assert c.slots[0].source.document_ids == ()
    assert c.slots[0].source.reason == 'no_matching_publication_date'
```

- [x] Additional behavior tests: explicit independent clauses; nested generic possessives; ambiguous anaphora; history/conditions not promoted to entity bridges; malformed metadata; same publisher/date repeated; quoted instruction-like source names not followed; unknown aliases/unsupported dates. Raw queries/spans stay in memory, public contract projection exposes only IDs, hashes, modes, fixed reason codes and counts.
- [x] Root verifies changed files and tests, then request fresh spec/quality reviews. The spec request failed on account limit; disclose unavailable independent reviews and use root fallback checks. Record actual reviewers and review independence. Mark Task1 complete before Task2 code changes.

## Task 2 — Model decisions only; program-owned execution structure

**Files:** Create `agent/supplement_proposal.py`, `tests/test_supplement_proposal.py`, `agent/supplement_repair.py`, `tests/test_supplement_repair.py`. Reuse private source/budget helpers from `agent/evidence_supplement.py`; leave that legacy default path intact.

- [x] TDD the narrow transport and execution behavior. Use Pydantic schemas through existing `_call`/`wire_schema`; no new provider wrapper, custom graph engine, retries or checkpoint implementation.
- [x] One policy proposal chooses an indexed existing slot, focused query or a bridge's initial query/entity description/plain followup prefix and suffix. Known independent route is constrained to independent; unknown routing may return independent/entity_bridge/unknown. Remove generation of step IDs, kinds, dependency lists, extraction type/names, iteration/date operations and tooling. Unknown/no useful query is a valid bounded skip, not a fabricated successful plan.

```python
class FocusProposal(BaseModel):
    model_config = ConfigDict(extra='forbid')
    target_slot: int
    query: str = Field(min_length=2, max_length=200)

class EntityBridgeProposal(BaseModel):
    model_config = ConfigDict(extra='forbid')
    target_slot: int
    first_query: str = Field(min_length=2, max_length=200)
    entity_description: str = Field(min_length=1, max_length=120)
    followup_prefix: str = Field(max_length=80)
    followup_suffix: str = Field(max_length=80)
```

- [x] Constrain target slot to program-created indices in emitted schema. Bridge prefix/suffix are bounded plain text without braces; their combined trimmed length must be at least2. The program inserts exactly one fixed `{s1.entity}` token. The <=160-character context fits the existing200-character step query bound. Program builds `s1` with mandatory `extract.name='entity'`/kind entity, and `s2` depending only on s1, with no extraction. Reuse `extract_value`, `focus_document`, `ungrounded_terms`, `literal_values` and `_shape` for binding/acceptance; don't remove provenance checks just to raise plan acceptance.
- [x] Independent branch skips entity extraction and has no fabricated required bridge source. Matched control uses original query; focused arm uses the authorized bounded focused slot query. Bridge branch retains shared source binding, placeholder-free control and literal bridge treatment. Both use official LangGraph conditional edges and the existing rank-only within-query/round-robin union as the initial selector; no cross-query score comparison.
- [x] Keep <=2 shared actual policy calls (proposal300/extraction60), <=1 additional search per arm, cumulative180 seconds,8 unique chunks/5000 tokens. Refresh current authorization before proposal/extraction, search and final selection; reject stale content. Failed transports consume attempts; interruption is preserved. Source validators freeze actual original routing scope and allow only effective query scopes contained in that original scope, with original routing context applied to search. No alias/date fallback broadening. Do not concatenate full original+focused queries past500 chars or silently truncate them.
- [x] Test independent query dispatch with one policy call/no extraction, valid bridge with2 calls and required source both arms, unknown route and malformed schema, no model-owned structural fields, repeated-source query scopes, malicious new publisher/date, placeholders, revocation/version changes, equal budgets/per-arm search failure, privacy and partial/interrupted results. Run scoped legacy supplement tests too.
- [x] Root fallback spec/quality checks, fixes and fresh root tests before Task3; independent reviewers unavailable. Save source/code file hashes as checkpoint.

## Task 3 — Structural evidence gate with explicit uncertainty

**Files:** Extend `app/supplement_contract.py`, its tests and repair graph only. Legacy `evidence_coverage` remains a lexical signal; don't relabel it semantic support.

- [x] Define `assess_slot_presence(contract, rows)` returning per-slot source/date/subject/attribute states, source-scope compatible witness IDs, lexical candidate IDs and fixed reasons. Rows must have authorized/current version bindings. Use `build_contract`/`bind_values` for supported literal numeric/enum/date patterns after source scope filtering. Same coherent witness must bind subject and attribute; separate actors/chunks must not cross-satisfy a slot. Complete table header+row/source scope may use the existing binding helper's documented source context, not arbitrary union of unrelated facts.

```python

def test_subject_and_attribute_must_bind_together():
    c = build_supplement_contract('Product A 的超时阈值是多少？', [])
    p = assess_slot_presence(c, [row('b', 'Product B 的超时阈值为 9 秒。')])
    assert p.slots[0].state != 'supported'


def test_no_generic_prose_entailment_claim():
    c = build_supplement_contract('What was the article\'s position on privacy?', [])
    p = assess_slot_presence(c, [row('x', 'Privacy was mentioned in this article.')])
    assert p.slots[0].state == 'unknown'
```

- [x] Supported means recognized structural/literal presence, never answer correctness/entailment. Fully specified resolvable requirements without a witness are missing in the inspected pool; unsupported language, absent metadata, ambiguity/anaphora and unknown condition branches are unknown. Missing source/date must not be certified by same-publisher evidence. Repeated dated mentions remain separate slots. Gate on missing/unknown active requirements with a bounded useful query; no labels/gold/expected paths.
- [x] Reuse condition evaluation to respect inactive branches when safely resolved; unknown conditions remain unknown. Model proposal may help choose query/route, not certify semantic support. If no structurally supported subject/attribute parser exists, keep unknown and report it rather than inventing a lexical confidence threshold.
- [x] Test all four dimensions, dates/event distinction, absent/stale metadata, source SHA/version mismatches, table/row labels, negation, ambiguous multiple values, history/conditions, empty inputs and word-only false positives. Gold-free tests prove input/API keys never accept benchmark labels. Root fallback spec/quality checks before Task4; independent reviewers unavailable.

## Task 4 — Candidate versus selection diagnosis; registered full Dev validation

**Files:** Extend `app/supplement_contract.py` diagnostics; add isolated source-slot reservation strategy to `app/evidence_selection.py` with tests; extend repair graph. Create `scripts/run_supplement_contract_dev.py`, its tests; append new section in `PROJECT_REPORT.md`. Preserve old experiment runners/outputs and frozen overlay.

- [x] Add `diagnose_slot_loss(candidate_presence, selected_presence)` with retained / candidate_absent / selection_loss / unresolved states. Candidate absence never means absence from corpus. Unknown assessment does not become zero/full support; avoidable loss only claimed when a feasible cap-respecting alternative is demonstrated.
- [x] Reserve source/date-compatible structural witness inside the same caps, then rank fill. For unknown semantic slots, scoped lexical rank may guide candidate relevance but is labelled lexical only; don't use this as semantic evidence certification. Keep original-query scores comparable only within original stream; supplemental candidates retain own retrieval order. Reuse real lane assignment/hard exclusions, exact token cost and immutable authorization snapshots. No gold-promoted candidate, slicing/truncation of raw evidence, or new reranker/judge/model call hidden in selection.
- [x] Register before scoring an offline matrix: unchanged relevant_doc seed; source-slot reservation under original capped context; feasible literal witness reservation. Use actual original question/source metadata, all47 original Dev IDs and reviewed overlay from20261009. Reproduce old seed/legacy artifacts and every source/code/corpus/index identity before comparisons. Preserve all failed/interrupted runs with new names. Compare revised native45 and accepted upstream19 separately; pending7 question completion stays null. Seed for live treatment is frozen before dispatch using per-task native no-loss guard and accepted upstream gain/fewer relaxations; fallback original relevant_doc if none qualifies.
- [x] Register actual repair method/budgets/models, baseline/context/overlay/manifest/parent/code hashes and ordering before one full47 matched live run. Live graph receives only original question, authorized source snapshots and evidence. Reuse existing Models/Search physical metering, CheckpointStore for experiment bookkeeping, read-only index/corpus verification and source reauthorization. No-reference tasks follow normal gate, history stays N/A. All missing rows/costs remain null, not0. Actual dispatched pair census is distinct from observed context pairs.
- [x] Report: route mode census, structural/unknown slot counts, schema/acceptance rejection reasons, valid-plan rate with attempted denominator, extraction/search/paired dispatch counts, accepted-reference proxy changes and gains/losses, pending annotations, backend/embedding/policy physical cost and error/null counts. Final generated-answer correctness stays pending unless an independent frozen answer-scoring run is actually performed; don't label source coverage answer accuracy. No unseen Core use.
- [x] Fresh independent spec/quality/result reviews (or explicitly disclosed unavailable review), correct any privacy/arithmetic/causal-claim issues. Run `make test`, `scripts/check_docs.py`, `git diff --check`, and initial artifact/HEAD/report-byte invariants. Preserve unrelated concurrent edits. Record final test totals and limits; provide self-contained Chinese outcome without Git integration menu.

## Validation and framework sources

Existing `langgraph==1.2.12` provides the needed graph nodes/conditional routing; Pydantic and Ollama `format` provide the schema transport. Consulted official docs: https://docs.langchain.com/oss/python/langgraph/workflows-agents and https://docs.ollama.com/capabilities/structured-outputs. No framework gap requires a custom execution engine. Root architectural review and a separate same-family model design advisor both identify current source-routing fallback and generic-schema/strict-execution mismatch; this is not independent gold/answer review.

## Actual review checkpoint

Task1: fresh root regression107 passed and Ruff clean. The fresh spec reviewer `/root/supplement_routing_spec` failed before review because of account usage limit. Execution therefore uses executing-plans fallback, root spec/code checks and tests; independent spec/quality review is unavailable, not passed. Task2 proposal/graph/legacy93 passed; Task3 contract/graph69 passed. Root checked scope pairing, date fallback, model field ownership, literal provenance, uncertainty and privacy. File hashes saved locally; no Git mutation.

## Freeze and rerun log

Offline v1: missing input registration, retained failed preflight. v2:141 completed checkpoint rows but seed-choice history/null bug; failed public report retained. v3: valid full47,45/45 native and15/19 baseline versus14/19 both new floors, baseline retained. Fixed exact-clock/nonstring identity handling and made no-useful-query decline explicit; no result-driven threshold change. v4:141 error-free rows but unrelated concurrent edits to app/config.py, app/clients.py and demo scripts caused code drift, correctly invalidated. Execution now uses an ignored local code snapshot (197 code files,49 Dev source files copied and hash-checked; no worktree or Git mutation, no Core tasks) while implementation remains in the authorized current directory. v5 snapshot preflight lacked its local runtime parent directory; retained failure, directory initialized and package validation passed. v6 repeats the registered matrix from frozen-code-v1. New interruption/unresolved-scope tests passed19, earlier full suite1256 passed36 skipped. Independent reviewers remain unavailable due account limit; these are root checks.


Offline v7 uses frozen-code-v2 and is complete/valid, all47 IDs, no drift/errors; all3 profiles deliver45/45 native; baseline15/19 accepted upstream versus14/19 for both new floors. The predeclared seed rule retains relevant_doc. Live v1 from frozen-code-v3 completed all47 but a plain-string versus JSON-string fingerprint mismatch invalidated posthoc bridge binding. Live v2 from frozen-code-v4 completed47, with5 actual pairs and one operational error when literal_values returnedNone; accepted-reference delivery cannot be fully compared because of that missing task. Both failed reports are retained. Neither supplies valid efficacy evidence.

Pre-live-v3 correction: red regression demonstrated absent nonliteral entity caused TypeError; guard now records normal literal_binding_mismatch without dispatch. Model transport supplies plain prefix/suffix rather than generating placeholders; the program constructs the fixed bridge token. Prompt schema remains grounded and literal provenance/ACL/versions/caps unchanged. Scoped168 passed; make test1291 passed36 skipped (1327 collected), Ruff and docs passed. Before dispatch registered live-registration-v3 and copied frozen-code-v5 (199 code files49 Dev sources, no Core). Independent review remains unavailable; root fallback review covers contracts, pairing, budgets, privacy and failure states. Live v3 is running; final result audit/report follows completion.


Final checkpoint: live v3 complete/valid, all47 IDs and all3 profiles,36 measured/11N/A/0missing, no execution errors or registered drift. Accepted16/34 proposals (10independent6bridge), completed13 pairs (9independent4bridge), all native; external0 pairs. Native45/45 and accepted upstream15/19 unchanged in seed/control/treatment, gained/lost references0. Declines17 no_grounded_query; unresolvedsource3; structurallycomplete5; history5; samequery1; bindingreject2; proposalreject1. Physical policy38 requests64218 input922 output tokens,807.73 seconds;26 embedding and52backend requests,26search tools, taskwall895.36 seconds. Budget policy38 attempted37 succeeded1 failed; HTTPall38 succeeded. Unknown telemetry retained asnull. Local audit checks frozen199 code/49 source hashes, matched caps, all47 IDs, zero errors, actualpair count, evaluator-only labels and no raw source/query payload in public output. Report appended only§20.33, prior report bytes preserved. Fullmake test1291 passed36 skipped, Ruff/docs passed. Initial602 artifact/benchmark hashes andHEAD unchanged; noGit mutations. Independent review unavailable, root review nonhuman/nonindependent; final answer/semantic/Core scores remainnull.
