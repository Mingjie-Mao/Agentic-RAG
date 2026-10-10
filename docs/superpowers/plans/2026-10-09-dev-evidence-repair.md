# Dev evidence granularity, gold audit, relevance selection and planner diagnosis

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct Dev evidence references before optimizing retrieval, then record exact planner rejection reasons and rerun the bounded matched supplement experiment.

**Architecture:** Preserve the canonical v1 package, Core and all historical artifacts. Add a hash-bound Dev-only annotation overlay with question-specific native source atoms and explicitly disclosed model source review. Reuse existing authorized candidate replay, selectors, budgets and official LangGraph; no production default change. Perform stages strictly in order, with spec and quality checks per implementation. Independent reviewers interrupted by quota cannot be represented as completed reviews; root checks and their lack of independence are disclosed.

**Tech Stack:** Python 3.13, Pydantic, pytest, existing PostgreSQL/OpenSearch readers, LangGraph, existing Qwen2.5 policy model.

**Workspace:** Current checkout as explicitly requested; retain unrelated Agent/site/frontend work. Initial file/HEAD hashes saved in `.runtime/dev-evidence-repair-20261009/initial-files.json`. No Git mutations, deployment, business database/index writes or automatic annotation rebuild. Commit checkpoints in the skill are replaced by retained file/hash checkpoints under the user's current-checkout workflow.

## Stage 1A: Blind Dev annotation proposal and independent source review

**Files:** Create local `.runtime/dev-evidence-repair-20261009/annotation-draft-v1.json`; then a new accepted annotation file and tracked text-free manifest under `benchmarks/enterprise_rag/v1/dev/evidence-20261009-v2/`. Never change `.runtime/benchmark-package/v1/dev/tasks.json`, its selection, `manifest.json`, Core or old reviewed packets.

- [x] A fresh proposer reads only the 47 original Dev tasks and their bound source files, not retrieval outputs/ranks. For all 26 active native questions, select minimal complete source sentences for required facts, true conditional branches and comparison operands. Preserve predicates, entity, units, negation, dates and versions; numbers alone are insufficient. Derived answers require source operands, not invented result quotes. History5 and no-reference6 stay explicit and present.
- [x] Each native atom has stable ID, exact source path/bytes SHA, document ID, original character start/end, verbatim quote, supported fact IDs and rationale. The quote must exactly equal the source substring. Audited source atoms measure evidence availability, not final generated answer correctness.
- [x] Review all 26 upstream references for source binding, question/source/date/negation scope and relevance, plus the original answer's consistency. Every original reference receives keep/reject/unclear and a reason; no selective review based on previous loss. Preserve original IDs/text/hashes locally; public output contains only hashes, counts and decisions. Unclear annotations prevent a fully-reviewed task-complete claim. Even when a task has no accepted references, keep its original eligibility and ID; use annotation_pending/null completion rather than converting it to a no-reference N/A or a successful zero-reference task. Denominators report original/accepted/rejected/pending separately.
- [x] Attempt fresh independent GPT source review in a separate conversation. Its quota interruption and lack of complete per-item review are explicitly recorded. Preserve partial review and blind proposal; finish source checks inline under executing-plans, with retrieval blindness and author independence both false, exact serving identity unavailable, same model family and zero human review disclosed. Keep disagreements pending. Reuse the model-allowed benchmark-package review validator; do not present fallback checks as independent final approval.
- [x] Freeze accepted overlay/manifest byte hashes before any revised score or retrieval policy selection. Validation rejects wrong source hashes/locations, duplicates, missing47 IDs, mismatched package/split, unreviewed decisions presented as final, and any Core input.

## Stage 1B: Opt-in revised evaluator and complete baseline replay

**Files:** Create `scripts/dev_evidence_annotations.py`, `tests/test_dev_evidence_annotations.py`, `scripts/run_revised_evidence_dev.py`, `tests/test_revised_evidence_dev.py`; modify `scripts/retrieval_quality_metrics.py` and its tests only to add explicit opt-in native source-atom scoring while preserving legacy outputs.

- [x] TDD cases: correct answer sentence with omitted disclaimer passes the revised protocol while legacy whole-document proxy can fail; bare unrelated matching number, wrong document/hash/version, missing negative qualifier and missing comparison operand fail; missing telemetry stays null. Original legacy tests and output fingerprints stay identical.

```python
def test_atom_does_not_require_disclaimer():
    atom = {'id': 'atom-rpo', 'document_id': 'doc', 'source_sha256': 'source',
            'basis': 'source_atom_exact_proxy', 'text': '生产数据库的 RPO 为 10 分钟。'}
    evidence = [{'chunk_id': 'c', 'document_id': 'doc', 'source_sha256': 'source',
                 'text': '生产数据库的 RPO 为 10 分钟。', 'rank': 1}]
    assert phase_coverage([atom], evidence)['delivered'] == 1
```

- [x] Use existing `phase_coverage`, `diagnose`, `summarize`, `pair_profiles` with explicit reviewed reference overrides; native acceptance requires each normalized complete content block within a matching-source evidence chunk; scope and value blocks may be in separate chunks of the same bound source. Structural document headings are not content atoms. For pending annotations report accepted-reference delivery separately, set full-question all-delivered to null, and count pending completion tasks explicitly. Report accepted/rejected/unclear/original denominators for each basis and task, including sensitivity to retaining unclear references. Audited upstream keeps the prior word-fourgram proxy; do not relabel it semantic recall. Reference fingerprints include the reviewed overlay, so revised vs legacy are never treated as the same scoring protocol.
- [x] Validate full frozen original47 task/source/selection/parent and overlay hashes before authorized replay. Reuse `rehydrate_task`, `verify_legacy`, `rebind_rows`, `validate_context`, model provenance checks and immutable output reservation/publication. Reproduce the old legacy result first; evaluate all relevant original selection/quota contexts under the new overlay in a fresh report, retaining legacy metrics separately. No policy/model calls.
- [x] Scope tests and two-stage independent review. Run the full47 revised baseline; report source atoms and audited upstream separately, unchanged historical denominators and excluded/pending counts. Reclassify genuine missing evidence before moving to Stage2.

## Stage 2: Relevance-preserving selection and quota comparisons

**Files:** Add an isolated strategy to `app/evidence_selection.py` with tests in `tests/test_evidence_selection.py`; extend new revised runner/tests, not historical runner defaults.

- [x] Register the exact profile matrix before scoring: legacy seed; globally comparable original-question rerank ordering with one feasible best candidate reserved per existing routed source group, fixed doc/source caps; same method with doc cap relaxed; source cap relaxed; both relaxed. All use cached original depth50/rerank scores, <=8 unique chunks and <=5000 tokens; no added reranker/planner/embedding calls.
- [x] Candidate priority is descending finite cached rerank relevance, then recorded parent rank/ID. Hard exclusions (authorization/current version/date/minimum similarity/boilerplate/unknown reason) remain blocked. Real lane alternatives and soft-cap semantics reuse existing `_Pool`/rank-only selector; no comparing scores from different queries or gold-based candidate promotion. No new learned threshold chosen by reference coverage.
- [x] Reserve source evidence inside the same budget, then fill by relevance. Only original question, authorized candidate metadata, actual routing groups and frozen caps enter the runtime strategy. Explain any infeasible source reservation explicitly; do not invent coverage for empty groups. Record context IDs, ranks, source-floor coverage, swaps, cap effects and latency, without raw text.

```python
def authorized_candidate(cid, *, score, rank):
    return {'chunk_id': cid, 'document_id': 'doc-' + cid, 'version_id': 'version',
            'source_sha256': 'source', 'title': 'Document', 'text': 'Relevant source sentence.',
            'rank': rank, 'parent_rank': rank, 'rerank_score': score,
            'lane_type': 'global', 'lane_sha256': 'global-hash',
            'admitted': False, 'excluded_because': 'top_k_full'}

def test_relevance_precedes_lexical_novelty():
    rows = [authorized_candidate('good', score=3.0, rank=1),
            authorized_candidate('background', score=-4.0, rank=2)]
    result = select_relevant_evidence(rows, limit=1, token_budget=5000,
                                    document_quota=None, source_quota=None, required_ids=[])
    assert [r['chunk_id'] for r in result.evidence] == ['good']
```

- [x] Test caps/required sources, duplicate lane alternatives, irrelevant novelty, invalid/missing/nonfinite score, hard exclusions, budget limits and source reauthorization through the real replay path. Spec then quality review before full47 replay.
- [x] Freeze a supplement seed from this revised Dev-only comparison: require all36 eligible tasks measured without errors/drift and no native atom loss versus revised legacy; maximize accepted upstream delivered references, then native atoms, then fewer relaxed caps. If no profile qualifies, use legacy. Selection is Dev tuning, not independent generalization. Save seed/report/overlay/parent hashes and exact configuration before supplement dispatch.

## Stage 3: Exact rejection telemetry and matched supplemental rerun

**Files:** Modify experiment-only `agent/evidence_supplement.py`/tests; extend revised runner or add a small reviewed-supplement runner reusing `scripts/run_evidence_supplement_dev.py`/tests. Keep production Agent graph/controller/adaptive/defaults untouched.

- [x] Add sanitized rejection codes and stages to the existing LangGraph helper: JSON/Pydantic validation, step count/IDs, dependencies/extraction kinds, unsupported operations/foreach/history, invalid placeholders, validation issues, subject focus, literal binding, scope, length and required-source infeasibility. Whitelist field paths and error codes; never export raw model values, prompts or exception messages. Keep the existing acceptance rules, 300/60 output caps, 2 shared policy calls, 1 additional search per arm, 180-second pair deadline and 8/5000 context budget.

```python
def test_rejected_dependency_records_reason_without_values():
    malformed = BridgePlan.model_validate({'steps': [
        {'id': 's1', 'purpose': 'Find entity', 'query': 'carrier',
         'extract': {'name': 'carrier', 'kind': 'entity', 'description': 'Carrier name'}},
        {'id': 's2', 'purpose': 'Find detail', 'query': '{s1.carrier} phone',
         'depends_on': ['s9']}]})
    with pytest.raises(_NoBridge) as error:
        _shape(malformed)
    assert 'second_dependency' in error.value.codes
    assert 'private' not in json.dumps(error.value.public())
```

- [x] Test each rejection family, successful bridge, no changed acceptance for previous cases, scope/version/revocation, per-arm search limits, physical metering and interrupted/null costs. Root spec then quality checks; independent-review quota limitation remains disclosed.
- [x] Extend the old runner by explicit opt-in hooks/configuration for accepted revised annotation and chosen registered seed, preserving the historical legacy path. Replay/evaluator alone receives reviewed gold; graph gate never gets reference coverage, task labels or answers. Freeze actual model endpoint digests and every input/code/corpus/index fingerprint; end-check drift.
- [x] Execute all47 tasks under the chosen frozen seed with parity-rotated matched control/bridge, actual shared planning charged in full to each arm and physical calls recorded once. Include normal no-reference gating/history N/A, failures and interrupted runs. Save a fresh report; do not retune the live method after seeing outcomes. If no matched pairs dispatch, treatment efficacy stays pending even if task execution completes.
- [x] Report rejection counts/fields, accepted-plan/extraction/source-binding/search/select transition counts, native atom/upstream proxy coverage and complete tasks, genuine missing subgroup (post-hoc only), errors/null costs, replacements and matched pair census. Raw third-party text and local checkpoints remain ignored.

## Stage 4: Full verification and report

**Files:** Append a new section to `PROJECT_REPORT.md`; complete this plan. Keep §§20.30–20.31 and all artifacts unchanged except a clearly dated additive correction explaining old proxy limitations.

- [x] Review actual artifact/report arithmetic, annotation provenance/uncertainty, source/version/ACL binding, equal budgets, absence of gold in runtime, no raw-text exports and causal claims. Root performed these checks inline and saved result-audit-v1.json; independent reviewers stopped on quota, so independent final review remains unavailable and is explicitly not claimed.
- [x] Run scoped tests during tasks, then `make test`, `scripts/check_docs.py` and `git diff --check`. Verify original file/HEAD hashes and historical artifact immutability, attributing unrelated concurrent changes without overwriting them.
- [x] Report measurement correction separately from algorithm improvement; generated-answer/semantic/Core success stays pending. Preserve current checkout with no integration menu or Git/deployment mutation, then provide self-contained Chinese results and material limitations.

## Execution provenance update

The blind GPT-6-astra proposer completed drafts v1/v2. The separate same-family source verifier confirmed mechanical bindings and most decisions in partial status, raised U22 uncertainty, but produced no complete per-item review before a quota error. The evaluator worker also stopped on quota. Following executing-plans, root continued inline; it has seen old retrieval outcomes and edited the final proposal, so both retrieval blindness and author independence are false. The serving model ID is unavailable; only the GPT-6 family from session instructions is recorded. Human review count is zero. No independent final approval is claimed. Local v3 failed compilation because the proposer retained a suggested U12 answer; v4 explicitly keeps that suggestion unapplied in local notes. Canonical answers remain unchanged.

Frozen overlay: `.runtime/dev-evidence-repair-20261009/accepted-annotation-v1.json`, SHA256 `894a7074c7fb69ab677809885da5e36de26d4b20c05ab5b4b218e494873ce8ee`. Public manifest: `benchmarks/enterprise_rag/v1/dev/evidence-20261009-v2/manifest.json`. Native 45 content atoms; upstream original26 = accepted19/rejected3/unclear4. All47 tasks remain,36 originally eligible,7 full-question completion pending.

Stage 1 result: `artifacts/dev-evidence-revised-20261009-v2.json` is complete/valid, all47 tasks, no input/code/corpus/index drift or execution errors. Legacy native source atoms45/45, accepted upstream13/19. Historical complementary native38/45 and upstream7/19; replacement variants38/45 and5/19. Remaining accepted losses are four upstream tasks: five admission losses plus one candidate miss; no native content loss. Uncertain full-question completion remains null for seven tasks. Failed preflight v1 is retained separately. The revised review checkbox records the failed independent-review attempt and disclosed inline fallback, not a completed independent final review.

Stage2 result: `artifacts/dev-evidence-relevance-20261009-v1.json` complete/valid/all47/no drift or errors. Legacy/strict/source-relaxed native45/45 and accepted upstream13/19; document-relaxed/both-relaxed native45/45 and15/19. Three upstream references gained, one lost (net+2). All routed source floors feasible; maximum context2112 tokens. Registered choice `relevant_doc` preserves every native reference and uses fewer relaxed caps than `relevant_both`. Independent reviewers remain unavailable; root spec/quality checks and scoped tests substitute with disclosed reduced independence.

Stage3 result: reviewed supplement v1 complete/valid/all47/no errors or drift. Gates26false/16no-valid-bridge/5history, zero extraction/search/matched treatment pairs; efficacy remains pending. Rejection codes now distinguish15second-dependency,7missing-first-extraction,5nonentity-extraction,6second-extraction and1schema-pattern task (codes overlap). HTTP16successful, budget15successful/1parsefailed,21154input/2710output tokens. Native45/45 and upstream15/19 unchanged in all arms. No retuning after results. Full suite1164passed36skipped; independent final review not available and not claimed.

Final verification: make test1164passed36skipped, Ruff/docs passed. Root result audit revalidated frozen overlay and all three complete47-row reports, exact seed/context reproduction, no original questions/facts in public artifacts, zero dispatched pairs kept pending,531 historical artifact hashes and pre-existing report bytes preserved. Unrelated concurrent site/frontend/cleanup/prerender changes were observed and left intact. No Git/database/index/deployment mutations. Independent/human final review unavailable; reduced independence retained as a material limitation, not a fabricated pass.
