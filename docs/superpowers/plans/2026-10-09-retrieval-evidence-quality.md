# Retrieval Evidence Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose where supporting evidence disappears, implement bounded opt-in retrieval candidates, and measure them on the existing Benchmark v1 Dev selection before changing defaults.

**Architecture:** Keep OpenSearch hybrid retrieval and LangGraph unchanged as the underlying frameworks. Add a candidate-pool override to the existing authorization boundary, reuse existing passage windows and index-header helpers, and build a read-only Dev evaluator that distinguishes candidate evidence from admitted and prepared generation context. Gold annotations stay exclusively in the evaluator; historical versions and unanswerable tasks have explicit applicability, not invented zero recall.

**Tech Stack:** Python 3.13, PostgreSQL/SQLAlchemy, OpenSearch, Ollama/BGE-M3, existing BGE reranker and pytest.

---

## Working boundaries

- The user selected implementation in the current directory. Preserve pre-existing edits and untracked adaptive-Agent work. The local checkpoint is `.runtime/retrieval-quality-20261009/initial-tracked.patch` plus file hashes.
- Baseline verification: `.venv/bin/pytest -q` produced 686 passed / 36 skipped on 2026-10-09. Real local services require execution outside the network sandbox; a read-only health check confirmed HTTP 200 and database `SELECT 1`.
- Only `.runtime/benchmark-package/v1/dev/tasks.json` and its frozen selection are used. Do not execute or tune on Core, Security or External suites; the historical MultiHop tasks included in Dev remain Dev.
- Existing experiment files, annotation files and published document versions remain intact. New output is exclusive-create and contains IDs, hashes and numerical derivations only. Checkpoints and raw local material stay under `.runtime`.
- Do not enable candidate depth, passage windows, text enrichment or reranking by default from retrieval-only results. Final correctness and semantic evidence coverage remain null until the applicable answer review is done.
- No source-file commit includes third-party questions, answers or article text. No unrelated file is staged or committed.

## File map

| File | Responsibility |
| --- | --- |
| `scripts/retrieval_quality_metrics.py` | Pure offline source-bound evidence delivery and paired diagnostic summaries |
| `tests/test_retrieval_quality_metrics.py` | Missing telemetry, version binding, phase loss and paired-denominator tests |
| `app/config.py` | Bounded opt-in candidate depth, default zero |
| `app/retrieval.py` | Apply candidate depth within existing ACL scope; retain pre/post ranking telemetry |
| `tests/test_retrieval_candidate_depth.py` | Default compatibility, widened candidates, fixed final budget and ACL behavior |
| `scripts/run_retrieval_quality_dev.py` | Hash-bound Dev-only runner, profiles, local context reads and derived artifacts |
| `tests/test_retrieval_quality_dev.py` | Dev selection guard, no gold in retrieval calls, immutable output and profile validation |
| `scripts/retrieval_quality_index.py` | Optional isolated text-enrichment index using original vectors and source IDs |
| `tests/test_retrieval_quality_index.py` | No production-index mutation, source/vector preservation and conservative cleaning |
| `PROJECT_REPORT.md` | Actual results, adopted/rejected candidates, limitations and reproduction commands |

## Task 1: Offline, source-bound phase diagnostics

- [x] Write tests before implementation. Test that a fact present in document B cannot satisfy gold for document A; a mismatched source SHA cannot satisfy a version-bound reference; and missing phase telemetry produces null rather than zero.

```python
def test_source_identity_is_required():
    fact = {"id": "f1", "document_id": "a", "text": "The limit is thirty minutes.",
            "source_sha256": "sha-a", "basis": "upstream_word4_proxy"}
    row = {"chunk_id": "b1", "document_id": "b", "source_sha256": "sha-a",
           "text": "The limit is thirty minutes."}
    assert phase_coverage([fact], [row])["delivered"] == 0


def test_missing_telemetry_is_not_a_measured_failure():
    assert phase_coverage([], None) is None


def test_wrong_version_cannot_satisfy_a_reference():
    fact = {"id": "f1", "document_id": "a", "text": "The limit is thirty minutes.",
            "source_sha256": "old", "basis": "upstream_word4_proxy"}
    row = {"chunk_id": "a1", "document_id": "a", "source_sha256": "new",
           "text": "The limit is thirty minutes."}
    assert phase_coverage([fact], [row])["delivered"] == 0
```

- [x] Run `.venv/bin/pytest -q tests/test_retrieval_quality_metrics.py`; confirm the new imports/tests fail before implementing them.
- [x] Implement `supporting_facts(task)`, `phase_coverage(facts, evidence)`, `diagnose(task, phases)` and `summarize(rows)` in the metrics module. Use `task['upstream_gold_evidence'][i]['fact']` and the existing `multihop_retrieval_eval.fact_delivered` for external Dev references. Map article titles through the existing `document_id` helper and bind them to `gold_evidence` source hashes. For native Dev sources use normalized character 4-gram coverage of the annotated source span, identified as `source_span_char4_proxy`; never call it semantic fact recall. Keep these bases separate in summaries.

```python
def source_matches(reference, row):
    return (row.get("document_id") == reference["document_id"]
            and row.get("source_sha256") == reference["source_sha256"])


def normalized_character_grams(value):
    text = "".join(value.casefold().split())
    return {text[i:i + 4] for i in range(max(0, len(text) - 3))} or ({text} if text else set())
```

- [x] Return reference count, matched reference IDs, all-delivered and first supporting ranks per phase. Diagnose `candidate_missing`, `admission_loss`, `context_loss`, `proxy_complete` and explicit `not_applicable` for historical-version single retrieval and no gold. A failed/unavailable retrieval has null coverage and an error status, not zero.
- [x] Pair profiles by task ID and source references; reject duplicate/mismatched tasks. Report complete-pair counts and missing-pair counts separately. Aggregate by evidence basis and category; keep `strict_task_success` and semantic recall null.
- [x] Run `.venv/bin/pytest -q tests/test_retrieval_quality_metrics.py` and `.venv/bin/ruff check scripts/retrieval_quality_metrics.py tests/test_retrieval_quality_metrics.py`; inspect results and perform separate spec and quality reviews.

## Task 2: Bounded candidate depth without changing the answer budget

- [x] Add behavior tests using fake scoped search: default behavior keeps prior sizes; depth 50 or 100 requests more ranked candidates but still admits at most the original final limit; requests never include unauthorized versions; final evidence is reauthorized. Preserve existing callable `_FUSE_GOAL` edits.
- [x] Run `.venv/bin/pytest -q tests/test_retrieval_candidate_depth.py` and confirm the depth override tests fail on the baseline.
- [x] Add a validated configuration field in `app/config.py`:

```python
retrieval_candidate_depth: int = Field(default=0, ge=0, le=100)
```

- [x] In `retrieve_authorized`, keep `limit` unchanged and widen candidate search only for an explicit nonzero depth. Source lanes, global lanes and optional planned searches use the same scoped `run` function.

```python
candidate_depth = getattr(cfg, "retrieval_candidate_depth", 0)
search_limit = max(search_limit, candidate_depth)
```

- [x] Apply `max(original_lane_size, candidate_depth)` to source lanes. Do not infer authorization from candidates and do not disable source/document quotas or context-token limits. Record each lane's retrieval position before reranking and retain the rerank score when available:

```python
lanes = [(lane, [{**hit, "retrieval_rank": rank}
                 for rank, hit in enumerate(hits, 1)]) for lane, hits in lanes]
```

- [x] Include `retrieval_rank` and `rerank_score` in candidate diagnostics only; no original text enters public candidate logs.
- [x] Run `.venv/bin/pytest -q tests/test_retrieval_candidate_depth.py tests/test_source_routing.py tests/test_retrieval_queries.py tests/test_query_planner.py` and Ruff. Perform separate spec and quality reviews before proceeding.

## Task 3: Dev runner and isolated keyword-context experiment

- [x] Write runner tests first. Reject a suite whose split is not `dev`, a selection hash/ID mismatch, existing result paths, duplicate profile names or unsupported overrides. Record the exact retrieval calls and prove that gold document IDs, reference text and accepted answers are never passed to retrieval or passage preparation.
- [x] Implement the CLI with required `--output`, optional `--suite` defaulting to the local v1 Dev suite, optional `--task-ids` for an explicitly labeled subset, and `--profiles` from the registered map below. Use fresh local artifacts and the existing checkpoint/atomic-write utilities rather than a new execution or recovery engine.

```python
PROFILES = {
    "baseline": {},
    "depth50": {"retrieval_candidate_depth": 50},
    "depth100": {"retrieval_candidate_depth": 100},
    "depth50_window": {"retrieval_candidate_depth": 50, "passage_window_enabled": True},
    "depth50_rerank": {"retrieval_candidate_depth": 50, "passage_rerank": True},
    "keyword_context": {},
}
```

- [x] Freeze the suite/selection hashes, task IDs, all profile overrides, effective safe settings, model identity, input source hashes and application/script code hashes before the first retrieval. Record the generator as not called. Keep top-k fixed at 8 and the same 5000-token context budget for all profiles. Interleave profile order across tasks. Validate that actual corpus source hashes and active-version identities remain unchanged at the end.
- [x] Collect raw ranked candidates by reauthorizing their chunk IDs, then collect admitted evidence and prepared context through the existing `prepare_passages` function for the window profile. Bind every evidence row to the actual `DocumentVersion.content_hash`. Catch per-task dependency failures and preserve null telemetry; do not silently drop failures.
- [x] Do not score history-dependent native tasks as if active-only retrieval could supply their old versions. Keep their IDs in the denominator accounting with `not_applicable` for this component experiment; null tasks report evidence counts separately and never receive fabricated recall or correctness.
- [x] Create `scripts/retrieval_quality_index.py` for `keyword_context`: use a unique isolated OpenSearch index and the official `_reindex` API to preserve the full original document population, mapping and vectors. Copying only the readable active subset would change BM25 corpus statistics and confound the experiment; out-of-scope documents must remain untouched and are never supplied to evaluation/model calls. Only transform reauthorized active chunks in the Dev users' readable scope, prepending existing `chunk_header(..., {'chunk_context': 'document_header'})` text. Use existing `is_web_boilerplate` only for entire furniture chunks; preserve all mixed content. Do not mutate PostgreSQL, raw files, published versions or the production index. Name this accurately as keyword-context enrichment with unchanged vectors, not contextual embeddings. Keep index identity, population counts and a source-hash ledger in the report.
- [x] For the index helper tests, use a fake search transport to inspect create, reindex and bulk requests. Verify untouched source population remains present, original vector/tenant/version/document IDs are preserved, a mixed article/advertisement chunk remains, and a target equal to the production index is rejected before any write.
- [x] Write outputs containing only task IDs, hashes, phase metrics, causes, rank positions, costs, status and pairs. Full retrieved text and questions remain in memory/local input files. Use exclusive output reservation; retain failed/interrupted runs. Checkpoints are local runtime files.
- [x] Run `.venv/bin/pytest -q tests/test_retrieval_quality_dev.py tests/test_retrieval_quality_index.py tests/test_retrieval_quality_metrics.py` and Ruff; inspect every result and perform separate spec and quality reviews.

## Task 4: Registered Dev measurements, decisions and final verification

- [x] Run a full Dev retrieval-only comparison using the first four profiles. The frozen pre-run registration prevents adding configurations after seeing those results.

```bash
.venv/bin/python scripts/run_retrieval_quality_dev.py \
  --profiles baseline,depth50,depth100,depth50_window \
  --output artifacts/retrieval-quality-dev-20261009-v1.json
```

- [x] Run the predeclared rerank and isolated keyword-context profiles with a separate immutable registration/output, including baseline under the same conditions:

```bash
.venv/bin/python scripts/run_retrieval_quality_dev.py \
  --profiles baseline,depth50_rerank,keyword_context \
  --output artifacts/retrieval-quality-dev-20261009-v2.json
```

**Operational correction:** `v1` completed valid with no drift. `v2` failed before its first retrieval because the full synchronous `_reindex` exceeded the existing 60-second HTTP timeout. Its artifact and isolated index remain intact. Reuse OpenSearch official asynchronous `_reindex` plus Tasks API, preserve its task ID locally, review that minimal fix, then run the same preregistered profiles with a fresh filename (no parameter changes):

```bash
.venv/bin/python scripts/run_retrieval_quality_dev.py \
  --profiles baseline,depth50_rerank,keyword_context \
  --output artifacts/retrieval-quality-dev-20261009-v3.json
```

`v3` also stopped before retrieval because the local OpenSearch long-wait task query returned a dependency error; the asynchronous copy subsequently completed. Preserve this failed artifact/index as well. Query the official task status without long-wait parameters and pause 1 second only between unfinished responses. After review, retry the same profiles as `artifacts/retrieval-quality-dev-20261009-v4.json`; no experimental parameters change.

- [x] Analyze whether candidate coverage improved, whether admission lost that improvement, and whether the existing window preserved or discarded necessary references. Report costs and eligibility by native/source-span and upstream-fact bases, not a misleading pooled fact metric. An observed retrieval win does not authorize a default switch without answer review. If a candidate loses, retain its result and record the reason.
- [x] Add a dated section to `PROJECT_REPORT.md` with methods, exact denominators, artifact links, actual changes and adoption decision. Core and system accuracy remain pending; do not change historical sections or README headline claims.
- [x] Verify all production and new tests, lint and documentation:

```bash
make test
git diff --check
```

- [x] Compare pre-existing files against the initial checkpoint: only this task's documented hunks may differ; all other tracked and untracked Agent/experimental work must remain. Review the final implementation separately for spec coverage and code quality.
- [x] Report the completed code, actual Dev measurements, rejected candidates, verification counts, remaining pending quality review, and exact artifact/document paths. The user requested a completion report; do not interrupt with an integration menu or push unrelated work.

**Final verification:** All four tasks completed. Valid Dev runs v1/v4 retained; failed pre-retrieval v2/v3 retained. Separate final review approved the implementation and report. Fresh `make test`: 801 passed / 36 skipped; Ruff and documentation consistency passed. Original user edits preserved by hash comparison. Work remains in the user-selected current checkout.
