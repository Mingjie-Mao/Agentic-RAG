<p align="left">
  <a href="./README.md"><img src="https://img.shields.io/badge/ENGLISH-2563EB?style=for-the-badge&labelColor=2563EB" alt="English"></a>
  <a href="./README_zh.md"><img src="https://img.shields.io/badge/中文-4B5563?style=for-the-badge&labelColor=4B5563" alt="中文"></a>
</p>

# Enterprise-RAG

A retrieval-augmented question answering system for internal company knowledge, built so that every
claim about its quality can be reproduced from a frozen evaluation set rather than taken on trust.

Most open-source RAG projects compete on feature breadth. This one competes on whether the numbers
hold up: retrieval and generation are measured separately, permission filtering is asserted on every
run, and results that turned out to be unsound are recorded as such instead of being quietly dropped.

> **Status:** stages S0–S5 complete, S6–S8 not started. Single-node deployment for a small team.
> Throughput and latency figures below describe one Apple M4 host, not a production SLA.

## What it does

An employee asks a question in Chinese or English. The system retrieves passages from the documents
that employee is allowed to read, answers only from that evidence, and returns citations that open
the original file at the exact page, paragraph or cell the answer came from. When the evidence is
insufficient, or two policies contradict each other, it says so instead of guessing.

- **Grounded answers.** The model selects numbered source spans; the server resolves each span back
  to stored text and re-authorises it before the answer is returned. An answer whose citations fail
  verification is withheld, not repaired.
- **Permission-aware retrieval.** Access control is applied inside the query, not after it. Across
  every evaluation run to date there have been zero out-of-scope hits and zero hidden-document leaks.
- **Document fidelity.** PDF, DOCX, XLSX and Markdown are parsed into one structure that keeps
  heading paths, real page numbers, layout coordinates and cell ranges. A spreadsheet formula with no
  cached value is reported as missing rather than treated as a computed number.
- **Reproducible evaluation.** A frozen corpus, ground truth bound to source locations, and a runner
  whose checkpoints are keyed to a configuration digest, so results from one snapshot can never be
  silently reused for another.

## Measured results

All figures come from the frozen development split and can be re-run with the commands below.
Held-out questions have never been executed or inspected.

### Retrieval — s3-v2 development split, 147 questions

| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Retrieval p95 |
| --- | --- | --- | --- | --- | --- |
| BM25 (CJK analyzer) | **0.984** | **0.992** | **0.927** | **0.944** | 0.04 s |
| Dense (BGE-M3, 1024-d) | 0.903 | 0.935 | 0.840 | 0.861 | 0.23 s |
| Hybrid (reciprocal rank fusion) | 0.935 | 0.976 | 0.900 | 0.918 | 0.26 s |

Denominator is the 124 questions that carry document-level ground truth; unanswerable and
permission-probe questions are excluded from retrieval metrics.

Hybrid ranks below BM25 on every retrieval metric, because reciprocal rank fusion pulls the stronger
run toward the weaker one when one path dominates. **It is still the default**, because retrieval
metrics turned out to mispredict end-to-end quality — see below.

### End to end — s3-v2 development split, 147 questions, prompt `grounded-v5-conflict-gated`

| Configuration | Correct answer state | Literal fact coverage | Conflict recall | Conflict precision | Out-of-scope hits |
| --- | --- | --- | --- | --- | --- |
| BM25 | 84.4% | 84.6% | 10/13 = 77% | 10/10 = 100% | 0 |
| Dense | 85.0% | 82.1% | 9/13 = 69% | 9/9 = 100% | 0 |
| **Hybrid (default)** | **89.1%** | **89.1%** | **11/13 = 85%** | 11/11 = 100% | 0 |

Hybrid retrieval achieved the best observed end-to-end accuracy (89.1%), although the improvement
over BM25 was not statistically significant on the current evaluation set (McNemar exact test,
p = 0.19, 147 paired questions). It is the default because it is best or tied on every measured
dimension and worse on none; the claim made here is "best observed", not "proven better".

**Document-level retrieval metrics mispredicted end-to-end quality.** Hybrid has the lowest Recall@5
of the three yet the highest answer accuracy. With an evidence budget of four passages, what decides
the outcome is which four slots get filled, not whether the gold document appears in the top ten.
Fusion promotes passages both runs agree on, so the budget is spent on consensus evidence.

Fact coverage is normalised string matching, not human-judged correctness. Semantic correctness,
faithfulness and citation entailment are **not yet measured**.

### Reranking — implemented, and switched off

A `bge-reranker-v2-m3` cross-encoder rescores the top 30 fused candidates. It lifts retrieval
clearly — Recall@5 0.935 → 0.968, MRR 0.900 → 0.935 — and end to end it does almost nothing:
89.1% → 90.5% correct answer state, McNemar p = 0.79 on 147 paired questions.

It stays off, under the same rule that put hybrid retrieval on: adopt what is best or tied on every
measured dimension and worse on none. Reranking fails that test — conflict precision drops from
100% to 92.3% and median latency rises 22% — so a 0.79 p-value does not buy it a default.

Worth recording: reranking moved the retrieval metrics a lot and end-to-end quality barely at all,
while fusion did the opposite. **Retrieval metrics and answer quality are not monotonically related
here**, which is why both are measured rather than one used as a proxy for the other.

### External validity — public benchmark subset, 39 questions

EnterpriseRAG-Bench v1.0.0, every question in the two published GitHub slices whose gold documents
are complete, plus 100 distractors. English documents, a different distribution from the authored
Chinese corpus.

| Configuration | Recall@5 | Correct answer state |
| --- | --- | --- |
| BM25 | 0.897 | 33/39 = 84.6% |
| Dense | 0.897 | 29/39 = 74.4% |
| **Hybrid (default)** | **0.974** | 33/39 = 84.6% |

**Hybrid's retrieval advantage is larger here than on the authored corpus**, the opposite of its
last-place retrieval ranking there. That supports the reading that BM25's lead on the authored corpus
comes from its distinctive noun phrases rather than from any general property of lexical search.

Literal fact coverage is **not applicable** on this dataset and is reported as such rather than as a
number: upstream gold is prose, the answers paraphrase it correctly, and substring matching built for
short Chinese values scores that near zero regardless of correctness.

This subset measures external validity, not statistical power. Its questions come from a different
population and are never pooled into the significance test above.

### Multi-turn follow-ups — 40 dialogues, 160 turns

A follow-up like "and the request timeout?" carries no subject, so single-turn retrieval fails it.
Four groups were compared on retrieval, since rewriting exists to change what is retrieved.

| Group | Overall R@5 | Follow-up R@5 | Anaphora R@5 | Condition fidelity | Topic-shift contamination | Rewrite latency |
| --- | --- | --- | --- | --- | --- | --- |
| A — no rewriting | 0.825 | 0.650 | 0.150 | 100% | 0% | 0 s |
| **B — prepend previous question (adopted)** | **0.988** | **0.975** | **0.900** | **100%** | **0%** | **0 s** |
| C — LLM rewrite, no history | 0.825 | 0.650 | 0.150 | 100% | 0% | 2.70 s |
| D — LLM rewrite + last 3 turns | 0.944 | 0.900 | 0.850 | 95% | 2.5% | 4.51 s |

**All of the benefit comes from the history, none from using a model to rewrite.** Group C is
identical to A on every metric, because without history there is nothing to resolve the ellipsis
against. A rule that costs no model call and no latency beat the LLM rewrite outright, and the LLM
variant failed two hard gates: it dropped a stated value in one case and rewrote a genuinely new
question back into the previous subject in another.

With empty history the rule is the identity function, so every single-turn result above stands
unchanged.

### Failure attribution

Of 86 failures in the pre-fix baseline, **6 (7%) were retrieval misses and 80 (93%) occurred after
the correct sources had already been retrieved.** This is why work went into answer-state logic
before recall optimisation.

### A defect this evaluation caught

The conflict detector was a second model call whose boolean verdict overwrote the answer state with
no evidence required. It fired on 24 of 120 questions while only one was a genuine conflict —
**4.2% precision**. It was gated to require two mutually contradictory span IDs from two different
documents; precision reached 100% and the correct-state rate rose by roughly 19 points.

Recall could not be measured at the time: the frozen v1 set contained a single conflict question. The
v2 set added 12 authored conflicts and 15 near-miss negatives — different metrics, different
subjects, superseded versions, and pairs that disagree across documents without contradicting each
other. On that set the gated detector reaches **85% recall at 100% precision**, and none of the 15
negatives is misreported.

## Architecture

```text
Ingest   upload → authorise → store immutable original → parse → chunk → embed
         → write pending index → verify searchable → promote active version

Query    identify user → retrieve within permitted version set → fuse → re-check
         version and grant → assemble evidence → generate → verify citations
         → re-authorise → return answer with source locations
```

| Layer | Choice | Reason |
| --- | --- | --- |
| API | Python, FastAPI, Pydantic | Explicit schemas; evaluation reuses the same retrieval code |
| Business store | PostgreSQL, SQLAlchemy, Alembic | Identity, grants, versions, jobs, answer provenance |
| Search | OpenSearch — BM25 and vector in one backend | One chunk ID space, one filter, one update path |
| Parsing | Docling, with openpyxl for cell-level anchors | Reuse a maintained parser; own the structure and provenance |
| Embedding | BGE-M3 (`bge-m3:567m`), dense output only | Reproducible, adequate for Chinese and English identifiers |
| Generation | Qwen2.5 7B Instruct via Ollama on Metal | Fixed seed and temperature so runs are comparable |
| Jobs | PostgreSQL job table with leases and retries | No message broker in the first release |
| Frontend | React, TypeScript, PDF.js | Document state, evidence drawer, retrieval inspector |

## Quickstart

Requires Docker, Node.js 22.12+, Python 3.13 and `uv`. First run downloads models and needs several
GB of disk.

```bash
make setup && make infra && make models && make migrate && make seed && make web && make run
```

Open `http://127.0.0.1:8000`. Demo accounts are listed on the sign-in page; the password comes from
`RAG_DEMO_PASSWORD` in `.env`. All seeded content is fictional.

| Account | Scope |
| --- | --- |
| `admin@xingqiao.demo` | Tenant administrator |
| `engineer@xingqiao.demo` | Engineering group |
| `support@xingqiao.demo` | Support group |
| `admin@haichuan.demo` | Second tenant, for isolation checks |

## Reproducing the evaluation

```bash
make test              # 82 unit and logic checks (17 need integration or layout flags)
make integration       # real database, permission and job-recovery checks
make s3-validate       # audit ground truth against parsed source text
make s3-freeze         # pin input hashes and model digests
make s3-retrieval      # BM25 / Dense / Hybrid retrieval metrics
make s3-generate       # end-to-end answers, citations and failure classes
```

Four guarantees are enforced by the harness rather than by convention:

1. Ground truth reaches scoring only — never retrieval, prompts, rewriting or caches.
2. `--split holdout` is refused outright until the final evaluation.
3. Checkpoint rows carry a configuration digest; resuming across snapshots fails loudly. This caught
   a run that silently reused results from a superseded corpus and reported identical metrics.
4. New evaluation material may not discuss a subject the existing corpus already answers. Adding a
   second opinion on a settled subject turns previously correct ground truth into a lie.

## Verified and not verified

Treating these two lists as one is the most common way RAG benchmarks mislead, so they stay separate.

**Verified with artefacts in `artifacts/`**

- Four document formats parse with checkable structure and source locations; 54 checks, 0 skipped.
- Twenty development questions pass a single complete browser run, 23/23 including upload and format
  checks.
- Citation identity: every returned citation resolves to stored text, version and location.
- Permission filtering: zero out-of-scope hits and zero hidden-document leaks across all runs.
- Restart recovery: files, grants, versions, chunks and vectors survive a full stack restart.

**Not verified**

- Semantic correctness, faithfulness and citation entailment — only literal matching so far.
- Public benchmark subset — adapter ready, never executed.
- Held-out split — 80 questions, never run, never inspected.
- Independent human review of ground truth — recorded as **0**. Programmatic auditing proves a fact
  appears in its own source; it does not prove the question or its answer is well posed.

## Limitations

1. Scanned PDFs and PPTX are not supported; a PDF with no text layer fails loudly rather than
   indexing empty content.
2. Hybrid retrieval is the default on an observed, not statistically significant, advantage. The
   public benchmark subset has been run and agrees; the held-out split remains sealed until final
   acceptance and is the intended tie-breaker for significance.
3. In-page PDF highlighting applies to documents ingested by the current parser. The thirty seed
   documents predate it and will be reprocessed as part of version replacement.
4. No working memory and no long-term memory. `/api/chat` stays stateless; a client may replay the
   user's own earlier questions, which complete a follow-up into a searchable query and nothing else
   — the permitted document set is always recomputed for the current user, and replayed text grants
   no access. Stored answer history is an audit record and never returns to the model.
5. Conflict detection is measured at 85% recall and 100% precision on 13 conflicts and 15 negatives.
   Three conflicts are still missed, and the sample is small.
6. Single-host demo deployment. Rate limiting, backups, TLS, multi-node operation and concurrency
   headroom are out of scope for this release.

## Project documentation

[`PROJECT_PLAN.md`](./PROJECT_PLAN.md) is the single source of truth: scope, stage status, technical
decisions, experiment results, failure samples and reproduction steps all live there. Machine
artefacts under `artifacts/` are execution evidence, not a second set of documentation.

All sample documents, questions and company names in this repository are fictional and authored for
this project. They do not represent any real organisation or policy.
