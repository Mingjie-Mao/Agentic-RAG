# Agentic-RAG

**Auditable, permission-aware RAG for enterprise knowledge.**

Agentic-RAG is an enterprise RAG system with ACL-aware retrieval, verifiable citations, document versioning, and controlled Agent workflows. It answers only from documents the current user may read, cites the exact source location, refuses when the evidence is insufficient, and reports a conflict instead of picking a side.

## Features

- Permission filtering inside the retrieval query, re-checked before an answer is returned
- Hybrid search: BM25 + BGE-M3 + RRF, with per-publication routing for multi-source questions
- Verifiable citations: PDF page and bounding box, DOCX heading and paragraph, XLSX sheet and cell range
- Safe document versioning (index first, then publish) and immediate permission revocation
- LangGraph workflow, dynamic Agent, Hybrid Agent and an experimental Planner over seven bounded tools, with auditable tool traces; `auto` uses the fixed workflow, or the bounded Hybrid Agent when the next step depends on an observation
- Query recovery, multi-hop retrieval, and version comparison
- Reproducible benchmarks with frozen inputs, one-shot holdouts, and failure analysis

Execution budgets and resumable benchmark checkpoints are implemented. Literal comparison, condition, and version contracts are enabled by default after Dev validation. Repair history is in [project report §20.26–20.27](PROJECT_REPORT.md#2026-统一评测包langgraph-迁移与-dev-修复2026-10-02-至-10-05) (generation recovery, entity/time scope, citation checks, P95). The latest frozen Dev 47 run (v6) gives strict success RAG 38, Workflow 38, Dynamic 38, Hybrid 37; its new answers were reviewed by Claude Opus 5.5 (not independent, not the GPT review the protocol specifies), so these are Dev diagnostics, not headline results. Formal Core acceptance remains pending.

The Planner (`mode=planner`, explicit only) makes one structured plan; the program executes it and accepts a bridge value only when it appears verbatim in the cited source. On a separate observation-dependent Dev set (30 tasks) it reached 14/30 strict success versus 4–7/30 for the four main methods, at P50 65 s. It was iterated on the same Dev set, its 40-task Test split has not been reviewed or run, and it is not a default path. See [project report §20.28–20.29](PROJECT_REPORT.md#2029-规划器planner一次规划程序执行桥接值逐字核对2026-10-08).

Agent node execution and recovery use LangGraph with its official PostgreSQL checkpointer (SQLite in tests). ACL, version validation and cumulative budgets remain application rules. See [project report §2.3.1](PROJECT_REPORT.md#231-langgraph-编排与恢复边界2026-10-02-迁移) for recovery boundaries and validation.

Slot evaluation and isolated post-completion shadow are implemented as candidates; semantic control remains off. The user-approved benchmark protocol accepts explicit GPT reviews. Formal Core acceptance has not run; old human-review gates for semantic-control calibration remain separate from the new benchmark protocol.

## Benchmark and Results

One [Enterprise-RAG Benchmark Package](benchmarks/enterprise_rag/v1/README.md): **Dev 47 / Core Test 70 / Security 16 / External 150**. Only Dev permits tuning. Core contains seven categories of ten tasks; Security is separate. The fixed external selection was evaluated previously and is not a fresh unseen validation.

The headline metric is **Strict Task Success Rate on all 70 Core tasks**: complete supported facts, correct condition/version scope, supported citations, correct abstention and safe execution. Retrieval, policy, answer quality, cost and per-category results are reported separately. Expected tool paths are diagnostic and never required for a pass.

| Method | Core Strict Task Success | Fact Recall | Retrieval Recall | Citation | Steps | Latency |
| --- | --- | --- | --- | --- | --- | --- |
| Plain RAG | Pending | — | — | — | — | — |
| Workflow | Pending | — | — | — | — | — |
| Dynamic Agent | Pending | — | — | — | — | — |
| Hybrid | Pending | — | — | — | — | — |

The package and the comparison/ablation matrix are prepared. Same-session GPT annotation review covers Core 70 and Security 16; it does not establish system accuracy or independent human gold. Formal freezing, execution and blinded GPT answer review remain; External annotation review is pending. No historical percentage is substituted for a Core result. Previous measurements are preserved in [project report §22](PROJECT_REPORT.md#22-历史评测记录不进入-v1-主表).

## Stack

FastAPI · LangGraph · PostgreSQL · OpenSearch · BGE-M3 · Qwen2.5 7B · Ollama · React · TypeScript

## Quick Start

Requires Docker, Node.js 22.12+, Python 3.13 and `uv`. The first run downloads several GB of models.

```bash
make setup
make infra
make models
make migrate
make seed
make web
make run
```

## Demo

<https://agentic-rag.pages.dev> — project page (Chinese, with an English toggle): screenshots, architecture and evaluation design. Its result tables are September 2026 historical experiments, labeled as such; no Benchmark v1 accuracy is shown until Core has run.

<https://agentic-rag.pages.dev/live> — the full system, relayed through a Cloudflare tunnel to the author's machine. The visitor account is read-only with a daily quota, and the page works only while that machine and the tunnel are up. After a tunnel restart, run `scripts/publish_live_page.sh <new tunnel URL>`.

## Documentation

See [`PROJECT_REPORT.md`](./PROJECT_REPORT.md) (Chinese), the single project document, for architecture, experiments, ablations, limitations, failure analysis, and the full evaluation commands. Current progress, open problems and the plan are summarized in [§20.1](PROJECT_REPORT.md#201-现状总览已完成当前问题与计划2026-10-08).

See [RAG and memory](RAG_AND_MEMORY.md) (Chinese) for the actual retrieval pipeline, three memory layers, and their storage locations.
