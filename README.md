# Agentic-RAG

**Auditable, permission-aware RAG for enterprise knowledge.**

Agentic-RAG is an enterprise RAG system with ACL-aware retrieval, verifiable citations, document versioning, and controlled Agent workflows. It answers only from documents the current user may read, cites the exact source location, refuses when the evidence is insufficient, and reports a conflict instead of picking a side.

## Features

- Permission filtering inside the retrieval query, re-checked before an answer is returned
- Hybrid search: BM25 + BGE-M3 + RRF, with per-publication routing for multi-source questions
- Verifiable citations: PDF page and bounding box, DOCX heading and paragraph, XLSX sheet and cell
- Safe document versioning (index first, then publish) and immediate permission revocation
- Fixed workflow and dynamic Agent over seven bounded tools, with auditable tool traces
- Query recovery, multi-hop retrieval, and version comparison
- Reproducible benchmarks with frozen inputs, one-shot holdouts, and failure analysis

## Results

| Benchmark | Result |
| --- | ---: |
| Frozen holdout (80 questions, run once) — answer-state accuracy | 90.0% |
| Frozen holdout — literal fact coverage | 81.9% |
| Unauthorized retrieval / hidden-document leakage, all runs | 0 |
| MultiHop-RAG external (150 unseen questions, run once) — answer correct | 101/150 |
| Hard Benchmark (development set, 21 shared tasks) — RAG | 12/21 |
| Hard Benchmark — Fixed workflow | 19/21 |
| Hard Benchmark — Dynamic Agent | 14/21 |

The controlled workflow currently outperforms the dynamic Agent while using less latency and fewer tokens. The main open gap is yes/no comparison questions on the external set, which still score below a majority-class baseline; controlled experiments trace it to the 7B generator and to noisy evidence, not to missing documents. Details, ablations and limitations are in the report.

## Stack

FastAPI · PostgreSQL · OpenSearch · BGE-M3 · Qwen2.5 7B · Ollama · React · TypeScript

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

Open <http://127.0.0.1:8000>. Demo accounts are listed on the login page; the password is `RAG_DEMO_PASSWORD` in `.env`. All seed documents, questions and company names are fictional.

## Demo

<https://agentic-rag.pages.dev> — a static showcase generated from the stored run artifacts (not a live system).

## Documentation

See [`PROJECT_REPORT.md`](./PROJECT_REPORT.md) (Chinese) for architecture, experiments, ablations, limitations, failure analysis, and the full evaluation commands.
