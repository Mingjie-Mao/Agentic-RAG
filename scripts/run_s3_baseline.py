"""Frozen S3 baselines: BM25 vs Dense on one snapshot. Gold only reaches scoring, never retrieval."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import os
import httpx
import fcntl

from app.clients import DependencyError, Models
from app.rerank import rerank
from app.config import settings
from app.evaluation import (
    aggregate,
    allowed_sources,
    answer_scores,
    canonical,
    digest,
    generate_answer,
    identities,
    prepare_index,
    read_manifest,
    retrieval_scores,
)

RETRIEVAL_DEPTH = 50
DEPENDENCY_ATTEMPTS = 6
DEPENDENCY_BACKOFF = 10


def await_dependencies(search, models, attempts=DEPENDENCY_ATTEMPTS):
    """A local model server or search node can drop out mid-run; that is transient,
    and losing hours of completed work to it is not acceptable. Only outages are
    retried — every other failure still stops the run."""
    for attempt in range(attempts):
        try:
            search.request("GET", "/_cluster/health")
            models.embed(["健康检查"])
            return True
        except DependencyError as exc:
            if attempt == attempts - 1:
                print(f"Dependencies still unavailable after {attempts} attempts: {exc}", flush=True)
                return False
            delay = DEPENDENCY_BACKOFF * 2**attempt
            print(f"Dependency unavailable ({exc}); retrying in {delay}s", flush=True)
            time.sleep(delay)
    return False


def model_versions():
    cfg = settings()
    response = httpx.get(cfg.ollama_url + "/api/tags", timeout=20, trust_env=False)
    response.raise_for_status()
    available = {m["name"]: m["digest"] for m in response.json()["models"]}
    return {
        "embedding": {"name": cfg.embed_model, "digest": available[cfg.embed_model]},
        "generation": {"name": cfg.chat_model, "digest": available[cfg.chat_model]},
    }


def run_config(directory, split, retrievers, generate, snapshot):
    cfg = settings()
    dataset = Path(directory)
    return {
        "snapshot": snapshot,
        "dataset": str(dataset),
        "split": split,
        "retrievers": retrievers,
        "generated": generate,
        "retrieval_depth": RETRIEVAL_DEPTH,
        "dataset_sha256": {
            name: hashlib.sha256((dataset / name).read_bytes()).hexdigest()
            for name in [
                "manifest.json",
                "questions.json",
                "split.json",
                "selection.json",
                "freeze.json",
            ]
            if (dataset / name).exists()
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path("app").glob("*.py"))
            + [Path(__file__).resolve().relative_to(Path.cwd())]
        },
        "models": model_versions(),
        "generation": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 700,
            "conflict_predict": 200,
            "prompt": "grounded-v5-conflict-gated",
        },
        "pipeline": {
            "parser": cfg.parser_version,
            "chunking": cfg.chunk_strategy,
            "chunk_chars": cfg.chunk_chars,
            "chunk_overlap": cfg.chunk_overlap,
            "top_k": cfg.top_k,
            "min_similarity": cfg.min_similarity,
            "context_token_budget": cfg.context_token_budget,
        },
        "evidence_admission": "Top 4 chunks within 5000 estimated tokens, without score cutoff for either retriever; production dense cutoff is not used.",
        "metric_definition": "Document ranking deduplicated from first 50 retrieved chunks; document Recall@5/10, MRR@10, binary nDCG@10. Gold-free queries excluded from retrieval metric denominators.",
    }


def load_questions(directory, split):
    raw = json.loads((Path(directory) / "questions.json").read_text())
    if raw and "question_id" in raw[0]:
        return [
            {
                "id": item["question_id"],
                "user": "public-benchmark",
                "question": item["question"],
                "expected": "answered",
                # Upstream gold is free-form English prose. Normalised substring
                # matching was built for short Chinese values and scores it near
                # zero regardless of correctness, so literal matching is switched
                # off here and the reference answer is kept for a future judge.
                "facts": [],
                "reference_answer": item.get("answer_facts", []),
                "literal_matching": "not applicable: prose gold, needs a calibrated judge",
                "source_ids": item["expected_doc_ids"],
                "kind": item["question_type"],
                "split": split,
                "review": {"status": "upstream_gold", "source": "EnterpriseRAG-Bench v1.0.0"},
            }
            for item in raw
        ]
    return [q for q in raw if q["split"] == split]


def classify(question, retrieval, metrics):
    expected_sources = question.get("source_ids") or []
    if metrics is None:
        if not expected_sources:
            return "no_retrieval_gold"
        return "retrieval_ok" if retrieval["recall_at_10"] == 1 else "retrieval_miss"
    if not expected_sources:
        return "ok" if metrics["status_correct"] else "false_answer"
    if retrieval["recall_at_10"] != 1:
        return "retrieval_miss"
    if not metrics["status_correct"]:
        return "generation_status_miss"
    if metrics["literal_fact_total"] and metrics["literal_fact_matches"] < metrics["literal_fact_total"]:
        return "generation_fact_miss"
    return "ok"


def evaluate(question, user, manifest, search, models, chunk_map, retriever, generate, config_digest):
    allowed = allowed_sources(manifest, user)
    version_ids = [source["sha256"] for source in allowed]
    readable = {source["id"] for source in allowed}
    started = time.monotonic()
    if retriever == "dense":
        vector = models.embed([question["question"]])[0]
        hits = search.retrieve(vector, user.tenant_id, version_ids, RETRIEVAL_DEPTH)
    elif retriever in {"hybrid", "hybrid_rerank"}:
        vector = models.embed([question["question"]])[0]
        hits = search.retrieve_hybrid(
            question["question"], vector, user.tenant_id, version_ids, RETRIEVAL_DEPTH
        )
        if retriever == "hybrid_rerank":
            hits = rerank(question["question"], hits, text_of=lambda h: chunk_map[h["chunk_id"]]["text"])
    else:
        hits = search.retrieve_bm25(question["question"], user.tenant_id, version_ids, RETRIEVAL_DEPTH)
    retrieval_seconds = time.monotonic() - started
    ranked = [chunk_map[hit["chunk_id"]]["document_id"] for hit in hits]
    violations = [key for key in ranked if key not in readable]
    leaks = [key for key in ranked if key in question.get("hidden_source_ids", [])]
    row = {
        "id": question["id"],
        "config_digest": config_digest,
        "kind": question["kind"],
        "user": question["user"],
        "expected": question["expected"],
        "retriever": retriever,
        "readable_documents": len(readable),
        "retrieval": retrieval_scores(question.get("source_ids"), list(dict.fromkeys(ranked))[:10]),
        "ranked_documents": list(dict.fromkeys(ranked))[:10],
        "hits": [
            {
                "chunk_id": hit["chunk_id"],
                "document_id": chunk_map[hit["chunk_id"]]["document_id"],
                "score": hit["score"],
                "locator": chunk_map[hit["chunk_id"]]["locator"],
            }
            for hit in hits[: settings().top_k]
        ],
        "permission_violations": len(violations),
        "hidden_source_leaks": len(leaks),
        "retrieval_seconds": retrieval_seconds,
    }
    if generate:
        answer = generate_answer(question["question"], hits, chunk_map)
        row["answer"] = answer
        row["answer_metrics"] = answer_scores(question, answer, chunk_map)
    row["total_seconds"] = time.monotonic() - started
    row["failure_class"] = classify(question, row["retrieval"], row.get("answer_metrics"))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fixtures/s3")
    parser.add_argument("--split", default="development")
    parser.add_argument("--retrievers", default="bm25,dense")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default="artifacts/s3-baseline")
    parser.add_argument("--final-acceptance", action="store_true")
    args = parser.parse_args()
    if args.split == "holdout":
        # Sealed on purpose. Unsealing needs a deliberate flag and frozen targets,
        # because the held-out answers can only be looked at once.
        if not args.final_acceptance:
            parser.error(
                "Holdout is sealed. The final S7 run needs --final-acceptance, and it may only "
                "happen after artifacts/s7-frozen-targets.json exists."
            )
        targets = Path("artifacts/s7-frozen-targets.json")
        if not targets.exists():
            parser.error("Freeze the acceptance targets before looking at held-out answers")
        print(
            f"留出集最终验收：目标已于 {json.loads(targets.read_text())['frozen_at']} 冻结", flush=True
        )
    if not set(args.retrievers.split(",")) <= {"bm25", "dense", "hybrid", "hybrid_rerank"}:
        parser.error("Unknown retriever")

    retrievers = args.retrievers.split(",")
    manifest = read_manifest(args.dataset)
    questions = load_questions(args.dataset, args.split)[: args.limit]
    assert questions, "No questions selected"
    snapshot, search, chunks, _ = prepare_index(args.dataset)
    chunk_map = {chunk["id"]: chunk for chunk in chunks}
    users = identities()
    models = Models()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = run_config(args.dataset, args.split, retrievers, args.generate, snapshot)
    config["question_ids"] = [q["id"] for q in questions]
    config_path = out / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, (
            "Configuration changed: use a new output directory"
        )
    else:
        config_path.write_text(canonical(config))
    config_digest = digest(config)
    frozen = Path(args.dataset) / "freeze.json"
    if frozen.exists():
        record = json.loads(frozen.read_text())
        for filename, expected in record["files"].items():
            assert hashlib.sha256(Path(args.dataset, filename).read_bytes()).hexdigest() == expected, (
                filename
            )
        assert record["models"] == config["models"], "Frozen model digest changed"
    summary = {
        "config": config,
        "chunks": len(chunks),
        "documents": len(manifest),
        "results": {},
        "state": "running",
    }
    summary_path = Path(f"{args.out}-summary.json")
    for retriever in retrievers:
        path = out / f"{args.split}-{retriever}{'-generated' if args.generate else '-retrieval'}.jsonl"
        rows = []
        if path.exists():
            for line in path.read_text().splitlines():
                rows.append(json.loads(line))
        stale = [row["id"] for row in rows if row.get("config_digest") != config_digest]
        assert not stale, (
            f"{len(stale)} checkpoint rows in {path} came from a different configuration "
            f"(first: {stale[0]}); resume only within one snapshot, otherwise use a new output directory"
        )
        completed = {row["id"] for row in rows}
        assert len(completed) == len(rows), "Duplicate checkpoint rows"
        with path.open("a") as handle:
            for number, question in enumerate(questions, 1):
                if question["id"] in completed:
                    continue
                try:
                    try:
                        row = evaluate(
                            question,
                            users[question["user"]],
                            manifest,
                            search,
                            models,
                            chunk_map,
                            retriever,
                            args.generate,
                            config_digest,
                        )
                    except DependencyError as outage:
                        print(f"{retriever} {question['id']}: {outage}", flush=True)
                        if not await_dependencies(search, models):
                            raise
                        row = evaluate(
                            question,
                            users[question["user"]],
                            manifest,
                            search,
                            models,
                            chunk_map,
                            retriever,
                            args.generate,
                            config_digest,
                        )
                except Exception as exc:
                    summary["state"] = "interrupted"
                    summary["error"] = {
                        "id": question["id"],
                        "retriever": retriever,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    summary_path.write_text(canonical(summary))
                    raise
                rows.append(row)
                handle.write(canonical(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                totals = aggregate(rows) if args.generate else retrieval_only(rows)
                totals["failure_classes"] = tally(rows, "failure_class")
                totals["by_kind"] = {
                    kind: (aggregate if args.generate else retrieval_only)(
                        [r for r in rows if r["kind"] == kind]
                    )
                    for kind in sorted({r["kind"] for r in rows})
                }
                totals["per_question"] = str(path)
                summary["results"][retriever] = totals
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
                if number % 10 == 0 or args.generate:
                    print(
                        f"{retriever}: {number}/{len(questions)} ({row['total_seconds']:.1f}s)",
                        flush=True,
                    )
        totals = aggregate(rows) if args.generate else retrieval_only(rows)
        totals["failure_classes"] = tally(rows, "failure_class")
        totals["by_kind"] = {
            kind: (aggregate if args.generate else retrieval_only)(
                [r for r in rows if r["kind"] == kind]
            )
            for kind in sorted({r["kind"] for r in rows})
        }
        totals["per_question"] = str(path)
        summary["results"][retriever] = totals
    summary["state"] = "complete"
    summary.pop("error", None)
    summary["digest"] = digest(summary["results"])
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: brief(v) for k, v in summary["results"].items()}, ensure_ascii=False, indent=2))


def tally(rows, key):
    counts = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    return dict(sorted(counts.items()))


def retrieval_only(rows):
    result = {"queries": len(rows)}
    for key in ["recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]:
        values = [r["retrieval"][key] for r in rows if r["retrieval"][key] is not None]
        result[key] = {"mean": sum(values) / len(values) if values else None, "n": len(values)}
    result["permission_violations"] = sum(r["permission_violations"] for r in rows)
    result["hidden_source_leaks"] = sum(r["hidden_source_leaks"] for r in rows)
    times = sorted(r["retrieval_seconds"] for r in rows)
    result["retrieval_seconds_p95"] = times[max(0, round(len(times) * 0.95) - 1)] if times else None
    return result


def brief(totals):
    return {
        key: round(totals[key]["mean"], 4)
        for key in ["recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]
        if totals.get(key, {}).get("mean") is not None
    } | {"failure_classes": totals["failure_classes"]}


if __name__ == "__main__":
    main()
