"""Chunk-level retrieval evaluation on the used MultiHop-RAG batches.

Document-level recall says the right article reached the model; it cannot say whether
the right *paragraph* did. MultiHop-RAG records, for every gold document, the sentence
the answer depends on (`evidence_list[].fact`), and every one of those sentences occurs
verbatim in the article body. So this counts a gold fact as delivered when most of its
word 4-grams occur in the evidence chunks admitted from that same document.

Retrieval only — nothing is generated — so a run takes minutes and is independent of
the chat model. Batches 1 and 2 are development data. Batch 3 is supported as a
one-time frozen evaluation, and must not be used to adjust retrieval settings.

The index is whatever is live when the script runs, so each run is labelled with the
index state it measured (`--label`), and source routing is switched per run.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

from app.clients import Models, Search
from app.config import settings
from app.db import SessionLocal
from app.models import User
from app.retrieval import retrieve_authorized
from app.task_analysis import multi_source_intent

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".runtime/multihop"
SUBSETS = {
    "b1": ROOT / "fixtures/multihop/subset.json",
    "b2": ROOT / "fixtures/multihop/subset-r2.json",
    "b3": ROOT / "fixtures/multihop/subset-r3.json",
}
OUT = ROOT / "artifacts/multihop-retrieval-v2"
USER = "mh-eval"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def document_id(title: str) -> str:
    return "mh-" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:24]


def grams(text: str, n: int = 4) -> set:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def fact_delivered(fact: str, chunks: list[str], threshold: float = 0.5) -> bool:
    wanted = grams(fact)
    if not wanted:
        return False
    present = set().union(*(grams(text) for text in chunks)) if chunks else set()
    return len(wanted & present) / len(wanted) >= threshold


def evaluate(db, user, question, cfg, models, search):
    top_k = 8 if multi_source_intent(question["query"]) else 6
    started = time.monotonic()
    found = retrieve_authorized(
        db, user, question["query"], cfg=cfg, models=models, search=search, top_k=top_k
    )
    latency = (time.monotonic() - started) * 1000
    by_document = {}
    for item in found.evidence:
        by_document.setdefault(item["document_id"], []).append(item["text"])
    gold_documents = sorted({document_id(row["title"]) for row in question["evidence_list"]})
    facts = [
        fact_delivered(row["fact"], by_document.get(document_id(row["title"]), []))
        for row in question["evidence_list"]
    ]
    return {
        "chunks": len(found.evidence),
        "documents": len(by_document),
        "gold_documents": len(gold_documents),
        "gold_documents_found": sum(one in by_document for one in gold_documents),
        "gold_facts": len(facts),
        "gold_facts_delivered": sum(facts),
        "routed_groups": len(found.source_groups or []),
        "retrieval_ms": round(latency, 1),
    }


def summarize(rows):
    rows = [row for row in rows if row["gold_facts"]]
    if not rows:
        return {"questions": 0}
    return {
        "questions": len(rows),
        "gold_document_recall": round(
            sum(r["gold_documents_found"] for r in rows) / sum(r["gold_documents"] for r in rows), 3
        ),
        "all_gold_documents": sum(r["gold_documents_found"] == r["gold_documents"] for r in rows),
        "gold_fact_recall": round(
            sum(r["gold_facts_delivered"] for r in rows) / sum(r["gold_facts"] for r in rows), 3
        ),
        "all_gold_facts": sum(r["gold_facts_delivered"] == r["gold_facts"] for r in rows),
        "routed_questions": sum(r["routed_groups"] > 0 for r in rows),
        "mean_chunks": round(statistics.mean(r["chunks"] for r in rows), 2),
        "median_retrieval_ms": round(statistics.median(r["retrieval_ms"] for r in rows), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="index state, e.g. index-v1")
    parser.add_argument("--routing", choices=["on", "off"], required=True)
    parser.add_argument("--clause", choices=["on", "off"], default="off")
    parser.add_argument("--document-quota", type=int, default=2)
    parser.add_argument("--rerank", choices=["on", "off"], default="off")
    parser.add_argument("--expand", type=int, default=0)
    parser.add_argument("--batches", default="b1,b2")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-type", type=int, help="first N of each type per batch (hash order)")
    args = parser.parse_args()
    cfg = settings().model_copy(
        update={"source_routing": args.routing == "on", "source_clause_queries": args.clause == "on",
                "document_quota": args.document_quota,
                "passage_rerank": args.rerank == "on",
                "passage_expand_documents": args.expand}
    )
    queries = {digest(row["query"]): row for row in json.loads((CACHE / "MultiHopRAG.json").read_bytes())}
    models, search = Models(), Search()
    rows = []
    with SessionLocal() as db:
        user = db.get(User, USER)
        if user is None:
            raise SystemExit("缺少 mh-eval 账号，先运行 scripts/ingest_multihop.py")
        for batch in args.batches.split(","):
            subset = json.loads(SUBSETS[batch].read_text())
            chosen, taken = [], {}
            for item in subset["items"][: args.limit]:
                kind = item["question_type"]
                if args.per_type and taken.get(kind, 0) >= args.per_type:
                    continue
                taken[kind] = taken.get(kind, 0) + 1
                chosen.append(item)
            for item in chosen:
                question = queries[item["query_sha256"]]
                if not question["evidence_list"]:
                    continue
                rows.append(
                    {
                        "id": item["id"],
                        "batch": batch,
                        "question_type": item["question_type"],
                        **evaluate(db, user, question, cfg, models, search),
                    }
                )
                if len(rows) % 50 == 0:
                    print(f"  {len(rows)} questions", flush=True)
    types = sorted({row["question_type"] for row in rows})
    output = {
        "label": args.label,
        "routing": args.routing,
        "clause_queries": args.clause,
        "document_quota": args.document_quota,
        "passage_rerank": args.rerank,
        "batches": args.batches.split(","),
        "metric": (
            "gold fact delivered = at least half of the fact's word 4-grams occur in the "
            "evidence chunks admitted from that fact's own document"
        ),
        "all": summarize(rows),
        "by_type": {kind: summarize([r for r in rows if r["question_type"] == kind]) for kind in types},
        "results": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = (
        ("-clause" if args.clause == "on" else "")
        + f"-q{args.document_quota}"
        + ("-rerank" if args.rerank == "on" else "")
        + (f"-expand{args.expand}" if args.expand else "")
        + (f"-sample{args.per_type}" if args.per_type else "")
    )
    out = OUT / f"{args.label}-routing-{args.routing}{suffix}.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("label", "routing", "all", "by_type")}, ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    main()
