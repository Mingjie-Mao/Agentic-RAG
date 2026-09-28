"""Does sentence-level selection deliver the gold fact sentences? Retrieval only.

Same stratified sample as the generation experiments (first N answerable questions of
each type per used batch; batch 3 is never read). Each question's candidate pool is
built once — the reranked, source-routed retrieval the chunk baseline already uses —
and every selection configuration is scored on that same pool, so the comparison is
paired and differs only in what is kept.

A gold fact counts as delivered when at least half of its word 4-grams occur in the
evidence taken from its own document, the same proxy the chunk-level evaluation uses.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

from app.clients import Models, Search
from app.config import settings
from app.db import SessionLocal
from app.evidence_selection import build_lanes, cross_encoder_scorer, select_sentences
from app.models import User
from app.task_analysis import multi_source_intent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multihop_retrieval_eval import document_id, fact_delivered  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".runtime/multihop"
SUBSETS = {"b1": ROOT / "fixtures/multihop/subset.json", "b2": ROOT / "fixtures/multihop/subset-r2.json"}
OUT = ROOT / "artifacts/sentence-selection-eval.json"
CONFIGS = {
    "sentences-q2": {"clause": False, "per_lane": 2, "global_extra": 2, "neighbors": 0},
    "sentences-clause2": {"clause": True, "per_lane": 2, "global_extra": 2, "neighbors": 0},
    "sentences-q3": {"clause": False, "per_lane": 3, "global_extra": 2, "neighbors": 0},
    "sentences-q2-nb1": {"clause": False, "per_lane": 2, "global_extra": 2, "neighbors": 1},
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sample(per_type):
    items = []
    for batch, path in SUBSETS.items():
        taken = {}
        for item in json.loads(path.read_text())["items"]:
            kind = item["question_type"]
            if kind != "null_query" and taken.get(kind, 0) < per_type:
                taken[kind] = taken.get(kind, 0) + 1
                items.append({**item, "batch": batch})
    return items


def measure(record, evidence):
    by_document = {}
    for item in evidence:
        by_document.setdefault(item["document_id"], []).append(item["text"])
    delivered = [
        fact_delivered(row["fact"], by_document.get(document_id(row["title"]), []))
        for row in record["evidence_list"]
    ]
    words = sum(len(item["text"].split()) for item in evidence)
    return {"facts": len(delivered), "delivered": sum(delivered), "items": len(evidence), "words": words}


def summarize(rows, key):
    facts = sum(r[key]["facts"] for r in rows)
    return {
        "questions": len(rows),
        "gold_fact_recall": round(sum(r[key]["delivered"] for r in rows) / facts, 3),
        "all_gold_facts": sum(r[key]["delivered"] == r[key]["facts"] for r in rows),
        "mean_words": round(statistics.mean(r[key]["words"] for r in rows), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=10)
    args = parser.parse_args()
    records = {digest(r["query"]): r for r in json.loads((CACHE / "MultiHopRAG.json").read_bytes())}
    raw = cross_encoder_scorer()
    memo = {}

    def score(query, texts):
        missing = [t for t in texts if (query, t) not in memo]
        for text, value in zip(missing, raw(query, missing) if missing else [], strict=True):
            memo[(query, text)] = value
        return [memo[(query, t)] for t in texts]

    cfg, models, search = settings(), Models(), Search()
    rows = []
    with SessionLocal() as db:
        user = db.get(User, "mh-eval")
        for item in sample(args.per_type):
            record = records[item["query_sha256"]]
            question = record["query"]
            top_k = 8 if multi_source_intent(question) else 6 if len(question) <= 30 else cfg.top_k
            row = {"id": item["id"], "batch": item["batch"], "question_type": item["question_type"]}
            with_clauses, chunks = build_lanes(
                db, user, question, cfg=cfg, models=models, search=search, top_k=top_k, clause_queries=True
            )
            lanes_by_clause = {
                True: with_clauses,
                False: [{**lane, "query": question} for lane in with_clauses],
            }
            row["chunks-reranked"] = measure(record, chunks)
            for name, config in CONFIGS.items():
                evidence = select_sentences(
                    question,
                    lanes_by_clause[config["clause"]],
                    score=score,
                    per_lane=config["per_lane"],
                    global_extra=config["global_extra"],
                    neighbors=config["neighbors"],
                )
                row[name] = measure(record, evidence)
            rows.append(row)
            if len(rows) % 10 == 0:
                print(f"  {len(rows)} questions", flush=True)
    keys = ["chunks-reranked", *CONFIGS]
    output = {
        "sample": f"first {args.per_type} answerable questions per type per batch (b1, b2)",
        "metric": "gold fact delivered = >=50% of its word 4-grams in evidence from its own document",
        "configs": CONFIGS,
        "all": {key: summarize(rows, key) for key in keys},
        "by_type": {
            kind: {key: summarize([r for r in rows if r["question_type"] == kind], key) for key in keys}
            for kind in sorted({r["question_type"] for r in rows})
        },
        "results": rows,
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("all", "by_type")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
