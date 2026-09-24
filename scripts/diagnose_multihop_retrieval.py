"""Paired retrieval ablation for the multi-source evidence budget.

Retrieval can be measured without generating anything, so the budget change can be
compared on the *same* questions instead of across two different batches. This runs
each question twice against the same corpus and the same index — once with the budget
the external run used (four chunks, no per-document quota) and once with the budget a
multi-source question now gets (eight chunks, at most two per document) — and reports
how many gold documents each one reaches.

It runs on batch 1, which has already been used for a full evaluation and therefore
serves as the development set. Batch 2 is untouched by this file.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from app import retrieval as retrieval_module
from app.clients import Models, Search
from app.config import settings
from app.db import SessionLocal
from app.models import User
from app.retrieval import retrieve_authorized
from app.task_analysis import multi_source_intent

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".runtime/multihop"
SUBSET = ROOT / "fixtures/multihop/subset.json"
OUT = ROOT / "artifacts/multihop-retrieval-ablation.json"
USER = "mh-eval"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def document_id(title: str) -> str:
    return "mh-" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:24]


def retrieve(db, user, query, *, top_k, multi_source):
    original = retrieval_module.multi_source_intent
    retrieval_module.multi_source_intent = (lambda _text: multi_source)
    try:
        started = time.monotonic()
        found = retrieve_authorized(
            db,
            user,
            query,
            cfg=settings(),
            models=Models(),
            search=Search(),
            top_k=top_k,
        )
        return found, (time.monotonic() - started) * 1000
    finally:
        retrieval_module.multi_source_intent = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    subset = json.loads(SUBSET.read_text())
    queries = {digest(row["query"]): row for row in json.loads((CACHE / "MultiHopRAG.json").read_bytes())}
    rows = []
    with SessionLocal() as db:
        user = db.get(User, USER)
        if user is None:
            raise SystemExit("缺少 mh-eval 账号，先运行 scripts/ingest_multihop.py")
        for index, item in enumerate(subset["items"][: args.limit], 1):
            question = queries[item["query_sha256"]]
            gold = sorted({document_id(row["title"]) for row in question["evidence_list"]})
            if not gold:
                continue
            record = {
                "id": item["id"],
                "question_type": item["question_type"],
                "gold_documents": len(gold),
                "multi_source_detected": multi_source_intent(question["query"]),
            }
            for label, top_k, multi_source in (("before", 4, False), ("after", 8, True)):
                found, latency = retrieve(db, user, question["query"], top_k=top_k, multi_source=multi_source)
                documents = {row["document_id"] for row in found.evidence}
                record[label] = {
                    "chunks": len(found.evidence),
                    "documents": len(documents),
                    "gold_found": sum(one in documents for one in gold),
                    "all_gold": all(one in documents for one in gold),
                    "retrieval_ms": round(latency, 1),
                }
            rows.append(record)
            if index % 25 == 0:
                print(f"  {index} questions", flush=True)

    def summarize(label, subset_rows):
        gold_total = sum(row["gold_documents"] for row in subset_rows)
        return {
            "questions": len(subset_rows),
            "gold_document_recall": round(
                sum(row[label]["gold_found"] for row in subset_rows) / gold_total, 3
            )
            if gold_total
            else None,
            "all_gold_retrieved": sum(row[label]["all_gold"] for row in subset_rows),
            "mean_documents": round(
                statistics.mean(row[label]["documents"] for row in subset_rows), 2
            )
            if subset_rows
            else None,
            "median_retrieval_ms": round(
                statistics.median(row[label]["retrieval_ms"] for row in subset_rows), 1
            )
            if subset_rows
            else None,
        }

    detected = [row for row in rows if row["multi_source_detected"]]
    output = {
        "subset": SUBSET.name,
        "note": (
            "同一批题、同一语料与索引，只改检索预算与每文档配额；不生成答案，因此与模型无关。"
            "第 1 批已经用于完整评测，这里把它当开发集用；第 2 批不参与。"
        ),
        "all_questions": {label: summarize(label, rows) for label in ("before", "after")},
        "multi_source_detected_only": {
            label: summarize(label, detected) for label in ("before", "after")
        },
        "results": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "results"}, ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    main()
