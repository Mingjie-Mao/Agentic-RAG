"""Locate gold facts in the *historical evidence actually sent to generation*.

This reads only batches 1 and 2, which are already development data. It does not run
retrieval against the current index: reprocessing an article must not rewrite what a
past answer saw. Gold-fact coverage is a word-4-gram proxy, not semantic entailment.
"""

import hashlib
import json
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.models import AgentTask, Answer, Chunk, Document, DocumentVersion
from multihop_retrieval_eval import fact_delivered


ROOT = Path(__file__).resolve().parents[1]
FAILURES = ROOT / "artifacts/multihop-failure-attribution.json"
SUBSETS = (ROOT / "fixtures/multihop/subset.json", ROOT / "fixtures/multihop/subset-r2.json")
DATASET = ROOT / ".runtime/multihop/MultiHopRAG.json"
OUT = ROOT / "artifacts/multihop-passage-attribution.json"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def document_id(title):
    return "mh-" + digest(title)[:24]


def actual_refs(arm, record):
    if record is None:
        return []
    if arm == "rag":
        return [
            row["chunk_id"]
            for row in (record.payload or {}).get("trace", {}).get("candidates", [])
            if row.get("admitted")
        ]
    return record.evidence_chunk_ids or []


def classify(facts, by_document, missing_documents):
    delivered = [
        fact_delivered(row["fact"], by_document.get(document_id(row["title"]), []))
        for row in facts
    ]
    if missing_documents:
        stage = "gold_document_missing"
    elif not all(delivered):
        stage = "gold_fact_proxy_missing"
    else:
        stage = "gold_facts_present_generation_or_scoring_review"
    return stage, delivered


def main():
    failures = json.loads(FAILURES.read_text())
    dataset = {digest(row["query"]): row for row in json.loads(DATASET.read_bytes())}
    lookup = {}
    for batch, path in enumerate(SUBSETS, 1):
        for row in json.loads(path.read_text())["items"]:
            lookup[f"B{batch}-{row['id']}"] = dataset[row["query_sha256"]]

    with SessionLocal() as db:
        answers = db.scalars(
            select(Answer).where(Answer.user_id == "mh-eval").order_by(Answer.created_at)
        ).all()
        tasks = db.scalars(
            select(AgentTask)
            .where(AgentTask.user_id == "mh-eval", AgentTask.mode == "workflow")
            .order_by(AgentTask.created_at)
        ).all()
        records = {
            "rag": {digest(row.question): row for row in answers},
            "workflow": {digest(row.goal): row for row in tasks},
        }
        refs = {
            (row["item_id"], row["arm"]): actual_refs(
                row["arm"], records[row["arm"]].get(digest(lookup[row["item_id"]]["query"]))
            )
            for row in failures["failures"]
        }
        all_ids = sorted({key for values in refs.values() for key in values})
        chunks = {}
        for offset in range(0, len(all_ids), 500):
            for key, body, doc_id in db.execute(
                select(Chunk.id, Chunk.text, Document.id)
                .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(Chunk.id.in_(all_ids[offset : offset + 500]))
            ):
                chunks[key] = (doc_id, body)

    rows = []
    for failure in failures["failures"]:
        key = (failure["item_id"], failure["arm"])
        by_document = {}
        for chunk_id in refs[key]:
            if chunk_id in chunks:
                doc_id, text = chunks[chunk_id]
                by_document.setdefault(doc_id, []).append(text)
        facts = lookup[failure["item_id"]]["evidence_list"]
        missing_document = failure["gold_documents_retrieved"] < failure["gold_documents"]
        stage, delivered = classify(facts, by_document, missing_document)
        rows.append({
            "item_id": failure["item_id"],
            "arm": failure["arm"],
            "stage_proxy": stage,
            "gold_facts": len(facts),
            "gold_facts_delivered_proxy": sum(delivered),
            "evidence_chunks": len(refs[key]),
            "missing_historical_chunks": sum(chunk_id not in chunks for chunk_id in refs[key]),
        })
    stages = sorted({row["stage_proxy"] for row in rows})
    report = {
        "scope": "B1/B2 failed arm runs; historical evidence, not current retrieval",
        "metric": "gold fact delivered when >=50% of its word 4-grams occur in admitted chunks from its own gold document",
        "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (FAILURES, *SUBSETS, DATASET)},
        "summary": {
            "failures": len(rows),
            "by_stage_proxy": {stage: sum(row["stage_proxy"] == stage for row in rows) for stage in stages},
            "historical_chunks_missing": sum(row["missing_historical_chunks"] for row in rows),
        },
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
