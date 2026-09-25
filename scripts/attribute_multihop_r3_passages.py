"""Classify frozen B3 answer failures using passages each run actually saw.

The 4-gram fact test is a retrieval proxy, not a semantic judgment. Question,
article, and answer text remain in the local dataset and database; only counts and
opaque IDs enter the committed artifact.
"""

import hashlib
import json
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.models import AgentTask, Answer, Chunk, Document, DocumentVersion
from attribute_multihop_passages import actual_refs, classify, digest


ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "fixtures/multihop/subset-r3-eval32.json"
RUN = ROOT / "artifacts/multihop-external-r3-eval32.json"
DATASET = ROOT / ".runtime/multihop/MultiHopRAG.json"
OUT = ROOT / "artifacts/multihop-r3-passage-attribution.json"


def main():
    subset_bytes, run_bytes = SUBSET.read_bytes(), RUN.read_bytes()
    subset, run = json.loads(subset_bytes), json.loads(run_bytes)
    if run["subset_sha256"] != hashlib.sha256(subset_bytes).hexdigest() or run["scoring_rule"] != "v2":
        raise ValueError("B3 run does not match frozen subset and scorer")
    items = {row["id"]: row for row in subset["items"]}
    dataset = {digest(row["query"]): row for row in json.loads(DATASET.read_bytes())}
    if any({row["id"] for row in run["results"][arm]} != set(items) for arm in ("rag", "workflow")):
        raise ValueError("B3 arms are incomplete")
    failed = [(arm, row) for arm in ("rag", "workflow")
              for row in run["results"][arm] if not row["answer_correct"]]
    with SessionLocal() as db:
        records = {
            "rag": {digest(row.question): row for row in db.scalars(
                select(Answer).where(Answer.user_id == "mh-eval").order_by(Answer.created_at)
            )},
            "workflow": {digest(row.goal): row for row in db.scalars(
                select(AgentTask).where(AgentTask.user_id == "mh-eval", AgentTask.mode == "workflow")
                .order_by(AgentTask.created_at)
            )},
        }
        refs = {}
        for arm, row in failed:
            question = dataset[items[row["id"]]["query_sha256"]]
            refs[(arm, row["id"])] = actual_refs(arm, records[arm].get(digest(question["query"])))
        all_ids = sorted({key for values in refs.values() for key in values})
        chunks = {}
        for offset in range(0, len(all_ids), 500):
            for key, body, doc_id in db.execute(
                select(Chunk.id, Chunk.text, Document.id)
                .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(Chunk.id.in_(all_ids[offset:offset + 500]))
            ):
                chunks[key] = (doc_id, body)

    rows = []
    for arm, failure in failed:
        question = dataset[items[failure["id"]]["query_sha256"]]
        chunk_ids = refs[(arm, failure["id"])]
        by_document = {}
        for chunk_id in chunk_ids:
            if chunk_id in chunks:
                doc_id, body = chunks[chunk_id]
                by_document.setdefault(doc_id, []).append(body)
        stage, delivered = classify(
            question["evidence_list"], by_document, not failure["all_gold_retrieved"]
        )
        rows.append({
            "item_id": failure["id"], "arm": arm, "question_type": failure["question_type"],
            "stage_proxy": stage, "gold_facts": len(delivered),
            "gold_facts_delivered_proxy": sum(delivered), "evidence_chunks": len(chunk_ids),
            "missing_historical_chunks": sum(key not in chunks for key in chunk_ids),
        })
    categories = sorted({row["stage_proxy"] for row in rows})
    report = {
        "scope": "all failed arm-runs in pre-registered B3 answer32; actual historical evidence handles",
        "metric": "gold fact delivered when >=50% of its word 4-grams occur in admitted chunks from its own gold document",
        "input_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (SUBSET, RUN, DATASET)
        },
        "summary": {
            "failed_arm_runs": len(rows),
            "by_arm": {
                arm: {category: sum(row["arm"] == arm and row["stage_proxy"] == category for row in rows)
                      for category in categories}
                for arm in ("rag", "workflow")
            },
            "historical_chunks_missing": sum(row["missing_historical_chunks"] for row in rows),
        },
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
