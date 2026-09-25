"""Build a private, reproducible queue for human review of failed answers.

The external dataset's copyrighted question/answer text remains under .runtime and
is not written into committed benchmark artifacts. Only already-used B1/B2 cases
are read; the third frozen batch is never inspected here.
"""

import hashlib
import json
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.models import AgentTask, Answer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".runtime/multihop/manual-review-queue.json"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    stages = json.loads((ROOT / "artifacts/multihop-passage-attribution.json").read_text())["rows"]
    dataset = {digest(row["query"]): row for row in json.loads((ROOT / ".runtime/multihop/MultiHopRAG.json").read_bytes())}
    lookup = {}
    for batch, name in enumerate(("subset.json", "subset-r2.json"), 1):
        subset = json.loads((ROOT / "fixtures/multihop" / name).read_text())
        lookup.update({f"B{batch}-{row['id']}": dataset[row["query_sha256"]] for row in subset["items"]})
    with SessionLocal() as db:
        answers = {digest(row.question): row.payload for row in db.scalars(
            select(Answer).where(Answer.user_id == "mh-eval").order_by(Answer.created_at)
        )}
        tasks = {digest(row.goal): row.result for row in db.scalars(
            select(AgentTask).where(AgentTask.user_id == "mh-eval", AgentTask.mode == "workflow")
            .order_by(AgentTask.created_at)
        )}
    rows = []
    for stage in stages:
        if stage["stage_proxy"] != "gold_facts_present_generation_or_scoring_review":
            continue
        item = lookup[stage["item_id"]]
        payload = (answers if stage["arm"] == "rag" else tasks).get(digest(item["query"])) or {}
        rows.append({
            "item_id": stage["item_id"], "arm": stage["arm"],
            "question_type": item["question_type"],
            "question": item["query"], "gold_answer": item["answer"],
            "status": payload.get("status"), "claims": [row.get("text") for row in payload.get("claims", [])],
            "verdict": payload.get("verdict"),
            "review_label": None,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(rows)} cases written to {OUT}")


if __name__ == "__main__":
    main()
