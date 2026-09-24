"""Read back what the external runs actually produced.

The two arms keep their results in different places — the single-turn path writes an
`Answer` row, the agent keeps the result on its task — and every diagnosis needs both,
joined to the frozen subsets and to the dataset the questions came from. Keeping that
join in one place means a diagnosis cannot quietly read one arm's output while
labelling it as the other's, which has already happened once.
"""

import hashlib
import json
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.models import AgentTask, Answer

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".runtime/multihop"
SUBSET_DIR = ROOT / "fixtures/multihop"
EVAL_USER = "mh-eval"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def document_id(title: str) -> str:
    return "mh-" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:24]


def dataset():
    """Every dataset question, keyed by the sha256 of its text."""
    return {
        digest(row["query"]): row
        for row in json.loads((CACHE / "MultiHopRAG.json").read_bytes())
    }


def subset(name: str) -> dict:
    return json.loads((SUBSET_DIR / name).read_text())


def batch_number(subset_name: str) -> int:
    stem = subset_name.replace(".json", "")
    return 1 if stem == "subset" else int(stem.rsplit("-r", 1)[-1])


def payloads():
    """{arm: {question_hash: payload}} for the evaluation account.

    Later rows win: a question rerun by `--resume` is represented by the run that
    actually produced the artifact record.
    """
    with SessionLocal() as db:
        answers = db.scalars(
            select(Answer).where(Answer.user_id == EVAL_USER).order_by(Answer.created_at)
        ).all()
        tasks = db.scalars(
            select(AgentTask)
            .where(AgentTask.user_id == EVAL_USER, AgentTask.mode == "workflow")
            .order_by(AgentTask.created_at)
        ).all()
    return {
        "rag": {digest(row.question): row.payload for row in answers},
        "workflow": {digest(row.goal): row.result for row in tasks if row.result},
    }


def gold_documents(question: dict) -> list[str]:
    return sorted({document_id(row["title"]) for row in question.get("evidence_list", [])})


def evidence_text_by_chunk(chunk_ids):
    """Chunk text for the evidence a claim cites, read back from the database."""
    from app.models import Chunk

    ids = list(dict.fromkeys(chunk_ids))
    if not ids:
        return {}
    with SessionLocal() as db:
        rows = db.scalars(select(Chunk).where(Chunk.id.in_(ids))).all()
    return {row.id: row.text for row in rows}
