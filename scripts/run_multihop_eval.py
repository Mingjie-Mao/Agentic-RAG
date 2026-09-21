"""One-shot external validation on a frozen MultiHop-RAG subset.

This is the only external evaluation in the project: 150 questions, fixed before the
first run, scored by rules written before the first run, reported separately from the
self-built Hard 30 and never used to tune a prompt, a rule or a threshold.

What it can show: whether the retrieval and grounded-generation path works on material
this project did not write, in a language it was not tuned for, with 609 distractor
documents. What it cannot show: semantic answer quality. The answer metric here is
literal, exactly like the internal one — a short gold answer has to appear in the
claims. Yes/No questions are reported separately because a single word is the weakest
possible match, and Chinese equivalents are accepted because the prompts are Chinese.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

from agent.controller import create_task, run_task, task_payload
from app.db import SessionLocal
from app.models import User
from app.qa import answer_question

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".runtime/multihop"
SUBSET = ROOT / "fixtures/multihop/subset.json"
TENANT_USER = "mh-eval"
YES = ("yes", "是的", "是", "确实", "存在")
NO = ("no", "不是", "否", "没有", "不存在")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def document_id(title: str) -> str:
    return "mh-" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:24]


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip(" .。,，;；:：!！?？\"'“”")


def token_present(text: str, word: str) -> bool:
    if re.fullmatch(r"[a-z]+", word):
        return re.search(rf"(?<![a-z]){word}(?![a-z])", text) is not None
    return word in text


def answer_matches(question_type, gold, status, claims_text):
    """Literal answer scoring, fixed before the run.

    A null query is answered correctly by refusing. A Yes/No gold has to appear as a
    standalone word, and the opposite word must not appear — a claim that says both is
    not an answer. Everything else is containment of the gold string.
    """
    if question_type == "null_query":
        return status == "insufficient_evidence"
    if status not in {"answered", "conflict"} or not claims_text:
        return False
    gold_normalized = normalized(gold)
    if gold_normalized in {"yes", "no"}:
        wanted, other = (YES, NO) if gold_normalized == "yes" else (NO, YES)
        return any(token_present(claims_text, word) for word in wanted) and not any(
            token_present(claims_text, word) for word in other
        )
    return gold_normalized in claims_text


def run_rag(db, user, query):
    started = time.monotonic()
    payload = answer_question(db, user, query)
    retrieved = [
        row["document_id"] for row in payload["trace"]["candidates"] if row.get("admitted")
    ]
    candidates = [row["document_id"] for row in payload["trace"]["candidates"]]
    return payload, retrieved, candidates, payload.get("usage", {}), (time.monotonic() - started) * 1000


def run_workflow(db, user, query):
    started = time.monotonic()
    row = create_task(db, user, query, "workflow", 6, {})
    run_task(db, user, row)
    observable = task_payload(db, user, row)
    payload = observable["result"] or {"status": "execution_failed", "claims": [], "citations": []}
    retrieved = []
    for event in observable["events"]:
        retrieved.extend(event.get("evidence_refs", []))
    # The subgoal and recovery machinery keys off Chinese clause markers, so on an
    # English question it may never fire. Record how often it does instead of assuming.
    fired = [
        event["event_type"]
        for event in observable["events"]
        if event["event_type"] in {"query_recovery", "coverage_check", "coverage_repair"}
    ]
    return payload, fired, retrieved, payload.get("usage", {}), (time.monotonic() - started) * 1000


def documents_for_chunks(db, chunk_ids):
    from sqlalchemy import select

    from app.models import Chunk, Document, DocumentVersion

    if not chunk_ids:
        return []
    rows = db.execute(
        select(Chunk.id, Document.id)
        .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(Chunk.id.in_(list(dict.fromkeys(chunk_ids))))
    ).all()
    return list(dict.fromkeys(document for _, document in rows))


def score(item, question, payload, retrieved_documents, latency_ms, usage, fired=()):
    gold_documents = sorted({document_id(row["title"]) for row in question["evidence_list"]})
    cited = list(
        dict.fromkeys(row.get("document_id") for row in payload.get("citations", []) if row.get("document_id"))
    )
    claims_text = normalized(" ".join(row.get("text", "") for row in payload.get("claims", [])))
    found = [document for document in gold_documents if document in set(retrieved_documents)]
    return {
        "id": item["id"],
        "question_type": item["question_type"],
        "status": payload.get("status"),
        "answer_correct": answer_matches(
            item["question_type"], question["answer"], payload.get("status"), claims_text
        ),
        "gold_documents": len(gold_documents),
        "gold_documents_retrieved": len(found),
        "all_gold_retrieved": bool(gold_documents) and len(found) == len(gold_documents),
        "gold_documents_cited": sum(document in set(cited) for document in gold_documents),
        "cited_documents": len(cited),
        "claim_count": len(payload.get("claims", [])),
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": usage.get("total_prompt_tokens", usage.get("prompt_tokens")) or 0,
        "planner_events": list(fired),
    }


def summarize(rows):
    graded = [row for row in rows if row["question_type"] != "null_query"]
    def ratio(numerator, denominator):
        return round(numerator / denominator, 3) if denominator else None
    return {
        "questions": len(rows),
        "answer_correct": sum(row["answer_correct"] for row in rows),
        "answer_correct_rate": ratio(sum(row["answer_correct"] for row in rows), len(rows)),
        "all_gold_retrieved": sum(row["all_gold_retrieved"] for row in graded),
        "all_gold_retrieved_rate": ratio(sum(row["all_gold_retrieved"] for row in graded), len(graded)),
        "gold_document_recall": ratio(
            sum(row["gold_documents_retrieved"] for row in graded),
            sum(row["gold_documents"] for row in graded),
        ),
        "gold_document_citation_recall": ratio(
            sum(row["gold_documents_cited"] for row in graded),
            sum(row["gold_documents"] for row in graded),
        ),
        "refused": sum(row["status"] == "insufficient_evidence" for row in rows),
        "coverage_repairs": sum("coverage_repair" in row.get("planner_events", []) for row in rows),
        "query_recoveries": sum("query_recovery" in row.get("planner_events", []) for row in rows),
        "p50_latency_ms": round(statistics.median(row["latency_ms"] for row in rows), 1) if rows else None,
        "mean_prompt_tokens": round(statistics.mean(row["prompt_tokens"] for row in rows), 1) if rows else None,
    }


def summarize_arm(rows):
    types = sorted({row["question_type"] for row in rows})
    return {
        "all": summarize(rows),
        "by_type": {
            question_type: summarize([row for row in rows if row["question_type"] == question_type])
            for question_type in types
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=["rag", "workflow"], default=["rag", "workflow"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default="artifacts/multihop-external.json")
    parser.add_argument(
        "--smoke",
        type=int,
        help="检查管线用：只跑不在冻结子集里的题，绝不动官方子集",
    )
    args = parser.parse_args()
    subset = json.loads(SUBSET.read_text())
    raw = (CACHE / "MultiHopRAG.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != subset["source"]["files"]["MultiHopRAG.json"]:
        raise SystemExit("本地数据集与冻结时的哈希不一致")
    by_query = {digest(row["query"]): row for row in json.loads(raw)}
    if args.smoke:
        frozen = {item["query_sha256"] for item in subset["items"]}
        pool = [row for key, row in sorted(by_query.items()) if key not in frozen]
        items = [
            {
                "id": f"SMOKE-{digest(row['query'])[:12]}",
                "question_type": row["question_type"],
                "query_sha256": digest(row["query"]),
                "answer_sha256": digest(row["answer"]),
            }
            for row in pool[: args.smoke]
        ]
    else:
        items = subset["items"][: args.limit]
    results = {arm: [] for arm in args.arms}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    def payload():
        return {
            "benchmark": subset["name"],
            "subset_sha256": hashlib.sha256(SUBSET.read_bytes()).hexdigest(),
            "dataset_files": subset["source"]["files"],
            "mode": "smoke" if args.smoke else "frozen_subset",
            "arms": args.arms,
            "questions": len(items),
            "results": results,
            "summary": {arm: summarize_arm(rows) for arm, rows in results.items() if rows},
            "scoring": (
                "null_query 正确 = 拒答；Yes/No 正确 = 出现对应词且不出现相反词（接受中文写法）；"
                "其余正确 = 金标答案串出现在 claims 中。检索口径为进入生成的证据所属文档。"
            ),
            "note": (
                "外部一次性验证，与自建 Hard 30 分开报告；未用于调 prompt、规则或阈值。"
                "答案指标是字面匹配，不是语义正确率。"
            ),
        }

    with SessionLocal() as db:
        user = db.get(User, TENANT_USER)
        if user is None:
            raise SystemExit("缺少 mh-eval 账号，先运行 scripts/ingest_multihop.py")
        for index, item in enumerate(items, 1):
            question = by_query.get(item["query_sha256"])
            if question is None or digest(question["answer"]) != item["answer_sha256"]:
                raise SystemExit(f"{item['id']}: 题面或金标与冻结哈希不符")
            for arm in args.arms:
                print(f"[{index}/{len(items)}] {item['id']} {arm} ...", flush=True)
                if arm == "rag":
                    result, retrieved, _, usage, latency = run_rag(db, user, question["query"])
                    fired = []
                else:
                    result, fired, refs, usage, latency = run_workflow(db, user, question["query"])
                    retrieved = documents_for_chunks(db, refs)
                results[arm].append(
                    score(item, question, result, retrieved, latency, usage, fired)
                )
            out.write_text(json.dumps(payload(), ensure_ascii=False, indent=2) + "\n")
    final = payload()
    out.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(final["summary"], ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    main()
