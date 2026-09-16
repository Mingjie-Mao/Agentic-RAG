"""Assemble the A-Q1 calibration set from real runs (S-A/Q1).

Two judgements are needed and they are not the same question:

  relevance  — does this passage answer *this* question?
  entailment — is this claim supported by the passage it cites?

A passage can be about the right subject and still not answer the question; a claim
can be word-for-word present in a passage that answers a different question. Measuring
them together hides which one failed, so each case carries both fields.

Cases are drawn from configurations that actually produced them, including the BM25
run, because the pattern this set exists to catch — a claim that restates the question
and attaches a number from an unrelated document — nearly vanished once hybrid
retrieval became the default, leaving too few examples to calibrate against.
"""

import argparse
import collections
import json
from pathlib import Path

from app.evaluation import prepare_snapshot

DATASET = Path("fixtures/s3-v2")
RUNS = {
    "hybrid": "artifacts/s3-v2-generated/development-hybrid-generated.jsonl",
    "bm25": "artifacts/s3-v2-generated/development-bm25-generated.jsonl",
}
# Deliberately over-sample the failure shapes; a calibration set of mostly easy
# positives cannot tell a useful scorer from a scorer that always says "relevant".
QUOTA = {
    "答非所问": 6,
    "该答却拒答": 6,
    "引用非金标文档": 4,
    "冲突": 5,
    "不可答": 5,
    "改写型提问命中": 5,
    "正常命中": 5,
}


def stratum(question, row):
    metrics = row["answer_metrics"]
    status = row["answer"]["status"]
    gold = set(question.get("source_ids") or [])
    cited = {c["document_id"] for c in row["answer"]["citations"]}
    if question["kind"] == "permission" and status == "answered":
        return "答非所问"
    if not metrics["status_correct"] and status == "insufficient_evidence":
        return "该答却拒答"
    if cited and gold and not (cited & gold):
        return "引用非金标文档"
    if question["expected"] == "conflict" and metrics["status_correct"]:
        return "冲突"
    if question["kind"] == "unanswerable" and metrics["status_correct"]:
        return "不可答"
    if metrics["status_correct"] and question["kind"] == "lookup":
        # Paraphrased questions avoid the document's wording; they are the cases a
        # naive lexical relevance rule gets wrong in the expensive direction.
        overlap = len(
            set(question["question"]) & set(" ".join(c["text"] for c in row["answer"]["citations"]))
        )
        return "改写型提问命中" if overlap < 18 else "正常命中"
    return None


def case(question, row, configuration, name, chunk_text):
    citations = row["answer"]["citations"]
    return {
        "case_id": f"{configuration}-{row['id']}",
        "stratum": name,
        "configuration": configuration,
        "question_id": row["id"],
        "question": question["question"],
        "question_kind": question["kind"],
        "expected_status": question["expected"],
        "actual_status": row["answer"]["status"],
        "gold_source_ids": question.get("source_ids") or [],
        "hidden_source_ids": question.get("hidden_source_ids") or [],
        "claims": [c["text"] for c in row["answer"]["claims"]],
        "evidence": [
            {
                "document_id": c["document_id"],
                "chunk_id": c["chunk_id"],
                "locator": c["locator"].get("label"),
                "text": c["text"],
            }
            for c in citations
        ],
        # Refusals cite nothing, so without the retrieved candidates there would be
        # nothing to judge — and "did we refuse because the material really did not
        # answer the question" is exactly what relevance is supposed to settle.
        "retrieved": [
            {
                "document_id": hit["document_id"],
                "chunk_id": hit["chunk_id"],
                "is_gold": hit["document_id"] in (question.get("source_ids") or []),
                "text": chunk_text.get(hit["chunk_id"], ""),
            }
            for hit in row["hits"]
        ],
        # Filled in by a reviewer; never derived, or this set would only restate the
        # metrics it exists to check.
        "label": {
            "relevance": None,
            "entailment": None,
            "note": "",
            "reviewer": None,
            "review_kind": None,
        },
    }


def build(out):
    questions = {q["id"]: q for q in json.loads((DATASET / "questions.json").read_text())}
    _, chunks, _ = prepare_snapshot(str(DATASET))
    chunk_text = {c["id"]: c["text"] for c in chunks}
    pools = collections.defaultdict(list)
    for configuration, path in RUNS.items():
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            question = questions[row["id"]]
            name = stratum(question, row)
            if name:
                pools[name].append(case(question, row, configuration, name, chunk_text))

    selected, shortfall = [], {}
    for name, quota in QUOTA.items():
        pool = sorted(pools.get(name, []), key=lambda c: c["case_id"])
        # Prefer one configuration per question so the same failure is not counted twice.
        seen, unique = set(), []
        for item in pool:
            if item["question_id"] in seen:
                continue
            seen.add(item["question_id"])
            unique.append(item)
        selected.extend(unique[:quota])
        if len(unique) < quota:
            shortfall[name] = {"wanted": quota, "available": len(unique)}

    report = {
        "version": "a-q1-calibration-v1",
        "purpose": "Calibrate relevance and entailment scorers; never used as their test set",
        "cases": len(selected),
        "strata": dict(collections.Counter(c["stratum"] for c in selected)),
        "shortfall": shortfall,
        "labels_present": 0,
        "review_status": "unlabelled",
    }
    Path(out).write_text(
        json.dumps({"meta": report, "cases": selected}, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if shortfall:
        print("注意：部分分层样本不足，已如实记录，不用其他分层填充凑数。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="fixtures/calibration/a-q1.json")
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    build(args.out)
