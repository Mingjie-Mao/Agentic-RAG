"""Why the Yes/No half of the external set scores near zero.

The external answer metric was fixed before the run and is not changed here. This
only splits the Yes/No questions (comparison and temporal) into three buckets, so the
report can say which part is the system failing to answer and which part is the metric
failing to see an answer:

- refused: the system said it had insufficient evidence;
- answered_without_verdict_word: it made a judgement in prose ("the two articles do
  not agree") without ever writing yes or no, so a word-level metric cannot read it;
- verdict_matches_gold / verdict_conflicts_or_wrong: it wrote a verdict word.

The third bucket is the only one the declared metric can score. Nothing here is a
corrected score: a claim in the second bucket may still be wrong, and this file does
not pretend to know.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

from sqlalchemy import select

from app.db import SessionLocal
from app.models import AgentTask, Answer

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".runtime/multihop"
SUBSET = ROOT / "fixtures/multihop/subset.json"
EXTERNAL = ROOT / "artifacts/multihop-external.json"
OUT = ROOT / "artifacts/multihop-verdict-diagnosis.json"
YES = ("yes", "是的", "确实", "存在")
NO = ("no", "不是", "否", "没有", "不存在")
BINARY = ("comparison_query", "temporal_query")
VERDICTS = {"yes", "no"}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_present(text: str, word: str) -> bool:
    if re.fullmatch(r"[a-z]+", word):
        return re.search(rf"(?<![a-z]){word}(?![a-z])", text) is not None
    return word in text


def verdict_analysis(rows, gold_by_id):
    """Accuracy of the explicit verdict against the trivial baselines.

    A binary question is guessable: reporting 58% without saying that answering "yes"
    to everything scores 61% on this batch would be reporting nothing. The confusion
    matrix is here for the same reason — it shows which way the errors go.
    """
    pairs = [
        (gold_by_id[row["id"]], row.get("verdict"))
        for row in rows
        if gold_by_id.get(row["id"]) in {"yes", "no"}
    ]
    answered = [(gold, said) for gold, said in pairs if said in {"yes", "no"}]
    golds = [gold for gold, _ in pairs]
    majority = max(golds.count("yes"), golds.count("no")) / len(golds) if golds else None
    matrix = {
        f"gold={gold}|verdict={said or 'none'}": sum(
            1 for one, two in pairs if one == gold and (two or "none") == (said or "none")
        )
        for gold in ("yes", "no")
        for said in ("yes", "no", "unclear", None)
    }
    return {
        "yes_no_questions": len(pairs),
        "gold_yes": golds.count("yes"),
        "gold_no": golds.count("no"),
        "majority_class_baseline": round(majority, 3) if majority else None,
        "verdict_given": len(answered),
        "verdict_correct": sum(gold == said for gold, said in answered),
        "accuracy_over_all_yes_no": round(
            sum(gold == said for gold, said in answered) / len(pairs), 3
        )
        if pairs
        else None,
        "accuracy_when_a_verdict_was_given": round(
            sum(gold == said for gold, said in answered) / len(answered), 3
        )
        if answered
        else None,
        "confusion": {key: value for key, value in matrix.items() if value},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default=str(SUBSET))
    parser.add_argument("--external", default=str(EXTERNAL))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    subset_path, external_path, out_path = Path(args.subset), Path(args.external), Path(args.out)
    subset = json.loads(subset_path.read_text())
    queries = json.loads((CACHE / "MultiHopRAG.json").read_bytes())
    gold_by_hash = {digest(row["query"]): row["answer"] for row in queries}
    item_by_id = {item["id"]: item for item in subset["items"]}
    external = json.loads(external_path.read_text())
    # Each arm keeps its result in its own place: the single-turn path writes an
    # Answer row, the agent keeps it on the task. Reading both from one table made the
    # two arms report identical numbers.
    with SessionLocal() as db:
        answers = db.scalars(
            select(Answer).where(Answer.user_id == "mh-eval").order_by(Answer.created_at)
        ).all()
        tasks = db.scalars(
            select(AgentTask)
            .where(AgentTask.user_id == "mh-eval", AgentTask.mode == "workflow")
            .order_by(AgentTask.created_at)
        ).all()
    payloads = {
        "rag": {digest(row.question): row.payload for row in answers},
        "workflow": {digest(row.goal): row.result for row in tasks if row.result},
    }
    report = {}
    for arm, rows in external["results"].items():
        buckets = {
            "questions": 0,
            "refused": 0,
            "answered_without_verdict_word": 0,
            "verdict_matches_gold": 0,
            "verdict_conflicts_or_wrong": 0,
            "not_recoverable": 0,
            "non_verdict_gold": 0,
        }
        for row in rows:
            if row["question_type"] not in BINARY:
                continue
            item = item_by_id[row["id"]]
            gold_answer = (gold_by_hash[item["query_sha256"]] or "").strip().lower()
            if gold_answer not in VERDICTS:
                # A few temporal questions ask for a name, not a verdict; they are
                # scored by the ordinary containment rule and do not belong here.
                buckets["non_verdict_gold"] += 1
                continue
            buckets["questions"] += 1
            payload = payloads.get(arm, {}).get(item["query_sha256"])
            if payload is None:
                buckets["not_recoverable"] += 1
                continue
            gold = gold_answer
            text = " ".join(claim.get("text", "") for claim in payload.get("claims", [])).lower()
            said_yes = any(token_present(text, word) for word in YES)
            said_no = any(token_present(text, word) for word in NO)
            if payload.get("status") == "insufficient_evidence":
                buckets["refused"] += 1
            elif not said_yes and not said_no:
                buckets["answered_without_verdict_word"] += 1
            elif (gold == "yes" and said_yes and not said_no) or (
                gold == "no" and said_no and not said_yes
            ):
                buckets["verdict_matches_gold"] += 1
            else:
                buckets["verdict_conflicts_or_wrong"] += 1
        report[arm] = buckets | {
            "verdict_field": verdict_analysis(
                [row for row in rows if row["question_type"] in BINARY],
                {
                    row["id"]: (gold_by_hash[item_by_id[row["id"]]["query_sha256"]] or "")
                    .strip()
                    .lower()
                    for row in rows
                },
            )
        }
    output = {
        "source": external_path.name,
        "subset": subset_path.name,
        "question_types": list(BINARY),
        "buckets": report,
        "note": (
            "诊断，不是重新评分：answered_without_verdict_word 只说明答案里没有判断词，"
            "不代表它语义正确。已声明的外部指标不因本文件改变。"
            "verdict_field 一节比较的是显式结论字段与两个平凡基线——随机 50% 与多数类——"
            "二元问题不给基线就等于没给结论。"
        ),
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
