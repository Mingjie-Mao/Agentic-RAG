"""Attribute the Yes/No errors before touching anything that produced them.

Batch 2 answered "yes" to 11 of the 30 questions whose gold is "no", and the verdict
accuracy over all Yes/No questions (53.2%) is below the "always answer yes" baseline
(61.0%). That is a finding, not yet a cause. The cause could be any of:

- the second document never reached the evidence (retrieval);
- the evidence was complete but the comparison was read the wrong way (reasoning);
- the claims are vague enough that no verdict follows from them (output);
- the claims are clear and the verdict classifier flipped them (classifier);
- the question or its gold is arguable (benchmark).

Only the first one can be decided mechanically, and this script decides exactly that
much: whether every gold document reached the evidence that generation saw. The rest
is written out with the question, the claims and the cited evidence so a person can
label it, because guessing between "reasoning" and "ambiguity" from a script is how a
reasoning failure gets invented. Nothing here changes a prompt, a rule or a score.
"""

import argparse
import csv
import json
from pathlib import Path

import multihop_outputs as outputs

ROOT = Path(__file__).resolve().parents[1]
HUMAN_BUCKETS = (
    "comparison_reasoning_wrong",
    "claims_too_vague_for_a_verdict",
    "verdict_classifier_flipped_clear_claims",
    "benchmark_gold_or_question_arguable",
    "other",
)


def build(external_path: Path, subset_name: str):
    external = json.loads(external_path.read_text())
    items = {item["id"]: item for item in outputs.subset(subset_name)["items"]}
    questions = outputs.dataset()
    stored = outputs.payloads()
    rows, counts, agreement = [], {}, {}
    for arm, records in external["results"].items():
        for record in records:
            item = items[record["id"]]
            question = questions[item["query_sha256"]]
            gold = (question["answer"] or "").strip().lower()
            if gold not in {"yes", "no"}:
                continue
            verdict = record.get("verdict")
            outcome = (
                "refused_or_no_verdict"
                if verdict not in {"yes", "no"}
                else "correct"
                if verdict == gold
                else f"{gold}_answered_{verdict}"
            )
            complete = record["gold_documents_retrieved"] == record["gold_documents"]
            mechanical = "evidence_complete" if complete else "retrieval_incomplete"
            key = f"{arm}|{outcome}|{mechanical}"
            counts[key] = counts.get(key, 0) + 1
            if outcome == "correct":
                # A correct verdict reached without all the gold evidence is not
                # evidence of judgement: on a batch with 47 "yes" golds, answering
                # "yes" gets there too.
                agreement[f"{arm}|gold={gold}|{mechanical}"] = (
                    agreement.get(f"{arm}|gold={gold}|{mechanical}", 0) + 1
                )
            if outcome in {"correct", "refused_or_no_verdict"}:
                continue
            payload = stored[arm].get(item["query_sha256"]) or {}
            claims = payload.get("claims", [])
            evidence = outputs.evidence_text_by_chunk(
                [chunk for claim in claims for chunk in claim.get("evidence_ids", [])]
            )
            rows.append(
                {
                    "id": record["id"],
                    "arm": arm,
                    "question_type": record["question_type"],
                    "question": question["query"],
                    "gold": gold,
                    "verdict": verdict,
                    "error": outcome,
                    "gold_documents": record["gold_documents"],
                    "gold_documents_retrieved": record["gold_documents_retrieved"],
                    "mechanical_bucket": mechanical,
                    "claims": [claim["text"] for claim in claims],
                    "cited_evidence": [
                        evidence.get(chunk, "")[:600]
                        for claim in claims
                        for chunk in claim.get("evidence_ids", [])
                    ],
                    "human_bucket": "",
                    "human_note": "",
                }
            )
    return external, rows, counts, agreement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", default="artifacts/multihop-external-r2.json")
    parser.add_argument("--subset", default="subset-r2.json")
    parser.add_argument("--out", default="artifacts/multihop-yes-bias-diagnosis.json")
    parser.add_argument("--csv", default="artifacts/multihop-yes-bias-review.csv")
    args = parser.parse_args()
    external, rows, counts, agreement = build(ROOT / args.external, args.subset)
    mechanical = {
        bucket: sum(
            value for key, value in counts.items() if key.endswith(bucket) and "answered" in key
        )
        for bucket in ("retrieval_incomplete", "evidence_complete")
    }
    output = {
        "source": args.external,
        "subset": args.subset,
        "scoring_rule": external.get("scoring_rule"),
        "counts_by_arm_outcome_and_evidence": counts,
        "wrong_verdicts_by_mechanical_bucket": mechanical,
        "correct_verdicts_by_gold_and_evidence": agreement,
        "human_buckets_available": list(HUMAN_BUCKETS),
        "note": (
            "只机械判定一件事：gold 文档是否全部进入了生成看到的证据。其余分类留空，"
            "由人工填写；脚本不猜 reasoning 与 ambiguity 的区别。本文件不改变任何成绩。"
        ),
        "rows": rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    review = ROOT / args.csv
    with review.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id", "arm", "gold", "verdict", "error", "gold_docs_retrieved",
                "mechanical_bucket", "question", "claims", "cited_evidence",
                f"human_bucket({'|'.join(HUMAN_BUCKETS)})", "human_note",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["id"], row["arm"], row["gold"], row["verdict"], row["error"],
                    f"{row['gold_documents_retrieved']}/{row['gold_documents']}",
                    row["mechanical_bucket"], row["question"],
                    "\n".join(row["claims"]), "\n---\n".join(row["cited_evidence"]), "", "",
                ]
            )
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print(f"{out}\n{review}  ({len(rows)} 条待人工归类)")


if __name__ == "__main__":
    main()
