"""Build the human annotation task that a semantic scorer has to be calibrated against.

§14A stalled for one reason: the three candidate scorers were compared on 34 cases
whose labels a model proposed. That is not a gold standard, so none of them could be
allowed to withhold an answer. The fix is not a better scorer — it is labels a person
actually wrote.

This produces the task, never the labels. It samples answers the system already
produced during the two external runs and writes two annotation files:

- claim level: does the cited evidence support this claim (entailment), and does this
  claim answer what was asked (relevance);
- answer level: does the answer cover everything the question required (completeness),
  and what verdict does a reader take away from it (verdict reading).

The verdict reading is what separates the three failures that currently look alike:
an answer that is right but unreadable to a word-level metric, a classifier that
misread a clear answer, and an answer that is simply wrong.

The annotator's files are blinded on purpose: no benchmark gold answer, no system
verdict, no indication of whether the current metric scored the item correct. Anchors
like those are how an annotation set ends up agreeing with the system it is supposed
to judge. The join back to gold happens afterwards, by item id.

Sampling is content-addressed — within each stratum the items are ordered by the
sha256 of their id — so the sample cannot be redrawn until it looks convenient.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import multihop_outputs as outputs

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts/semantic-calibration"
SOURCES = (
    ("artifacts/multihop-external.json", "subset.json"),
    ("artifacts/multihop-external-r2.json", "subset-r2.json"),
)
ENTAILMENT = "entailment(supported|partial|unsupported)"
RELEVANCE = "relevance(answers|related|off_topic)"
COMPLETENESS = "completeness(complete|partial|missing)"
READING = "verdict_reading(yes|no|unclear|other_choice|not_applicable)"
EVIDENCE_CHARS = 1500


def arm_for(item_id: str) -> str:
    """Alternate arms deterministically so both paths are represented once each."""
    return "workflow" if int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 2 else "rag"


def collect():
    questions = outputs.dataset()
    stored = outputs.payloads()
    pool = []
    for artifact_name, subset_name in SOURCES:
        path = ROOT / artifact_name
        if not path.exists():
            continue
        external = json.loads(path.read_text())
        items = {item["id"]: item for item in outputs.subset(subset_name)["items"]}
        batch = outputs.batch_number(subset_name)
        seen = set()
        for arm, records in external["results"].items():
            for record in records:
                if record["id"] in seen:
                    continue
                if arm != arm_for(record["id"]):
                    continue
                seen.add(record["id"])
                item = items[record["id"]]
                payload = stored[arm].get(item["query_sha256"]) or {}
                claims = payload.get("claims", [])
                if not claims:
                    # Nothing was asserted, so there is nothing to label for
                    # entailment or relevance. Whether refusing was right is already
                    # decided by the dataset, not by a person.
                    continue
                pool.append(
                    {
                        "item_id": f"B{batch}-{record['id']}",
                        "batch": batch,
                        "arm": arm,
                        "question_type": record["question_type"],
                        "question": questions[item["query_sha256"]]["query"],
                        "claims": claims,
                    }
                )
    return pool


def sample(pool, per_stratum):
    """Stratified, content-addressed, and interleaved.

    The rows are written round-robin across strata rather than grouped by stratum, so
    stopping after any prefix still leaves a balanced sample. Labelling 560 judgements
    in one sitting is not realistic, and a half-labelled file that is balanced is
    usable while a half-labelled file that is all one stratum is not.
    """
    strata = {}
    for row in pool:
        strata.setdefault((row["batch"], row["question_type"]), []).append(row)
    picked = {
        key: sorted(rows, key=lambda row: hashlib.sha256(row["item_id"].encode()).hexdigest())[
            :per_stratum
        ]
        for key, rows in strata.items()
    }
    chosen = []
    for index in range(per_stratum):
        for key in sorted(picked):
            if index < len(picked[key]):
                chosen.append(picked[key][index])
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-stratum", type=int, default=17, help="每个(批次×题型)取几条")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    pool = collect()
    chosen = sample(pool, args.per_stratum)
    evidence = outputs.evidence_text_by_chunk(
        [chunk for row in chosen for claim in row["claims"] for chunk in claim["evidence_ids"]]
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    claim_rows, answer_rows = [], []
    for row in chosen:
        for index, claim in enumerate(row["claims"], 1):
            claim_rows.append(
                {
                    "item_id": row["item_id"],
                    "claim_index": index,
                    "question": row["question"],
                    "claim": claim["text"],
                    "cited_evidence": "\n---\n".join(
                        evidence.get(chunk, "")[:EVIDENCE_CHARS] for chunk in claim["evidence_ids"]
                    ),
                }
            )
        answer_rows.append(
            {
                "item_id": row["item_id"],
                "question": row["question"],
                "answer": "\n".join(claim["text"] for claim in row["claims"]),
            }
        )

    def write(name, fields, rows, labels):
        path = out_dir / name
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields + labels + ["note"])
            for row in rows:
                writer.writerow([row[field] for field in fields] + [""] * (len(labels) + 1))
        return path

    claims_csv = write(
        "claims-to-label.csv",
        ["item_id", "claim_index", "question", "claim", "cited_evidence"],
        claim_rows,
        [ENTAILMENT, RELEVANCE],
    )
    answers_csv = write(
        "answers-to-label.csv",
        ["item_id", "question", "answer"],
        answer_rows,
        [COMPLETENESS, READING],
    )
    frozen = {
        "name": "semantic-calibration-round-1",
        "sources": [name for name, _ in SOURCES],
        "sampling": (
            "按(批次×题型)分层，层内按 sha256(item_id) 升序取前 N；arm 由 sha256(item_id) 的"
            "奇偶决定，因此两条链路各占一半且不重复同一道题。只取产生了 claim 的回答。"
        ),
        "blinding": (
            "标注文件不含金标答案、系统的结论字段，也不含当前指标是否判对；"
            "这些在标注完成后按 item_id 合并。"
        ),
        "labels": {
            "entailment": "被引用的原文是否支持这条 claim",
            "relevance": "这条 claim 是否回答了问题所问",
            "completeness": "整个回答是否覆盖了问题要求的全部部分",
            "verdict_reading": "读者从这个回答里读到的结论是 yes / no / 说不清",
        },
        "items": len(chosen),
        "claims": len(claim_rows),
        "judgements": 2 * len(claim_rows) + 2 * len(chosen),
        "labelling_order": (
            "行按分层轮转排列：只标完前 N 行也仍然是一个平衡样本，可以分几次标。"
        ),
        "verdict_correctness": (
            "不作为标注项。标注者在盲标状态下写下 verdict_reading，之后按 item_id 与金标、"
            "与系统的结论字段合并，正确与否是算出来的——让标注者给系统的结论打分会引入锚定。"
        ),
        "by_stratum": {
            f"B{row['batch']}|{row['question_type']}": sum(
                1
                for other in chosen
                if other["batch"] == row["batch"] and other["question_type"] == row["question_type"]
            )
            for row in chosen
        },
        "by_arm": {
            arm: sum(1 for row in chosen if row["arm"] == arm) for arm in ("rag", "workflow")
        },
        "item_ids": [row["item_id"] for row in chosen],
    }
    frozen_path = out_dir / "sample.json"
    frozen_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in frozen.items() if k != "item_ids"}, ensure_ascii=False, indent=2))
    print(f"{frozen_path}\n{claims_csv}\n{answers_csv}")


if __name__ == "__main__":
    main()
