"""Build the CODEBOOK v3 final-review package for a fresh GPT session.

Round 2 was labelled by two annotators (Claude's machine prepass and GPT) — but GPT
labelled against CODEBOOK v2, which never carried the owner's four rulings. v3 writes
them in as rules 11–14, and the version discipline says every affected sample, from
both rounds, is re-decided under v3. This builds that review:

- entailment: every round-2 disagreement (labels shown as anonymous A/B), every claim
  both annotators called `unsupported` (the critical class), a fixed 20% of the other
  agreements, and every relational claim from either round (rule 12);
- relevance: all claims from both rounds, blind — rules 11 and 14 change how relevance
  is read, so no earlier relevance label survives unreviewed;
- answers: round 2 had one annotator, so every `partial` / `unclear` / `other_choice`
  plus a fixed 20% of the rest, and round-1 gold answers that rule 13 can touch.

Nothing here decides a label. The reviewer fills the `final_*` columns.
"""

import csv
import hashlib
import re
from pathlib import Path

from merge_semantic_labels import audit_sample

DIR = Path(__file__).resolve().parent.parent / "artifacts" / "semantic-calibration"
OUT = DIR / "round2"

ENT = "entailment(supported|partial|unsupported)"
REL = "relevance(answers|related|off_topic)"
COMP = "completeness(complete|partial|missing)"
VERD = "verdict_reading(yes|no|unclear|other_choice|not_applicable)"

RELATIONAL = re.compile(
    r"\b(consisten\w*|inconsisten\w*|chang\w*|same|differ\w*|dissimilar|similar\w*|agree\w*|contrast\w*)\b"
    r"|一致|变化|不同|相同",
    re.IGNORECASE,
)


def rows(name):
    with (DIR / name).open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def keyed(items, *fields):
    return {tuple(row[field] for field in fields): row for row in items}


def a_b(key, first, second):
    """Present two annotators as A/B in an order the reviewer cannot map back to a source."""
    if int(hashlib.sha256("|".join(key).encode()).hexdigest(), 16) % 2:
        return second, first
    return first, second


def write(name, header, body):
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(body)
    return path, len(body)


def main():
    OUT.mkdir(exist_ok=True)
    text1 = keyed(rows("claims-to-label.csv"), "item_id", "claim_index")
    text2 = keyed(rows("claims-to-label-rest126.csv"), "item_id", "claim_index")
    gold1 = keyed(rows("gold-claims.csv"), "item_id", "claim_index")
    mine = keyed(rows("machine-prepass-claims-rest.csv"), "item_id", "claim_index")
    gpt = keyed(rows("gpt-prepass-claims-rest.csv"), "item_id", "claim_index")

    # entailment
    contested, unsupported, agreed = [], [], []
    for key in text2:
        a, b = mine[key]["entailment_MODEL"], gpt[key][ENT]
        if a != b:
            contested.append(key)
        elif a == "unsupported":
            unsupported.append(key)
        else:
            agreed.append(key)
    sampled = audit_sample(agreed, 0.2, (), 1.0)
    entail = []
    for key in text2:
        text = text2[key]
        a, b = mine[key]["entailment_MODEL"], gpt[key][ENT]
        if key in contested:
            left, right = a_b(key, a, b)
            reason, current = "disagreement", ""
        elif key in unsupported:
            left = right = ""
            reason, current = "audit_unsupported_agreement", a
        elif key in sampled:
            left = right = ""
            reason, current = "audit_sample_20pct", a
        elif RELATIONAL.search(text["claim"]):
            left = right = ""
            reason, current = "v3_rule12_recheck", a
        else:
            continue
        entail.append([*key, "R2", reason, text["question"], text["claim"], text["cited_evidence"],
                       left, right, current, "", ""])
    for key, gold in gold1.items():
        text = text1[key]
        if RELATIONAL.search(text["claim"]):
            entail.append([*key, "R1", "v3_rule12_recheck", text["question"], text["claim"],
                           text["cited_evidence"], "", "", gold["entailment"], "", ""])

    # relevance: blind, every claim in both rounds
    relevance = [[*key, "R1", text1[key]["question"], text1[key]["claim"], "", ""] for key in gold1]
    relevance += [[*key, "R2", text["question"], text["claim"], "", ""] for key, text in text2.items()]

    # answers
    answers2 = rows("gpt-prepass-answers-rest.csv")
    flagged = [row["item_id"] for row in answers2
               if row[COMP] == "partial" or row[VERD] in ("unclear", "other_choice")]
    rest = [row["item_id"] for row in answers2 if row["item_id"] not in flagged]
    answer_sample = {key[0] for key in audit_sample([(item,) for item in rest], 0.2, (), 1.0)}
    answer_rows = []
    for row in answers2:
        item = row["item_id"]
        if item in flagged:
            reason = "partial_unclear_or_other_choice"
        elif item in answer_sample:
            reason = "audit_sample_20pct"
        else:
            continue
        answer_rows.append([item, "R2", reason, row["question"], row["answer"], row[COMP], row[VERD], "", "", ""])
    text_a1 = keyed(rows("answers-to-label.csv"), "item_id")
    for gold in rows("gold-answers.csv"):
        if gold["verdict_reading"] == "other_choice":
            text = text_a1[(gold["item_id"],)]
            answer_rows.append([gold["item_id"], "R1", "v3_rule13_recheck", text["question"], text["answer"],
                                gold["completeness"], gold["verdict_reading"], "", "", ""])

    outputs = [
        write("final-review-entailment.csv",
              ["item_id", "claim_index", "round", "reason", "question", "claim", "cited_evidence",
               "label_A", "label_B", "current_label", "final_entailment(supported|partial|unsupported)", "note"],
              entail),
        write("final-review-relevance.csv",
              ["item_id", "claim_index", "round", "question", "claim",
               "final_relevance(answers|related|off_topic)", "note"],
              relevance),
        write("final-review-answers.csv",
              ["item_id", "round", "reason", "question", "answer", "current_completeness", "current_verdict_reading",
               "final_completeness(complete|partial|missing)",
               "final_verdict_reading(yes|no|unclear|other_choice|not_applicable)", "note"],
              answer_rows),
    ]
    for path, count in outputs:
        print(f"{path.relative_to(DIR.parent.parent)}  {count} 行")
    reasons = {}
    for row in entail:
        reasons[row[3]] = reasons.get(row[3], 0) + 1
    print("entailment:", reasons)


if __name__ == "__main__":
    main()
