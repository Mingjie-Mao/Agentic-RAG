"""Merge the CODEBOOK v3 final review into one gold set, and freeze it.

Inputs: the v2 gold (round 1), both independent round-2 annotations, and the three
final-review files a fresh GPT session filled in (`round2/final-review-*.csv`).
Every row records who decided each label, and a row the final review never saw keeps
the tier `model_consensus`, which the evaluation does not use.

Round 2 audited agreements by sampling: every `unsupported` agreement and every
relational claim was reviewed, but only 20% of the other agreements. The sampled rows
carry an inverse-probability weight so the evaluation can report metrics for the class
mix the labels actually came from, not for the mix the audit happened to over-sample.

`--freeze` writes `gold-v3-manifest.json` with the sha256 of every input and output;
the evaluation refuses to run against files that no longer match it. A manifest that
already exists is never overwritten.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_semantic_annotation_set import arm_for  # noqa: E402
from build_semantic_final_review import RELATIONAL  # noqa: E402
import multihop_outputs as outputs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "artifacts" / "semantic-calibration"
REVIEW = DIR / "round2"

ENTAILMENT = {"supported", "partial", "unsupported"}
RELEVANCE = {"answers", "related", "off_topic"}
COMPLETENESS = {"complete", "partial", "missing"}
VERDICT = {"yes", "no", "unclear", "other_choice", "not_applicable"}
GOLD_TIERS_V2 = {
    "owner_rule_adjudicated",
    "owner_rule_applied",
    "gpt_final_review_accept",
    "gpt_final_review_override",
    "human_item_review",
}
SUBSETS = {"1": "subset.json", "2": "subset-r2.json"}

# Relevance labels the blind v3 review got wrong against the owner's rulings (2026-09-24).
# The owner kept their item-level ruling on B1-MH-050827e2c17f and had rule 11 rewritten
# (v3.1) to match it; the same rule then applies to every question of that shape, so the
# blind labels that contradict it are corrected here rather than left inconsistent.
RELEVANCE_RULINGS = {
    ("B1-MH-050827e2c17f", "1"): ("answers", "owner_ruling_decision_1"),
    ("B1-MH-050827e2c17f", "2"): ("answers", "owner_ruling_decision_1"),
    ("B1-MH-0c6d6ba18794", "1"): ("answers", "owner_rule_applied_v3_1"),
    ("B1-MH-0c6d6ba18794", "2"): ("answers", "owner_rule_applied_v3_1"),
    ("B1-MH-08c95043a751", "1"): ("answers", "owner_rule_applied_v3_1"),
    ("B1-MH-08c95043a751", "2"): ("answers", "owner_rule_applied_v3_1"),
    ("B2-MH-12b9416cf2b0", "1"): ("answers", "owner_rule_applied_v3_1"),
    ("B2-MH-12b9416cf2b0", "2"): ("answers", "owner_rule_applied_v3_1"),
    ("B1-MH-0fb5fdf6ee28", "1"): ("answers", "owner_rule_applied_v3_1"),
}


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def column(row, prefix):
    return next(value for name, value in row.items() if name.startswith(prefix)).strip().lower()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata():
    """{item_id: (batch, question_type, arm)} for every labelled item."""
    found = {}
    for batch, name in SUBSETS.items():
        for item in outputs.subset(name)["items"]:
            found[f"B{batch}-{item['id']}"] = (f"B{batch}", item["question_type"], arm_for(item["id"]))
    return found


def check(value, allowed, where):
    if value not in allowed:
        raise SystemExit(f"{where}: 非法或未填的标签 {value!r}（允许 {sorted(allowed)}）")
    return value


def review_decision(final, current):
    return "accept" if final == current else "override"


def build_claims(directory, review):
    text = {(r["item_id"], r["claim_index"]): r for r in rows(directory / "claims-to-label.csv")}
    gold_v2 = {(r["item_id"], r["claim_index"]): r for r in rows(directory / "gold-claims.csv")}
    mine = {(r["item_id"], r["claim_index"]): r for r in rows(directory / "machine-prepass-claims-rest.csv")}
    gpt = {(r["item_id"], r["claim_index"]): r for r in rows(directory / "gpt-prepass-claims-rest.csv")}
    entail_review = {(r["item_id"], r["claim_index"]): r for r in rows(review / "final-review-entailment.csv")}
    relevance_review = {(r["item_id"], r["claim_index"]): r for r in rows(review / "final-review-relevance.csv")}

    # Inverse-probability weights for the round-2 stratum that was sampled rather than
    # reviewed whole: agreements that are neither `unsupported` nor relational (those two
    # were always reviewed, so a relational claim the sample also drew still has weight 1).
    stratum = [
        key
        for key in mine
        if mine[key]["entailment_MODEL"] == column(gpt[key], "entailment")
        and mine[key]["entailment_MODEL"] != "unsupported"
        and not RELATIONAL.search(text[key]["claim"])
    ]
    sampled = {key for key in stratum if entail_review.get(key, {}).get("reason") == "audit_sample_20pct"}
    weight = round(len(stratum) / len(sampled), 4) if sampled else 1.0

    out = []
    for key in text:
        where = f"{key[0]}#{key[1]}"
        record = {"item_id": key[0], "claim_index": key[1]}
        reviewed = entail_review.get(key)
        if key in gold_v2:
            record["round"] = "R1"
            current = gold_v2[key]["entailment"]
            if reviewed:
                final = check(column(reviewed, "final_entailment"), ENTAILMENT, where)
                record["entailment"] = final
                record["entailment_provenance"] = f"gpt_final_review_v3_{review_decision(final, current)}"
            else:
                if gold_v2[key]["provenance"] not in GOLD_TIERS_V2:
                    raise SystemExit(f"{where}: 第 1 轮行不在 gold 档：{gold_v2[key]['provenance']}")
                record["entailment"] = current
                record["entailment_provenance"] = f"v2_{gold_v2[key]['provenance']}"
            record["weight"] = 1.0
        else:
            record["round"] = "R2"
            a, b = mine[key]["entailment_MODEL"], column(gpt[key], "entailment")
            if reviewed:
                final = check(column(reviewed, "final_entailment"), ENTAILMENT, where)
                record["entailment"] = final
                if reviewed["reason"] == "disagreement":
                    record["entailment_provenance"] = "gpt_final_review_v3_disagreement"
                else:
                    record["entailment_provenance"] = f"gpt_final_review_v3_{review_decision(final, a)}"
                record["weight"] = weight if key in sampled else 1.0
            else:
                # Two annotators agreed and nobody reviewed it: not gold (owner, 2026-09-23).
                record["entailment"] = a if a == b else ""
                record["entailment_provenance"] = "model_consensus" if a == b else "contested_unresolved"
                record["weight"] = 0.0
        relevance = relevance_review.get(key)
        if relevance is None:
            raise SystemExit(f"{where}: relevance 盲标文件里缺这一行")
        record["relevance"] = check(column(relevance, "final_relevance"), RELEVANCE, where)
        record["relevance_provenance"] = "gpt_final_review_v3_blind"
        if key in RELEVANCE_RULINGS:
            record["relevance"], record["relevance_provenance"] = RELEVANCE_RULINGS[key]
        out.append(record)
    return out, {"stratum": len(stratum), "sampled": len(sampled), "weight": weight}


def build_answers(directory, review):
    gold_v2 = {r["item_id"]: r for r in rows(directory / "gold-answers.csv")}
    gpt = {r["item_id"]: r for r in rows(directory / "gpt-prepass-answers-rest.csv")}
    reviewed = {r["item_id"]: r for r in rows(review / "final-review-answers.csv")}
    out = []
    for item, source in [*gold_v2.items(), *gpt.items()]:
        record = {"item_id": item, "round": "R1" if item in gold_v2 else "R2"}
        current = (
            (source["completeness"], source["verdict_reading"])
            if item in gold_v2
            else (column(source, "completeness"), column(source, "verdict_reading"))
        )
        if item in reviewed:
            final = (
                check(column(reviewed[item], "final_completeness"), COMPLETENESS, item),
                check(column(reviewed[item], "final_verdict_reading"), VERDICT, item),
            )
            record["completeness"], record["verdict_reading"] = final
            record["provenance"] = f"gpt_final_review_v3_{review_decision(final, current)}"
        elif item in gold_v2:
            record["completeness"], record["verdict_reading"] = current
            record["provenance"] = f"v2_{source['provenance']}"
        else:
            record["completeness"], record["verdict_reading"] = current
            record["provenance"] = "single_annotator_unreviewed"
        out.append(record)
    return out


def write(path, records):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def composition(claims, answers):
    meta = metadata()
    gold = [row for row in claims if row["weight"] > 0]

    def count(records, key):
        found = {}
        for row in records:
            value = key(row)
            found[value] = found.get(value, 0) + 1
        return dict(sorted(found.items()))

    return {
        "claims_total": len(claims),
        "entailment_gold": len(gold),
        "entailment_by_label": count(gold, lambda r: r["entailment"]),
        "entailment_by_provenance": count(claims, lambda r: r["entailment_provenance"]),
        "unsupported_in_gold": sum(r["entailment"] == "unsupported" for r in gold),
        "by_batch": count(gold, lambda r: meta[r["item_id"]][0]),
        "by_question_type": count(gold, lambda r: meta[r["item_id"]][1]),
        "by_arm": count(gold, lambda r: meta[r["item_id"]][2]),
        "unsupported_by_question_type": count(
            [r for r in gold if r["entailment"] == "unsupported"], lambda r: meta[r["item_id"]][1]
        ),
        "relevance_by_label": count(claims, lambda r: r["relevance"]),
        "relevance_by_provenance": count(claims, lambda r: r["relevance_provenance"]),
        "answers_gold": sum(r["provenance"] != "single_annotator_unreviewed" for r in answers),
        "answers_by_provenance": count(answers, lambda r: r["provenance"]),
        "human_item_level_review": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="写冻结清单；已存在则拒绝")
    args = parser.parse_args()

    claims, weights = build_claims(DIR, REVIEW)
    answers = build_answers(DIR, REVIEW)
    write(DIR / "gold-v3-claims.csv", claims)
    write(DIR / "gold-v3-answers.csv", answers)
    summary = {"codebook": "v3", "sampling_weight": weights, **composition(claims, answers)}
    (DIR / "gold-v3-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.freeze:
        manifest = DIR / "gold-v3-manifest.json"
        if manifest.exists():
            raise SystemExit(f"{manifest.name} 已存在：gold 已冻结，不覆盖")
        tracked = [
            "gold-v3-claims.csv",
            "gold-v3-answers.csv",
            "gold-v3-summary.json",
            "CODEBOOK.md",
            "PREREGISTRATION-P2.md",
            "round2/final-review-entailment.csv",
            "round2/final-review-relevance.csv",
            "round2/final-review-answers.csv",
        ]
        manifest.write_text(
            json.dumps(
                {
                    "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "final_reviewer": "gpt (fresh session); human item-level review = 0",
                    "sha256": {name: sha256(DIR / name) for name in tracked},
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(f"已冻结：{manifest}")


if __name__ == "__main__":
    main()
