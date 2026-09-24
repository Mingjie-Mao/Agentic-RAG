"""Apply the human adjudication to the two model label sets and emit tiered gold.

Two models labelled the same items independently. Where they disagreed, a person
decided — not item by item, but by writing four rules that the disagreements clustered
into. This applies those rules and records, for every row, *who* decided it:

- `human_adjudicated`  the row was contested and a human rule settled it;
- `rule_applied`       the models agreed, but a human rule changed the label anyway
                       (a rule has to apply uniformly or the gold is inconsistent);
- `model_consensus`    the models agreed and no rule touched it — **not gold**;
- `model_consensus_audit_pending` the same, but drawn into the audit sample a person
                       still has to spot-check;
- `model_consensus_gpt_confirmed` / `model_override_gpt_confirmed`  a further model
                       read the agreed label and accepted or changed it. A third judgement
                       adds information, but it is still a model's — **not gold**.

Project standard (decided by the project owner on 2026-09-23): GPT review is the final
review and no person reviews items one by one. Every adjudicated tier is therefore used as
this project's gold — but each row still records *who* decided it, and the count of
item-level human review is reported as what it is: zero. A reader can then apply a
stricter standard without re-deriving anything.

The adjudication table below is written out item by item on purpose: a reader can
check each decision against `ADJUDICATION.md` without rerunning anything.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "artifacts/semantic-calibration"

# 决定 1：合取问句里只回答一支 → relevance = answers（12 条争议，全部采纳「我」的一侧）
# 决定 2：关系断言「A 与 B 一致」——两侧事实足以推出 → supported；两侧都在但不足以推出 → unsupported；
#         一侧完全无证据 → unsupported；只有写成「A 说 X、B 说 Y、故一致」且 X/Y 部分成立时才 partial
# 决定 3：给了数值但没选定分支 → completeness=partial，verdict_reading=unclear；
#         other_choice 只用于明确选定了某一非是非分支
# 决定 4：wh 问句里只确认识别性限定语 → relevance = related
CLAIM_DECISIONS = {
    # 决定 1（relevance 争议 12 条）
    ("B1-MH-050827e2c17f", "1"): {"relevance": ("answers", "decision-1")},
    ("B1-MH-050827e2c17f", "2"): {"relevance": ("answers", "decision-1")},
    ("B1-MH-063f46090a91", "1"): {"relevance": ("answers", "decision-1")},
    ("B1-MH-063f46090a91", "2"): {"relevance": ("answers", "decision-1")},
    ("B1-MH-0a7a2c277008", "1"): {"relevance": ("answers", "decision-1")},
    ("B1-MH-0a7a2c277008", "2"): {
        "relevance": ("answers", "decision-1"),
        "entailment": ("unsupported", "方向写反，属与证据冲突而非否定式断言"),
    },
    ("B2-MH-1291bbe87804", "1"): {"relevance": ("answers", "decision-1")},
    ("B2-MH-1291bbe87804", "2"): {"relevance": ("answers", "decision-1")},
    ("B2-MH-19c9e32e360e", "1"): {"relevance": ("answers", "decision-1")},
    ("B2-MH-19c9e32e360e", "2"): {"relevance": ("answers", "decision-1")},
    ("B2-MH-179cc0ffdb05", "1"): {"relevance": ("answers", "decision-1")},
    ("B2-MH-179cc0ffdb05", "2"): {"relevance": ("answers", "decision-1")},
    # 决定 2（关系断言）
    ("B1-MH-0acddcd35850", "2"): {"entailment": ("supported", "decision-2: 两篇被引报道都指向缺席，足以推出一致")},
    ("B2-MH-1144d2e80b35", "1"): {"entailment": ("supported", "decision-2: 两侧摘录都强调信仰实践，足以推出一致")},
    ("B2-MH-1afa5edb190c", "5"): {"entailment": ("unsupported", "decision-2: TechCrunch 一侧完全不在引用里")},
    ("B1-MH-0cd2bbe92b19", "1"): {"entailment": ("unsupported", "decision-2: 一致的另一侧（U2 项目）不在引用里；两模型原本都判 partial")},
    # 决定 4 与两条个案
    ("B2-MH-0f910380b759", "3"): {"relevance": ("related", "decision-4")},
    ("B2-MH-10482bbfd0ca", "1"): {"entailment": ("unsupported", "与可见信息冲突：引用写 GPT-4，claim 写 GPT-3.5")},
}
ANSWER_DECISIONS = {
    "B2-MH-141809cfa779": {
        "completeness": ("partial", "decision-3: 必答项是比较本身"),
        "verdict_reading": ("unclear", "decision-3: 未选定分支"),
    },
    "B2-MH-1291bbe87804": {"verdict_reading": ("unclear", "decision-3: 析取问句，未选定分支")},
}
# 一致项的确认。确认人必须写明：只有 human 能把一行升级为金标；模型做的确认是又一次
# 模型判读，信息量更高，但性质不变。2026-09-23 这一轮由 GPT 完成，项目负责人确认过来源。
CONFIRMATIONS = (
    ("consensus-confirmed-claims.csv", "claims", "gpt"),
    ("consensus-confirmed-answers.csv", "answers", "gpt"),
)
KEYS = {"claims": ("item_id", "claim_index"), "answers": ("item_id",)}
# 项目标准（负责人 2026-09-23 决定）：GPT 审核为最终审核，不做人工逐条复核。
# 因此这些档都作为本项目的 gold 使用；但每一行都记下由谁定，人工逐条复核计数如实为 0。
ADJUDICATED = {
    "owner_rule_adjudicated",
    "owner_rule_applied",
    "gpt_final_review_accept",
    "gpt_final_review_override",
    "human_item_review",
}
LABELS = {"claims": ("entailment", "relevance"), "answers": ("completeness", "verdict_reading")}


def read(path, kind):
    with Path(path).open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    columns = {}
    for name in LABELS[kind]:
        for column in rows[0]:
            if name in column.lower() and not column.lower().startswith("note"):
                columns.setdefault(name, column)
    return {
        tuple(row[key].strip() for key in KEYS[kind]): {
            name: row[column].strip().lower() for name, column in columns.items()
        }
        for row in rows
    }


def confirmations(kind):
    """{key: (decision, {label: override}, confirmed_by, note)} from every confirmation file."""
    found = {}
    for name, confirmed_kind, confirmed_by in CONFIRMATIONS:
        path = DIR / name
        if confirmed_kind != kind or not path.exists():
            continue
        if confirmed_by not in {"human", "gpt"}:
            raise SystemExit(f"{name}: 确认人必须是 human 或 gpt，得到 {confirmed_by!r}")
        with path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                decision_column = next(c for c in row if c.startswith("accept_or_override"))
                decision = row[decision_column].strip().lower()
                if decision not in {"accept", "override"}:
                    raise SystemExit(f"{name}: 未填或非法的决定 {decision!r}")
                overrides = {
                    label: row[f"override_{label}"].strip().lower()
                    for label in LABELS[kind]
                    if row.get(f"override_{label}", "").strip()
                }
                key = tuple(row[column].strip() for column in KEYS[kind])
                found[key] = (decision, overrides, confirmed_by, row.get("note", "").strip())
    return found


def build(kind, first, second, decisions, audit):
    left, right = read(first, kind), read(second, kind)
    confirmed = confirmations(kind)
    rows, tiers = [], {}
    for key in sorted(set(left) & set(right)):
        decided = decisions.get(key if kind == "claims" else key[0], {})
        record = dict(zip(KEYS[kind], key))
        contested = any(left[key][name] != right[key][name] for name in LABELS[kind])
        basis = []
        for name in LABELS[kind]:
            if name in decided:
                record[name], reason = decided[name]
                basis.append(f"{name}:{reason}")
            elif left[key][name] == right[key][name]:
                record[name] = left[key][name]
            else:
                # Contested and not covered by a rule: the gold cannot claim a value.
                record[name] = ""
                basis.append(f"{name}:unresolved")
        if contested and decided:
            tier = "owner_rule_adjudicated"
        elif decided:
            tier = "owner_rule_applied"
        elif key in confirmed:
            decision, overrides, confirmed_by, note = confirmed[key]
            for name, value in overrides.items():
                record[name] = value
                basis.append(f"{name}:override_by_{confirmed_by}" + (f" ({note[:80]})" if note else ""))
            if confirmed_by == "human":
                tier = "human_item_review"
            else:
                tier = "gpt_final_review_override" if decision == "override" else "gpt_final_review_accept"
        elif key in audit:
            tier = "model_consensus_audit_pending"
        elif contested:
            tier = "contested_unresolved"
        else:
            tier = "model_consensus"
        record["provenance"] = tier
        record["basis"] = "; ".join(basis)
        rows.append(record)
        tiers[tier] = tiers.get(tier, 0) + 1
    return rows, tiers


def main():
    summary = {}
    for kind, first, second, decisions in (
        ("claims", "machine-prepass-claims.csv", "gpt-prepass-claims.csv", CLAIM_DECISIONS),
        ("answers", "machine-prepass-answers.csv", "gpt-prepass-answers.csv", ANSWER_DECISIONS),
    ):
        review = DIR / f"to-adjudicate-{kind}.csv"
        audit = set()
        if review.exists():
            with review.open(encoding="utf-8-sig") as handle:
                audit = {
                    tuple(row[key].strip() for key in KEYS[kind])
                    for row in csv.DictReader(handle)
                    if row["reason"] == "audit_of_agreement"
                }
        rows, tiers = build(kind, DIR / first, DIR / second, decisions, audit)
        path = DIR / f"gold-{kind}.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        gold_rows = [row for row in rows if row["provenance"] in ADJUDICATED]
        summary[kind] = {
            "rows": len(rows),
            "by_provenance": tiers,
            "usable_as_gold": len(gold_rows),
            "human_item_level_review": tiers.get("human_item_review", 0),
        }
        if kind == "claims":
            summary[kind]["unsupported_in_gold"] = sum(row["entailment"] == "unsupported" for row in gold_rows)
            summary[kind]["unsupported_awaiting_review"] = sum(
                row["entailment"] == "unsupported" and row["provenance"] not in ADJUDICATED
                for row in rows
            )
    (DIR / "gold-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
