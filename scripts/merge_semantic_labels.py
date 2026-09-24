"""Compare two independent label sets and hand a person only the disagreements.

Two models labelling the same items is not a gold standard — it is a way to make one
cheap. Where the two agree, the item is provisional; where they disagree, a person
decides. That turns 558 judgements into the few dozen that are actually contested.

Three provenance tiers come out of this, and they are never merged in a report:

- `human_adjudicated`  a person decided this item;
- `model_consensus`    two independent models agreed, nobody checked;
- `contested`          the models disagreed and nobody has decided yet.

Only the first tier may be called gold. A consensus of two models can still be two
models sharing one blind spot, which is why the audit sample below exists: it draws a
fixed fraction of the *agreements* for a person to check, so a shared error has a way
of being noticed instead of being ratified.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "artifacts/semantic-calibration"
LABELS = ("entailment", "relevance", "completeness", "verdict_reading")
KEYS = {"claims": ("item_id", "claim_index"), "answers": ("item_id",)}


def label_columns(fieldnames):
    """Map a label name to whichever column carries it, whatever the suffix."""
    found = {}
    for name in LABELS:
        for column in fieldnames:
            if name in column.lower() and not column.lower().startswith("note"):
                found.setdefault(name, column)
    return found


def read(path, kind):
    with Path(path).open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} 是空的")
    columns = label_columns(rows[0].keys())
    if not columns:
        raise SystemExit(f"{path} 里找不到任何标签列（entailment / relevance / …）")
    keyed = {}
    for row in rows:
        key = tuple(row[name].strip() for name in KEYS[kind])
        keyed[key] = {name: (row.get(column) or "").strip().lower() for name, column in columns.items()}
    return keyed, columns


def audit_sample(keys, fraction, critical=(), critical_fraction=1.0):
    """A fixed, content-addressed slice of the agreements for a person to spot-check.

    `critical` names the agreements that carry the metric P2 exists to measure — an
    `unsupported` claim wrongly read as `supported` is the one error that lets the
    system state something it has no evidence for. Those are audited at their own
    rate, by default all of them; everything else keeps the ordinary fraction.
    """
    def slice_of(subset, share):
        ordered = sorted(subset, key=lambda key: hashlib.sha256("|".join(key).encode()).hexdigest())
        return set(ordered[: max(1, round(len(ordered) * share))]) if ordered else set()

    critical = set(critical) & set(keys)
    return slice_of(critical, critical_fraction) | slice_of(set(keys) - critical, fraction)


def compare(kind, first, second, fraction, critical_label=None, critical_value=None, critical_fraction=1.0):
    left, _ = read(first, kind)
    right, _ = read(second, kind)
    shared = sorted(set(left) & set(right))
    stats, contested, agreed = {}, [], []
    for key in shared:
        disagreed = []
        for name in sorted(set(left[key]) & set(right[key])):
            a, b = left[key][name], right[key][name]
            if not a or not b:
                continue
            bucket = stats.setdefault(name, {"compared": 0, "agree": 0, "pairs": {}})
            bucket["compared"] += 1
            bucket["agree"] += a == b
            bucket["pairs"][f"{a}|{b}"] = bucket["pairs"].get(f"{a}|{b}", 0) + 1
            if a != b:
                disagreed.append((name, a, b))
        (contested if disagreed else agreed).append((key, disagreed))
    critical = [
        key
        for key, _ in agreed
        if critical_label and left[key].get(critical_label) == critical_value
    ]
    audited = audit_sample(
        [key for key, _ in agreed], fraction, critical, critical_fraction
    )
    for name, bucket in stats.items():
        bucket["agreement_rate"] = round(bucket["agree"] / bucket["compared"], 3) if bucket["compared"] else None
    return {
        "kind": kind,
        "items_compared": len(shared),
        "only_in_first": len(set(left) - set(right)),
        "only_in_second": len(set(right) - set(left)),
        "contested_items": len(contested),
        "agreed_items": len(agreed),
        "audit_sample": len(audited),
        "audit_critical": len(set(critical) & audited),
        "human_judgements_required": len(contested) + len(audited),
        "by_label": stats,
    }, contested, agreed, audited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["claims", "answers"], required=True)
    parser.add_argument("--first", required=True, help="第一份模型标注")
    parser.add_argument("--second", required=True, help="第二份模型标注（必须独立完成）")
    parser.add_argument("--audit-fraction", type=float, default=0.2)
    parser.add_argument("--critical-label", default="entailment")
    parser.add_argument("--critical-value", default="unsupported")
    parser.add_argument("--critical-fraction", type=float, default=1.0)
    parser.add_argument("--out-dir", default=str(DIR))
    args = parser.parse_args()
    summary, contested, agreed, audited = compare(
        args.kind,
        args.first,
        args.second,
        args.audit_fraction,
        args.critical_label,
        args.critical_value,
        args.critical_fraction,
    )
    out_dir = Path(args.out_dir)
    review = out_dir / f"to-adjudicate-{args.kind}.csv"
    with review.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(KEYS[args.kind]) + ["reason", "label", "first_says", "second_says", "human_decision", "note"])
        for key, disagreed in contested:
            for name, a, b in disagreed:
                writer.writerow(list(key) + ["disagreement", name, a, b, "", ""])
        for key, _ in agreed:
            if key in audited:
                writer.writerow(list(key) + ["audit_of_agreement", "", "", "", "", ""])
    path = out_dir / f"agreement-{args.kind}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"{path}\n{review}  （需要人判断 {summary['human_judgements_required']} 处）")


if __name__ == "__main__":
    main()
