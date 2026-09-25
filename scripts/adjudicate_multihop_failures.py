"""Record an assistant review of all historical fact-present B1/B2 failures.

This is an observable-output diagnosis, not independent human annotation or a
fresh evaluation. The private queue holds the third-party question and answer
text; this committed artifact contains IDs and failure categories only.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / ".runtime/multihop/manual-review-queue.json"
OUT = ROOT / "artifacts/multihop-assistant-review-all42.json"


# Ordered like the private review queue, but each row is identified independently
# below so that accidental reordering cannot silently attach a label to a new run.
REVIEWS = """
B1-MH-0085f76defbe rag missing_explicit_verdict
B1-MH-0175090e1c35 rag reversed_conclusion
B1-MH-03c8f9157597 rag missing_explicit_verdict
B1-MH-04e2c330afa3 rag false_refusal
B1-MH-050827e2c17f rag missing_explicit_verdict
B1-MH-05810be2d274 rag missing_explicit_verdict
B1-MH-063f46090a91 rag reversed_conclusion
B1-MH-0b856ab69624 rag missing_explicit_verdict
B1-MH-0c69b8fdbe62 rag reversed_conclusion
B1-MH-0e1f6d12ec46 rag gold_interpretation_needs_source_review
B1-MH-10cbd523d4ee rag missing_explicit_verdict
B1-MH-0422ce95acd1 rag false_refusal
B1-MH-035ed76f80f8 rag missing_explicit_verdict
B1-MH-0401d5c77bd1 rag incomplete_temporal_comparison
B1-MH-0085f76defbe workflow missing_explicit_verdict
B1-MH-0175090e1c35 workflow missing_explicit_verdict
B1-MH-03c8f9157597 workflow missing_explicit_verdict
B1-MH-04e2c330afa3 workflow false_refusal
B1-MH-050827e2c17f workflow missing_explicit_verdict
B1-MH-063f46090a91 workflow missing_explicit_verdict
B1-MH-0b65c4cf02ac workflow question_echo
B1-MH-0b856ab69624 workflow missing_explicit_verdict
B1-MH-0c69b8fdbe62 workflow missing_explicit_verdict
B1-MH-0e1f6d12ec46 workflow gold_interpretation_needs_source_review
B1-MH-10cbd523d4ee workflow missing_explicit_verdict
B1-MH-035ed76f80f8 workflow missing_explicit_verdict
B1-MH-0401d5c77bd1 workflow incomplete_temporal_comparison
B1-MH-0b661c6477e4 workflow missing_explicit_verdict
B1-MH-0cd2bbe92b19 workflow missing_explicit_verdict
B2-MH-12b9416cf2b0 rag wrong_comparison
B2-MH-1388f62eabd7 rag scorer_true_false_gap
B2-MH-172c2bb1310b rag false_refusal
B2-MH-1b7a0b37afed rag false_refusal
B2-MH-141809cfa779 rag missing_comparison_label
B2-MH-12b9416cf2b0 workflow wrong_comparison
B2-MH-1388f62eabd7 workflow scorer_true_false_gap
B2-MH-172c2bb1310b workflow false_refusal
B2-MH-17e563aaa33e workflow verdict_claim_mismatch
B2-MH-185f9f13efd8 workflow verdict_claim_mismatch
B2-MH-1b7a0b37afed workflow false_refusal
B2-MH-11d9213f77f1 workflow null_false_answer
B2-MH-209e10929f9c workflow incomplete_temporal_comparison
"""


def main():
    queue_bytes = QUEUE.read_bytes()
    queue = json.loads(queue_bytes)
    lookup = {(row["item_id"], row["arm"]): row for row in queue}
    if len(lookup) != len(queue):
        raise ValueError("duplicate private review key")
    labels = {}
    for line in REVIEWS.strip().splitlines():
        item_id, arm, label = line.split()
        key = (item_id, arm)
        if key in labels:
            raise ValueError(f"duplicate label: {key}")
        labels[key] = label
    if labels.keys() != lookup.keys():
        raise ValueError(f"review coverage mismatch: missing={lookup.keys() - labels.keys()}, extra={labels.keys() - lookup.keys()}")
    rows = [{"item_id": key[0], "arm": key[1], "category": labels[key]}
            for key in sorted(labels)]
    result = {
        "scope": "all 42 B1/B2 failed arm-runs whose gold fact 4-gram proxy reached the generation context",
        "reviewer_type": "assistant observable-output review; not independent human gold",
        "private_queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "caution": "Gold and source support were not independently adjudicated. Categories describe visible answer and scorer behavior only.",
        "human_reviewed": 0,
        "assistant_reviewed": len(rows),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["category_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
