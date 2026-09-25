"""Record assistant-only diagnoses for B3 failures whose fact proxy arrived.

These labels are retrospective observations of stored claims and verdicts. They
must not be treated as independent human gold or used to tune the frozen B3 run.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/multihop-r3-passage-attribution.json"
OUT = ROOT / "artifacts/multihop-r3-assistant-review.json"


LABELS = {
    ("MH-1cdb794399af", "rag"): (
        "verdict_claim_mismatch", "Claims negate the first condition; the yes verdict reverses their conjunction."
    ),
    ("MH-1cdb794399af", "workflow"): (
        "verdict_claim_mismatch", "Claims say the first source describes legal conduct; the yes verdict disagrees."
    ),
    ("MH-20d14646e9ba", "rag"): (
        "conjunction_not_resolved", "Both requested source propositions are stated, but the verdict remains unclear."
    ),
    ("MH-20d14646e9ba", "workflow"): (
        "conjunction_not_resolved", "Both requested source propositions are stated, but the verdict remains unclear."
    ),
    ("MH-1f989865bc85", "workflow"): (
        "comparison_dimension_needs_source_review", "Claims mix the holiday date with store opening hours."
    ),
    ("MH-229ee6f69011", "rag"): (
        "comparison_dimension_needs_source_review", "Claim compares player availability rather than fantasy strategy."
    ),
}


def main():
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    present = {
        (row["item_id"], row["arm"])
        for row in source["rows"]
        if row["stage_proxy"] == "gold_facts_present_generation_or_scoring_review"
    }
    if present != LABELS.keys():
        raise ValueError(f"review coverage mismatch: missing={present - LABELS.keys()}, extra={LABELS.keys() - present}")
    output = {
        "scope": "six B3 failed arm-runs whose gold fact 4-gram proxy reached the answer context",
        "reviewer_type": "assistant observable-output review; source entailment not independently adjudicated",
        "human_reviewed": 0,
        "assistant_reviewed": len(LABELS),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "rows": [
            {"item_id": key[0], "arm": key[1], "category": label, "basis": basis}
            for key, (label, basis) in sorted(LABELS.items())
        ],
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(LABELS)} assistant reviews written to {OUT}")


if __name__ == "__main__":
    main()
