"""Build a reproducible review queue from the two *used* external batches.

This is a triage report, not a new accuracy score. Document presence is recorded by
the original benchmark run; semantic labels are joined only for the sampled arm and
only when they passed the project's final review. Signals can overlap and cannot, by
themselves, distinguish a wrong paragraph from wrong reasoning. No batch-3 data is
read and no model or database service is needed.
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "artifacts/multihop-external.json",
    ROOT / "artifacts/multihop-external-r2.json",
)
CLAIMS = ROOT / "artifacts/semantic-calibration/gold-v3-claims.csv"
ANSWERS = ROOT / "artifacts/semantic-calibration/gold-v3-answers.csv"
OUTPUT = ROOT / "artifacts/multihop-failure-attribution.json"
MANIFEST = ROOT / "artifacts/semantic-calibration/gold-v3-manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_gold(manifest_path=MANIFEST, claims_path=CLAIMS, answers_path=ANSWERS):
    manifest = json.loads(manifest_path.read_text())
    for path in (claims_path, answers_path):
        if sha256(path) != manifest["sha256"][path.name]:
            raise ValueError(f"frozen semantic labels changed: {path}")


def reviewed_labels(claims_path: Path, answers_path: Path):
    claims = defaultdict(list)
    with claims_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["entailment_provenance"] != "model_consensus":
                claims[row["item_id"]].append(row["entailment"])
    answers = {}
    with answers_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["provenance"] != "single_annotator_unreviewed":
                answers[row["item_id"]] = row
    return claims, answers


def signals_for(record: dict, claim_labels: list[str], answer_label: dict | None):
    signals = []
    if record["gold_documents_retrieved"] < record["gold_documents"]:
        signals.append("gold_document_missing")
    elif record["gold_documents"]:
        # The run did not record gold-fact passage coverage. This is a review prompt,
        # not evidence that the right paragraph was retrieved.
        signals.append("gold_document_present_fact_unverified")
    if any(label in {"partial", "unsupported"} for label in claim_labels):
        signals.append("reviewed_claim_not_fully_supported")
    if answer_label and answer_label["completeness"] in {"partial", "missing"}:
        signals.append("reviewed_answer_incomplete")
    reader = answer_label["verdict_reading"] if answer_label else None
    system = record.get("verdict")
    if reader in {"yes", "no"} and system in {"yes", "no"} and reader != system:
        signals.append("system_verdict_disagrees_with_reviewed_reading")
    if not claim_labels and not answer_label:
        signals.append("no_reviewed_semantic_label")
    return signals


def build(sources=SOURCES, claims_path=CLAIMS, answers_path=ANSWERS):
    claims, answers = reviewed_labels(claims_path, answers_path)
    failures = []
    seen = set()
    for batch, path in enumerate(sources, 1):
        artifact = json.loads(path.read_text())
        for arm, records in artifact["results"].items():
            for record in records:
                key = (batch, arm, record["id"])
                if key in seen:
                    raise ValueError(f"duplicate benchmark result: {key}")
                seen.add(key)
                if record["answer_correct"]:
                    continue
                item_id = f"B{batch}-{record['id']}"
                # Each annotated question selected exactly one arm before labelling.
                selected_arm = "workflow" if int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 2 else "rag"
                claim_labels = claims.get(item_id, []) if arm == selected_arm else []
                answer_label = answers.get(item_id) if arm == selected_arm else None
                signals = signals_for(record, claim_labels, answer_label)
                failures.append(
                    {
                        "item_id": item_id,
                        "arm": arm,
                        "question_type": record["question_type"],
                        "status": record["status"],
                        "gold_documents": record["gold_documents"],
                        "gold_documents_retrieved": record["gold_documents_retrieved"],
                        "gold_documents_cited": record["gold_documents_cited"],
                        "claim_count": record["claim_count"],
                        "system_verdict": record.get("verdict"),
                        "reviewed_claim_labels": claim_labels,
                        "reviewed_answer_completeness": answer_label["completeness"] if answer_label else None,
                        "reviewed_answer_verdict_reading": answer_label["verdict_reading"] if answer_label else None,
                        "signals": signals,
                    }
                )
    by_signal = Counter(signal for row in failures for signal in row["signals"])
    by_type = Counter(row["question_type"] for row in failures)
    by_batch_arm = Counter(
        f"{row['item_id'].split('-', 1)[0]}|{row['arm']}" for row in failures
    )
    return {
        "purpose": "Review queue; overlapping signals, not mutually exclusive root causes or a new accuracy score",
        "sources_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (*sources, claims_path, answers_path)},
        "summary": {
            "failed_arm_runs": len(failures),
            "with_reviewed_semantic_label": sum(
                bool(row["reviewed_claim_labels"] or row["reviewed_answer_completeness"])
                for row in failures
            ),
            "by_signal": dict(sorted(by_signal.items())),
            "by_question_type": dict(sorted(by_type.items())),
            "by_batch_arm": dict(sorted(by_batch_arm.items())),
        },
        "failures": failures,
    }


def main():
    verify_gold()
    report = build()
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
