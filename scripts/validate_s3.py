"""Audit the frozen S3 gold: sources exist, facts are in the text, and ACL expectations hold."""

import json
from pathlib import Path
from statistics import median

from app.evaluation import (
    allowed_sources,
    fact_present,
    identities,
    prepare_snapshot,
    read_manifest,
    verify_sources,
)

DATASET = "fixtures/s3"

manifest = read_manifest(DATASET)
verify_sources(manifest)
questions = json.loads(Path(DATASET, "questions.json").read_text())
split = json.loads(Path(DATASET, "split.json").read_text())
sources = {row["id"]: row for row in manifest}
users = identities()
snapshot, chunks, blocks = prepare_snapshot(DATASET)
text_of = {key: "\n".join(b["text"] for b in value) for key, value in blocks.items()}

report = {"snapshot": snapshot, "documents": len(manifest), "questions": len(questions), "checks": []}
for question in questions:
    user = users[question["user"]]
    readable = {row["id"] for row in allowed_sources(manifest, user)}
    expected = question.get("source_ids", [])
    hidden = question.get("hidden_source_ids", [])
    assert all(key in sources for key in expected + hidden), question["id"]
    assert set(expected) <= readable, f"{question['id']} gold is not readable by {user.id}"
    assert not set(hidden) & readable, f"{question['id']} hidden source is readable by {user.id}"
    if question["kind"] == "unanswerable":
        assert not expected, question["id"]
    if question["kind"] == "permission":
        assert hidden and not expected, question["id"]
    combined = "\n".join(text_of[key] for key in expected)
    missing = [fact for fact in question.get("facts", []) if not fact_present(fact, combined)]
    assert not missing, f"{question['id']} facts not found in its own sources: {missing}"
    if question["kind"] == "permission":
        withheld = "\n".join(text_of[key] for key in hidden)
        assert question.get("hidden_facts"), "Permission cases must have an answer in a hidden source"
        assert all(fact_present(fact, withheld) for fact in question["hidden_facts"]), (
            f"{question['id']} is not a real permission case: the answer is absent from the hidden source"
        )
    report["checks"].append(
        {
            "id": question["id"],
            "kind": question["kind"],
            "expected_status": question["expected"],
            "gold_documents": expected,
            "hidden_documents": hidden,
            "facts_located": len(question.get("facts", [])),
            "readable_documents": len(readable),
            "source_locations": [
                {
                    "document_id": key,
                    "version_sha256": sources[key]["sha256"],
                    "locator": block["locator"],
                    "text": block["text"],
                }
                for key in expected
                for block in blocks[key]
                if any(fact_present(fact, block["text"]) for fact in question.get("facts", []))
            ],
        }
    )

families = {q["family"] for q in questions if q["split"] == "development"} & {
    q["family"] for q in questions if q["split"] == "holdout"
}
assert not families, f"Family leaks across splits: {families}"
# Check the source families as well as question labels; renamed questions cannot hide leakage.
development_sources = {
    k for q in questions if q["split"] == "development" for k in q.get("source_ids", [])
}
holdout_sources = {k for q in questions if q["split"] == "holdout" for k in q.get("source_ids", [])}
assert not (
    {sources[k]["family"] for k in development_sources} & {sources[k]["family"] for k in holdout_sources}
)
assert len({q["id"] for q in questions}) == len(questions)
assert len({s["id"] for s in manifest}) == len(manifest)
report["corpus"] = {
    "characters": sum(map(len, text_of.values())),
    "median_document_characters": median(map(len, text_of.values())),
    "min_document_characters": min(map(len, text_of.values())),
    "max_document_characters": max(map(len, text_of.values())),
    "chunks": len(chunks),
    "documents_over_700_characters": sum(len(v) > 700 for v in text_of.values()),
    "authoring": "synthetic; editorial cases and five genres, shared procedural language remains",
}
report["split"] = {
    "development": sum(q["split"] == "development" for q in questions),
    "holdout": sum(q["split"] == "holdout" for q in questions),
    "shared_families": 0,
}
report["review_state"] = split["review_state"]
report["gold_review"] = {
    "machine_audited": len(questions),
    "independent_human_review": 0,
    "note": "Facts were located in their own source text by program; that is provenance, not human review.",
}
Path("artifacts/s3-validation.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
print(
    f"S3 gold audited: {len(questions)} questions, {len(manifest)} documents, snapshot {snapshot[:16]}"
)
print("Independent human review is still outstanding and recorded as 0.")
