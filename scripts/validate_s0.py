"""Validate source provenance/ACL and emit version-bound gold locators, not chunk IDs."""

import hashlib
import json
from pathlib import Path

from app.parsing import parse_document

root = Path(__file__).resolve().parent.parent
catalog = json.loads((root / "fixtures/catalog.json").read_text())
manifest = json.loads((root / "fixtures/manifest.json").read_text())
questions = json.loads((root / "fixtures/questions.json").read_text())["questions"]
contracts = json.loads((root / "fixtures/contracts.json").read_text())
users = {u["id"]: u for u in catalog["users"]}
sources = {d["source_key"]: d for d in manifest}
assert len(manifest) == 30 and len(questions) == 20
assert len(contracts["engineering_scenarios"]) == 10
assert len({d["tenant_id"] for d in manifest}) == 2
for row in manifest:
    assert hashlib.sha256((root / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    assert row["provenance"] and row["version"]
gold = []
for question in questions:
    user = users[question["user"]]
    locations, all_text = [], ""
    for key in question["sources"]:
        source = sources[key]
        assert source["family"] not in contracts["reserved_future_holdout_families"]
        assert user["tenant_id"] == source["tenant_id"]
        assert (
            user["role"] == "admin"
            or source["tenant_public"]
            or set(user["groups"]) & set(source["groups"])
        )
        path = root / source["path"]
        pages = parse_document(
            path.read_bytes(), "application/pdf" if path.suffix == ".pdf" else "text/markdown"
        )
        for page in pages:
            all_text += page.text
            locator = dict(page.locator)
            if locator["kind"] == "markdown":
                locator.update(line_start=1, line_end=page.text.count("\n") + 1)
            locations.append(
                {
                    "source_key": key,
                    "path": source["path"],
                    "sha256": source["sha256"],
                    "version": source["version"],
                    "locator": locator,
                    "text": page.text,
                }
            )
    assert all(fact in all_text for fact in question["facts"]), question["id"]
    gold.append(
        {
            "question_id": question["id"],
            "user_id": question["user"],
            "expected": question["expected"],
            "source_locations": locations,
        }
    )
out = root / "artifacts"
out.mkdir(exist_ok=True)
result = {
    "stage": "S0",
    "documents": 30,
    "questions": 20,
    "engineering_scenarios": 10,
    "tenants": 2,
    "source_hash_acl_and_fact_checks": "passed",
    "gold_type": "development_only_source_level_not_frozen_test",
    "questions_with_locations": gold,
}
(out / "s0-validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print("S0: 30 documents / 20 questions / 10 scenarios; source hashes, facts and ACL passed")
