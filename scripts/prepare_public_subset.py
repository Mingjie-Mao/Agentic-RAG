"""Pinned, single-source EnterpriseRAG-Bench subset with all gold docs + distractors."""

from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

import httpx

ROOT = Path("fixtures/public_subset")
ROOT.mkdir(parents=True, exist_ok=True)
COMMIT = "d36685e273713975ee20299bbf1ab64165575b3c"
BASE = "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0/"
documents = {}
inputs = []
with httpx.Client(timeout=120, follow_redirects=True, trust_env=False) as client:
    qbytes = client.get(
        f"https://raw.githubusercontent.com/onyx-dot-app/EnterpriseRAG-Bench/{COMMIT}/questions.jsonl"
    )
    qbytes.raise_for_status()
    questions = [json.loads(line) for line in qbytes.text.splitlines() if line.strip()]
    for name in ["github_slice_0001.zip", "github_slice_0002.zip"]:
        response = client.get(BASE + name)
        response.raise_for_status()
        inputs.append({"url": BASE + name, "sha256": hashlib.sha256(response.content).hexdigest()})
        with ZipFile(BytesIO(response.content)) as archive:
            for info in archive.infolist():
                match = re.search(r"dsid_[a-f0-9]{32}", info.filename)
                if match and not info.is_dir():
                    documents[match.group()] = {
                        "text": archive.read(info).decode("utf-8-sig"),
                        "upstream_filename": info.filename,
                    }
    license_response = client.get(
        f"https://raw.githubusercontent.com/onyx-dot-app/EnterpriseRAG-Bench/{COMMIT}/LICENSE"
    )
    license_response.raise_for_status()
    (ROOT / "LICENSE.upstream").write_bytes(license_response.content)
eligible = [
    q
    for q in questions
    if q["expected_doc_ids"]
    and set(q["expected_doc_ids"]) <= documents.keys()
    and set(q["source_types"]) == {"github"}
]
selected = []
for category, limit in [
    ("basic", 8),
    ("semantic", 6),
    ("intra_document_reasoning", 2),
    ("constrained", 2),
    ("conflicting_info", 2),
]:
    selected.extend(
        sorted(
            [q for q in eligible if q["question_type"] == category],
            key=lambda q: hashlib.sha256(q["question_id"].encode()).hexdigest(),
        )[:limit]
    )
for q in sorted(eligible, key=lambda q: hashlib.sha256(q["question_id"].encode()).hexdigest()):
    if len(selected) >= 20:
        break
    if q not in selected:
        selected.append(q)
assert len(selected) == 20
gold = {key for q in selected for key in q["expected_doc_ids"]}
distractors = sorted(set(documents) - gold, key=lambda key: hashlib.sha256(key.encode()).hexdigest())[
    :100
]
manifest = []
(ROOT / "documents").mkdir(exist_ok=True)
for key in sorted(gold | set(distractors)):
    item = documents[key]
    path = ROOT / "documents" / (key + ".md")
    path.write_text(item["text"])
    manifest.append(
        {
            "id": key,
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "title": item["text"].splitlines()[0][:200],
            "tenant_id": "public-benchmark",
            "groups": [],
            "tenant_public": True,
            "upstream_filename": item["upstream_filename"],
        }
    )
(ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
(ROOT / "questions.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
(ROOT / "selection.json").write_text(
    json.dumps(
        {
            "name": "EnterpriseRAG-Bench GitHub-only fixed subset",
            "question_commit": COMMIT,
            "questions_sha256": hashlib.sha256(qbytes.content).hexdigest(),
            "release": "v1.0.0",
            "inputs": inputs,
            "question_ids": [q["question_id"] for q in selected],
            "gold_documents": sorted(gold),
            "distractors": distractors,
            "selection": "fixed SHA-256 order within categories; only GitHub questions whose complete gold is available",
            "limitations": "one source only; 20 questions + 100 distractors, not the full benchmark; no official LLM-judge score claimed",
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
print(
    f"Pinned public subset: {len(selected)} questions, {len(gold)} gold documents, {len(distractors)} distractors"
)
