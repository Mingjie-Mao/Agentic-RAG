"""Pinned, single-source EnterpriseRAG-Bench subset with all gold docs + distractors."""

from io import BytesIO
import hashlib
import os
import json
from pathlib import Path
import re
from zipfile import ZipFile

import httpx

ROOT = Path("fixtures/public_subset")
ROOT.mkdir(parents=True, exist_ok=True)
COMMIT = "d36685e273713975ee20299bbf1ab64165575b3c"
BASE = "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0/"
SLICES = [f"github_slice_{n:04}.zip" for n in range(1, int(os.environ.get("RAG_BENCH_SLICES", "2")) + 1)]
TARGET = int(os.environ.get("RAG_BENCH_QUESTIONS", "20"))
documents = {}
inputs = []
with httpx.Client(timeout=120, follow_redirects=True, trust_env=False) as client:
    qbytes = client.get(
        f"https://raw.githubusercontent.com/onyx-dot-app/EnterpriseRAG-Bench/{COMMIT}/questions.jsonl"
    )
    qbytes.raise_for_status()
    questions = [json.loads(line) for line in qbytes.text.splitlines() if line.strip()]
    for name in SLICES:
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
scale = TARGET / 20
for category, limit in [
    ("basic", round(8 * scale)),
    ("semantic", round(6 * scale)),
    ("intra_document_reasoning", round(2 * scale)),
    ("constrained", round(2 * scale)),
    ("conflicting_info", round(2 * scale)),
]:
    selected.extend(
        sorted(
            [q for q in eligible if q["question_type"] == category],
            key=lambda q: hashlib.sha256(q["question_id"].encode()).hexdigest(),
        )[:limit]
    )
for q in sorted(eligible, key=lambda q: hashlib.sha256(q["question_id"].encode()).hexdigest()):
    if len(selected) >= TARGET:
        break
    if q not in selected:
        selected.append(q)
selected = selected[:TARGET]
print(f"eligible={len(eligible)} selected={len(selected)} slices={len(SLICES)}")
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
            "limitations": f"one source only (github); {len(selected)} questions is every eligible question in the two published github slices, not the full benchmark; {len(distractors)} distractors; no official LLM-judge score claimed",
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
print(
    f"Pinned public subset: {len(selected)} questions, {len(gold)} gold documents, {len(distractors)} distractors"
)
