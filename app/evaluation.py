"""Reproducible offline evaluation. Gold is used by scoring, never by retrieval/generation."""

import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
from types import SimpleNamespace
import unicodedata
import uuid

from app.chunking import chunk_blocks, token_count
from app.clients import Models, Search
from app.config import settings
from app.parsing import parse_document
from app.qa import validate_claims
from app.security import can_read

MEDIA = {
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_manifest(directory):
    return json.loads((Path(directory) / "manifest.json").read_text())


def verify_sources(manifest):
    for source in manifest:
        assert hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() == source["sha256"], source[
            "id"
        ]


def prepare_snapshot(directory):
    directory = Path(directory)
    manifest = read_manifest(directory)
    verify_sources(manifest)
    cfg = settings()
    pipeline = {
        "parser": cfg.parser_version,
        "chunking": cfg.chunk_strategy,
        "size": cfg.chunk_chars,
        "overlap": cfg.chunk_overlap,
        "embedding_model": cfg.embed_model,
        "dimension": cfg.embed_dimension,
        "implementation": {
            name: hashlib.sha256(Path("app", name).read_bytes()).hexdigest()
            for name in ["parsing.py", "office_parsing.py", "pdf_parsing.py", "chunking.py"]
        },
    }
    snapshot = digest({"documents": manifest, "pipeline": pipeline})
    cache = Path(".runtime/evaluation") / snapshot
    cache.mkdir(parents=True, exist_ok=True)
    blocks_path = cache / "blocks.json"
    chunks_path = cache / "chunks.json"
    if chunks_path.exists():
        return snapshot, json.loads(chunks_path.read_text()), json.loads(blocks_path.read_text())
    chunks, all_blocks = [], {}
    for number, source in enumerate(manifest, 1):
        path = Path(source["path"])
        blocks = parse_document(path.read_bytes(), MEDIA[path.suffix], structured=True)
        all_blocks[source["id"]] = [{"text": b.text, "locator": b.locator} for b in blocks]
        for ordinal, piece in enumerate(
            chunk_blocks(blocks, cfg.chunk_chars, cfg.chunk_overlap, cfg.chunk_strategy)
        ):
            chunks.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{snapshot}:{source['id']}:{ordinal}")),
                    "document_id": source["id"],
                    "version_id": source["sha256"],
                    "text": piece.text,
                    "locator": piece.locator,
                    "title": source["title"],
                    "tenant_id": source["tenant_id"],
                    "metadata": {
                        "effective_from": source.get("effective_from"),
                        "effective_to": source.get("effective_to"),
                    },
                }
            )
        if number % 10 == 0:
            print(f"Parsed {number}/{len(manifest)} documents", flush=True)
    blocks_path.write_text(canonical(all_blocks))
    chunks_path.write_text(canonical(chunks))
    return snapshot, chunks, all_blocks


def prepare_index(directory):
    snapshot, chunks, blocks = prepare_snapshot(directory)
    search = Search("enterprise-rag-eval-" + snapshot[:16])
    search.ensure_index()
    marker = Path(".runtime/evaluation") / snapshot / "indexed.json"
    if marker.exists():
        count = search.request("GET", f"/{search.index}/_count")["count"]
        if count == len(chunks):
            return snapshot, search, chunks, blocks
    models = Models()
    for offset in range(0, len(chunks), 8):
        batch = chunks[offset : offset + 8]
        vectors = models.embed([c["text"] for c in batch])
        body = []
        for item, vector in zip(batch, vectors, strict=True):
            body.extend(
                [
                    canonical({"index": {"_index": search.index, "_id": item["id"]}}),
                    canonical(
                        {k: item[k] for k in ["tenant_id", "document_id", "version_id", "text", "title"]}
                        | {"embedding": vector}
                    ),
                ]
            )
        result = search.request(
            "POST",
            "/_bulk",
            content="\n".join(body) + "\n",
            headers={"Content-Type": "application/x-ndjson"},
        )
        assert not result.get("errors"), "Partial index failure; snapshot not ready"
        if offset % 80 == 0:
            print(f"Indexed {min(offset + 8, len(chunks))}/{len(chunks)} chunks", flush=True)
    search.request("POST", f"/{search.index}/_refresh")
    assert search.request("GET", f"/{search.index}/_count")["count"] == len(chunks)
    marker.write_text(canonical({"snapshot": snapshot, "chunks": len(chunks), "index": search.index}))
    return snapshot, search, chunks, blocks


def identities():
    users = json.loads(Path("fixtures/catalog.json").read_text())["users"]
    result = {u["id"]: SimpleNamespace(**u, active=True) for u in users}
    result["public-benchmark"] = SimpleNamespace(
        id="public-benchmark", tenant_id="public-benchmark", role="admin", groups=[], active=True
    )
    return result


def allowed_sources(manifest, user):
    return [
        s
        for s in manifest
        if can_read(
            user,
            SimpleNamespace(
                tenant_id=s["tenant_id"],
                owner_id=s.get("owner_id", ""),
                tenant_public=s["tenant_public"],
                read_groups=s["groups"],
                deleted=False,
            ),
        )
    ]


def normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def fact_present(fact, text):
    fact, text = normalized(fact), normalized(text)
    pattern = re.escape(fact)
    if fact and fact[0].isdigit():
        pattern = r"(?<!\d)" + pattern
    if fact and fact[-1].isdigit():
        pattern += r"(?!\d)"
    return bool(re.search(pattern, text))


def retrieval_scores(expected, ranked):
    ranked = list(dict.fromkeys(ranked))
    if not expected:
        return {"recall_at_5": None, "recall_at_10": None, "mrr": None, "ndcg_at_10": None}
    expected = set(expected)
    dcg = sum(1 / math.log2(i + 2) for i, key in enumerate(ranked[:10]) if key in expected)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(expected))))
    return {
        "recall_at_5": len(expected & set(ranked[:5])) / len(expected),
        "recall_at_10": len(expected & set(ranked[:10])) / len(expected),
        "mrr": next((1 / (i + 1) for i, key in enumerate(ranked) if key in expected), 0),
        "ndcg_at_10": dcg / ideal,
    }


def generate_answer(question, hits, chunk_map):
    cfg = settings()
    evidence = []
    budget = 0
    for hit in hits[: cfg.top_k]:
        chunk = chunk_map[hit["chunk_id"]]
        # Offline comparison holds evidence admission constant for both retrievers.
        # Production's dense score cutoff is a separate policy, not a BM25 analogue.
        cost = token_count(chunk["text"]) + token_count(chunk["title"]) + 100
        if budget + cost > cfg.context_token_budget:
            continue
        budget += cost
        evidence.append({**chunk, "id": f"E{len(evidence) + 1}", "chunk_id": chunk["id"]})
    if not evidence:
        return {
            "status": "insufficient_evidence",
            "claims": [],
            "citations": [],
            "usage": {},
            "context_tokens_estimate": 0,
        }
    generated, usage = Models().generate(question, evidence)
    claims, status = validate_claims(generated, evidence)
    if status == "answered" and usage.get("answer_status") == "conflict":
        status = "conflict"
    used = {key for claim in claims for key in claim["evidence_ids"]}
    citations = [
        {
            "chunk_id": e["chunk_id"],
            "document_id": e["document_id"],
            "version_id": e["version_id"],
            "locator": e["locator"],
            "text": e["text"],
        }
        for e in evidence
        if e["chunk_id"] in used
    ]
    return {
        "status": status,
        "claims": claims,
        "citations": citations,
        "usage": usage,
        "context_tokens_estimate": budget,
    }


def answer_scores(question, answer, chunk_map=None):
    text = "\n".join(c["text"] for c in answer["claims"])
    facts = question.get("facts", [])
    used = {c["document_id"] for c in answer["citations"]}
    expected = question.get("source_ids", [])
    return {
        "literal_fact_matches": sum(fact_present(f, text) for f in facts),
        "literal_fact_total": len(facts),
        "status_correct": answer["status"] == question.get("expected", "answered"),
        "citation_document_recall": len(set(expected) & used) / len(set(expected)) if expected else None,
        "citation_identity_valid": (
            all(
                c["chunk_id"] in chunk_map
                and all(
                    c[key] == chunk_map[c["chunk_id"]][key]
                    for key in ["document_id", "version_id", "text", "locator"]
                )
                for c in answer["citations"]
            )
            if chunk_map is not None
            else None
        ),
        "forbidden_marker_leaks": sum(
            marker in canonical(answer) for marker in question.get("forbidden", [])
        ),
        "semantic_correctness": None,
        "citation_entailment": None,
    }


def aggregate(rows):
    result = {"queries": len(rows), "errors": sum(bool(r.get("error")) for r in rows)}
    for key in ["recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]:
        values = [r["retrieval"][key] for r in rows if r.get("retrieval", {}).get(key) is not None]
        result[key] = {"mean": sum(values) / len(values) if values else None, "n": len(values)}
    answered = [r for r in rows if r.get("answer")]
    result["generated"] = len(answered)
    result["status_correct"] = {
        "count": sum(r["answer_metrics"]["status_correct"] for r in answered),
        "n": len(answered),
    }
    result["literal_fact_coverage"] = {
        "matches": sum(r["answer_metrics"]["literal_fact_matches"] for r in answered),
        "total": sum(r["answer_metrics"]["literal_fact_total"] for r in answered),
        "label": "literal normalized matching; not human correctness",
    }
    result["permission_violations"] = sum(r.get("permission_violations", 0) for r in rows)
    result["semantic_correctness"] = result["citation_entailment"] = (
        "not measured; requires human review/calibrated judge"
    )
    times = sorted(r["total_seconds"] for r in rows)
    result["latency_seconds"] = {
        "p50": median(times) if times else None,
        "p95": times[math.ceil(len(times) * 0.95) - 1] if times else None,
    }
    return result
