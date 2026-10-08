"""Export actual development splitter inputs and generated claims for blind review.

No gold labels are copied from the old mechanically labelled experiment. All source
text stays in the ignored local runtime directory. Expanded rows include the pilot.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_runtime import atomic_json  # noqa: E402
from research_review import digest, group_splits, packet  # noqa: E402


def prepare(out):
    from sqlalchemy import select
    from agent.planner import subgoals
    from app.db import SessionLocal
    from app.models import Answer, AgentTask, AgentEvent, Chunk, DocumentVersion, User
    from app.security import require_chunk
    from semantic_evidence_eval import build_s3, build_multihop, S3_SNAPSHOT

    chunks = json.loads(S3_SNAPSHOT.read_text())
    metadata = {c["id"]: c for c in chunks}
    examples = build_s3() + build_multihop()
    rows, seen = [], set()
    with SessionLocal() as db:
        def enrich(p):
            p = dict(p)
            meta = metadata.get(p["chunk_id"])
            if meta:
                p.update(document_id=meta["document_id"], version_id=meta.get("version_id", ""))
            else:
                chunk = db.get(Chunk, p["chunk_id"])
                version = db.get(DocumentVersion, chunk.version_id)
                p.update(document_id=version.document_id, version_id=version.id)
            p.pop("relevant", None)  # Blind: never copy mechanical labels.
            return p
        for ex in examples:
            evidence = [enrich(p) for p in ex["evidence"]]
            families = sorted({p["document_id"] for p in evidence})
            for number, text in enumerate(subgoals(ex["question"]), 1):
                slot = {"id": f"s{number}", "text": text, "required": True, "origin": "agent.planner.subgoals"}
                base = {"corpus": ex["corpus"], "qid": ex["qid"], "question": ex["question"],
                        "slot": slot, "families": families, "variant": ex["variant"]}
                cov = {**base, "kind": "coverage", "evidence": evidence}
                cov["id"] = "coverage-" + digest(cov)[:24]
                rows.append(cov)
                for p in evidence:
                    rel = {**base, "kind": "relevance", "evidence": [p], "families": [p["document_id"]]}
                    rel.pop("variant")
                    key = digest(rel)
                    if key not in seen:
                        seen.add(key)
                        rows.append({**rel, "id": "relevance-" + key[:24]})
        # All already-used Chinese development answers and English batches 1/2.
        # Explicit tenant allow-list excludes unseen/holdout and arbitrary user data.
        development_questions = {ex["question"] for ex in examples}
        hard_questions = {r["goal"] for r in json.loads((ROOT / "fixtures/agent/hard_tasks.json").read_text())["tasks"]}
        development_questions |= hard_questions
        records = list(db.scalars(select(Answer).where(Answer.tenant_id.in_(["multihop", "xingqiao"]))))
        records += list(db.scalars(select(AgentTask).where(AgentTask.tenant_id.in_(["multihop", "xingqiao"]))))
        for record in records:
            user = db.get(User, record.user_id)
            payload = record.payload if isinstance(record, Answer) else record.result
            question = record.question if isinstance(record, Answer) else record.goal
            if question not in development_questions:
                continue
            citations = {c.get("id", c.get("chunk_id")): c for c in payload.get("citations", [])}
            if isinstance(record, AgentTask):
                events = db.scalars(select(AgentEvent).where(AgentEvent.task_id == record.id,
                    AgentEvent.event_type == "generation_context").order_by(AgentEvent.sequence)).all()
                context_refs = list(dict.fromkeys(ref for event in events for ref in event.evidence_chunk_ids))
            else:
                # Legacy QA answers have no actual-context snapshot. Their final
                # citations remain valid claim inputs, but cannot represent all
                # evidence shown to generation for a coverage example.
                context_refs = payload.get("trace", {}).get("generation_context_chunk_ids", [])
            actual_evidence = []
            for cid in context_refs:
                try:
                    chunk, version, doc = require_chunk(db, user, cid, active_only=False)
                except Exception:
                    continue
                actual_evidence.append({"chunk_id": chunk.id, "document_id": doc.id,
                                        "version_id": version.id, "title": doc.title, "text": chunk.text})
            if question in hard_questions and actual_evidence:
                # Add already-spent Hard source families, including version chains,
                # as actual generation inputs. These are not new formal test labels.
                for number, text in enumerate(subgoals(question), 1):
                    row = {"kind": "coverage", "corpus": "s3", "qid": digest(question), "question": question,
                           "variant": "actual_generation_context", "slot": {"id": f"s{number}", "text": text,
                                  "required": True, "origin": "agent.planner.subgoals"},
                           "evidence": actual_evidence, "families": sorted({p["document_id"] for p in actual_evidence})}
                    key = digest(row)
                    if key not in seen:
                        seen.add(key)
                        rows.append({**row, "id": "coverage-" + key[:24]})
            for number, claim in enumerate(payload.get("claims", []), 1):
                evidence = []
                for ref in dict.fromkeys(claim.get("evidence_ids", claim.get("citation_ids", claim.get("evidence_refs", [])))):
                    citation = citations.get(ref)
                    if not citation or not citation.get("chunk_id"):
                        continue
                    try:
                        chunk, version, doc = require_chunk(db, user, citation["chunk_id"], active_only=False)
                    except Exception:
                        continue
                    evidence.append({"chunk_id": chunk.id, "document_id": doc.id, "version_id": version.id,
                                     "title": doc.title, "text": chunk.text})
                if not evidence:
                    continue
                row = {"id": f"claim-{record.id}-{number}", "kind": "claim", "corpus": "mh" if record.tenant_id == "multihop" else "s3",
                       "qid": digest(question), "question": question, "generated_run_id": record.id,
                       "slot": {"id": f"claim{number}", "text": claim["text"], "purpose": "claim",
                                "allowed_refs": [p["chunk_id"] for p in evidence]}, "evidence": evidence,
                       "families": sorted({p["document_id"] for p in evidence})}
                rows.append(row)
    group_splits(rows)
    rows.sort(key=lambda r: (r["kind"], r["corpus"], digest(r["id"])))
    # All available inputs are retained. Pilot balances corpora without splitting
    # evidence variants; it is explicitly development, never the final test.
    pilot = []
    for kind, target in (("relevance", 100), ("coverage", 80), ("claim", 80)):
        pools = [[r for r in rows if r["kind"] == kind and r["corpus"] == corpus] for corpus in ("s3", "mh")]
        picked = []
        while len(picked) < target and any(pools):
            for pool in pools:
                if pool and len(picked) < target:
                    picked.append(pool.pop(0))
        pilot.extend(picked)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in (("expanded", rows), ("pilot", pilot)):
        target = out / f"{name}.json"
        if target.exists():
            raise ValueError("review packets are immutable; choose a new directory")
        atomic_json(target, packet(items, name + " actual-input component review"))
    counts = Counter(r["kind"] for r in rows)
    summary = {"version": "semantic-slot-review-v2", "independent_human_labels": 0,
               "path": str(out), "source_sha256": digest(rows), "expanded_counts": dict(counts),
               "pilot_counts": dict(Counter(r["kind"] for r in pilot)),
               "minimum_sample_ready": all(counts[k] >= n for k, n in (("relevance", 300), ("coverage", 240), ("claim", 120))),
               "strata": dict(Counter(f'{r["kind"]}:{r["corpus"]}:{r["split"]}' for r in rows)),
               "families": dict(Counter(r["family_id"] for r in rows)),
               "activation": "blocked_pending_independent_review_and_component_gates"}
    atomic_json(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".runtime/semantic-slot-v2/review")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir), ensure_ascii=False, indent=2))
