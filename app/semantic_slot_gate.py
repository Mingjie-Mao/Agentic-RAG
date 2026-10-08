"""Runtime semantic control remains off unless reviewed component gates are bound."""
import json
from pathlib import Path
import re

from app.config import settings
from app.semantic_evidence import SlotEvidenceEvaluator, Thresholds


def controlled_evaluator(db, user, question, *, historical):
    from scripts.paired_semantic_replay import require_gate
    from scripts.research_review import digest
    from app.security import require_chunk

    cfg = settings()
    if not cfg.semantic_slot_gate_path or not cfg.semantic_slot_calibration_path:
        raise ValueError("P2 reviewed component gate and frozen calibration required for semantic control")
    gate = require_gate(Path(cfg.semantic_slot_gate_path), "semantic-slot-v2-gate")
    from scripts.run_unseen_benchmark import configuration
    if gate["model_signature"] != configuration()["models"]:
        raise ValueError("runtime/model fingerprint differs from passed component gate")
    calibration = json.loads(Path(cfg.semantic_slot_calibration_path).read_text())
    if gate["calibration_sha256"] != digest(calibration):
        raise ValueError("runtime calibration differs from passed component gate")
    corpus = "s3" if re.search(r"[\u4e00-\u9fff]", question) else "mh"
    cut = calibration["thresholds"][corpus]
    def authorize(p):
        chunk, version, doc = require_chunk(db, user, p["chunk_id"], active_only=not historical())
        return chunk.text == p["text"] and version.id == p.get("version_id") and doc.id == p.get("document_id")
    return SlotEvidenceEvaluator(authorize=authorize, thresholds=Thresholds(
        keep=cut["keep"], high=cut["high"], version="semantic-slot-v2"), model_fingerprint=gate["model_signature"])
