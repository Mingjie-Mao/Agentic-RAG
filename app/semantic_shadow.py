"""Calibrated claim/evidence scorer used for observation, never answer gating.

The cross-encoder is a relevance model, not an entailment proof. Its threshold was
selected on the frozen, model-reviewed v3 labels; supported recall was only 0.65, so
blocking an answer with it would hide many correct claims.
"""

from app.rerank import rerank


THRESHOLD = 2.44
WINDOW = 1200
STEP = 800


def windows(text: str) -> list[str]:
    if len(text) <= WINDOW:
        return [text]
    return [text[start : start + WINDOW] for start in range(0, len(text) - STEP, STEP)]


def score_claims(evidence: list[dict], claims: list[dict]) -> dict:
    by_id = {row["chunk_id"]: row["text"] for row in evidence}
    scores = []
    for claim in claims:
        cited = [by_id[key] for key in claim["evidence_ids"] if key in by_id]
        if not cited:
            scores.append(None)
            continue
        candidates = [{"text": part} for text in cited for part in windows(text)]
        ranked = rerank(claim["text"], candidates, window=len(candidates))
        scores.append(round(max(row["rerank_score"] for row in ranked), 4))
    return {
        "scorer": "bge-reranker-v2-m3-v3-shadow",
        "threshold": THRESHOLD,
        "claim_support_scores": scores,
        "mode": "shadow",
        "action": "recorded only; no answer is withheld on these scores",
        "calibration": "artifacts/semantic-scorer-calibration.json",
    }
