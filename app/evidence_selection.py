"""Sentence-level evidence selection (the CRAG / FILCO idea), off the production path.

A generator given the gold passages and nothing else answers the external yes/no set
far better than one given the same gold passages plus the retrieval around them, and
far better than one given retrieval alone. Whole chunks carry both problems: the right
sentence is often not in them, and when it is, it arrives with a paragraph of material
that pulls the answer elsewhere.

This keeps retrieval as it is and changes what reaches the model: the chunks of each
lane are split into sentences, every sentence is scored against the question (or
against the part of the question about that lane's publication), and only the best few
per publication are kept. Every selected sentence is a verbatim substring of an
authorized chunk, so evidence items keep their chunk id and citation validation works
unchanged. Nothing here grants access: it only narrows chunks retrieval already
returned for this user.
"""

import re

# A boundary is end punctuation followed by whitespace (so "$14.99" stays whole), not
# after an initial or a title ("U.S. District", "Mr. Smith"), or any Chinese full stop.
_BOUNDARY = re.compile(
    r"(?<!\b[A-Z]\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bvs\.)"
    r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)]))\s+"
    r"|(?<=[。！？])|\n+"
)


def split_sentences(text: str) -> list[str]:
    """Sentences as verbatim substrings of `text`, fragments under four words dropped."""
    sentences = []
    for piece in _BOUNDARY.split(text):
        piece = (piece or "").strip()
        if len(piece.split()) >= 4 or (piece and not piece.isascii() and len(piece) >= 8):
            sentences.append(piece)
    return sentences


def header_of(document) -> str:
    metadata = document.metadata_json or {}
    published = str(metadata.get("published_at") or "")[:10]
    return " · ".join(part for part in (document.title, metadata.get("source"), published) if part)


def select_sentences(
    question: str,
    lanes: list[dict],
    *,
    score,
    per_lane: int = 2,
    global_extra: int = 2,
    max_total: int = 8,
    neighbors: int = 0,
):
    """Pick the evidence sentences for `question`.

    `lanes` is a list of {"name", "query", "chunks": [{chunk_id, document_id, version_id,
    title, text, locator, metadata, header}]}, named lanes first and the global lane last.
    `score(query, texts) -> list[float]` is the relevance scorer (a cross-encoder in
    practice). Returns evidence items in the shape retrieval produces, one per chunk that
    contributed a sentence, with the chosen sentences joined in their original order.
    """
    chosen: dict[str, dict] = {}
    order: list[str] = []
    seen_sentences: set[str] = set()

    lane_of: dict[str, str] = {}

    def take(lane, count):
        candidates = []
        for chunk in lane["chunks"]:
            sentences = split_sentences(chunk["text"])
            for index, sentence in enumerate(sentences):
                candidates.append((chunk, sentences, index, sentence))
        if not candidates or count <= 0:
            return
        scores = score(lane["query"], [f"{c[0]['header']}\n{c[3]}" for c in candidates])
        ranked = sorted(zip(scores, range(len(candidates)), candidates), key=lambda row: (-row[0], row[1]))
        taken = 0
        for _score, _position, (chunk, sentences, index, sentence) in ranked:
            if taken >= count or sum(len(v["indices"]) for v in chosen.values()) >= max_total:
                break
            if sentence in seen_sentences:
                continue
            seen_sentences.add(sentence)
            entry = chosen.setdefault(chunk["chunk_id"], {"chunk": chunk, "sentences": sentences, "indices": set()})
            if chunk["chunk_id"] not in order:
                order.append(chunk["chunk_id"])
                lane_of[chunk["chunk_id"]] = lane["name"]
            for offset in range(-neighbors, neighbors + 1):
                if 0 <= index + offset < len(sentences):
                    entry["indices"].add(index + offset)
            taken += 1

    named, rest = lanes[:-1], lanes[-1:]
    for lane in named:
        take(lane, per_lane)
    for lane in rest:
        take(lane, global_extra if named else max_total)

    evidence = []
    for number, chunk_id in enumerate(order, 1):
        entry = chosen[chunk_id]
        chunk = entry["chunk"]
        text = " ".join(entry["sentences"][i] for i in sorted(entry["indices"]))
        evidence.append(
            {
                "id": f"E{number}",
                "chunk_id": chunk_id,
                "document_id": chunk["document_id"],
                "version_id": chunk["version_id"],
                "title": chunk["title"],
                "text": text,
                "locator": chunk["locator"],
                "metadata": chunk["metadata"],
                "selected_sentences": len(entry["indices"]),
                "lane": lane_of[chunk_id],
            }
        )
    return evidence


def cross_encoder_scorer():
    """Score with the project's cross-encoder (bge-reranker-v2-m3), batched."""
    from app.rerank import _model
    from app.config import settings

    def score(query: str, texts: list[str]) -> list[float]:
        tokenizer, model, device, torch = _model()
        cfg = settings()
        values = []
        for start in range(0, len(texts), 16):
            batch = texts[start : start + 16]
            encoded = tokenizer(
                [query] * len(batch), batch, padding=True, truncation=True,
                max_length=cfg.rerank_max_tokens, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                values.extend(model(**encoded).logits.view(-1).float().tolist())
        return values

    return score


def build_lanes(db, user, question, *, cfg, models, search, top_k=8, chunks_per_lane=6, clause_queries=False):
    """Candidate chunks per retrieval lane, re-authorized, with the query each lane is
    scored against: the whole question, or the part of it about that publication."""
    from app.retrieval import retrieve_authorized, route_sources
    from app.security import readable_documents, require_chunk

    found = retrieve_authorized(
        db, user, question, cfg=cfg.model_copy(update={"passage_rerank": True}),
        models=models, search=search, top_k=top_k,
    )
    clauses = {}
    if clause_queries:
        for group in route_sources(question, readable_documents(db, user)):
            clauses[f"source:{group['mention']}"] = group.get("clause") or ""
    lanes: dict[str, list] = {}
    for candidate in found.candidates:
        lane = candidate.get("lane", "global")
        if len(lanes.setdefault(lane, [])) >= chunks_per_lane:
            continue
        chunk, version, document = require_chunk(db, user, candidate["chunk_id"], active_only=True)
        lanes[lane].append(
            {
                "chunk_id": chunk.id,
                "document_id": document.id,
                "version_id": version.id,
                "title": document.title,
                "text": chunk.text,
                "locator": chunk.locator,
                "metadata": document.metadata_json,
                "header": header_of(document),
            }
        )
    named = [name for name in lanes if name != "global"]
    ordered = [
        {"name": name, "query": (f"{clauses[name]} {question}" if clauses.get(name) else question), "chunks": lanes[name]}
        for name in named
    ]
    ordered.append({"name": "global", "query": question, "chunks": lanes.get("global", [])})
    return ordered, found.evidence
