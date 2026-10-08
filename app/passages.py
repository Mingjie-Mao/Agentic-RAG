"""Budgeted, ACL-checked small-to-big generation context; no model or gold input."""
from sqlalchemy import select

from agent.planner import evidence_coverage, subgoals, keywords, _grams, _EN_STOP
from app.chunking import token_count
from app.models import Chunk
from app.security import require_chunk


def prepare_passages(db, user, question, evidence, cfg, *, historical=False,
                     authorize=require_chunk):
    items = subgoals(question)
    pool = {row['chunk_id']: dict(row) for row in evidence}
    initial = evidence_coverage(items, list(pool.values()), question)
    extra = []
    scan_count = 0
    scanned_versions = set()
    version_count = len({row['version_id'] for row in evidence})
    per_version_limit = 1 if version_count > 1 else cfg.passage_window_extra
    added_by_version = {}
    wanted = _grams(keywords(question)) - _EN_STOP
    # Neighbours never cross a version boundary; each has its own real citation ID.
    for row in evidence:
        if len(extra) >= cfg.passage_window_extra:
            break
        if added_by_version.get(row['version_id'], 0) >= per_version_limit:
            continue
        chunk, version, document = authorize(db, user, row['chunk_id'], active_only=not historical)
        neighbours = db.scalars(select(Chunk).where(
            Chunk.version_id == version.id,
            Chunk.ordinal.in_([chunk.ordinal - 1, chunk.ordinal + 1]),
        ).order_by(Chunk.ordinal)).all()
        targeted = []
        if version.id not in scanned_versions:
            scanned_versions.add(version.id)
            scanned = db.scalars(select(Chunk).where(Chunk.version_id == version.id)
                                 .order_by(Chunk.ordinal).limit(cfg.passage_scan_limit)).all()
            scan_count += len(scanned)
            targeted = sorted((part for part in scanned if part.id not in pool
                               and wanted & _grams(part.text)),
                              key=lambda part: -len(wanted & _grams(part.text)))[:1]
        for neighbour in targeted + neighbours:
            if (neighbour.id in pool or len(extra) >= cfg.passage_window_extra
                    or added_by_version.get(version.id, 0) >= per_version_limit):
                continue
            _, checked_version, checked_document = authorize(
                db, user, neighbour.id, active_only=not historical)
            candidate = dict(row, chunk_id=neighbour.id, text=neighbour.text,
                             locator=neighbour.locator, version_id=checked_version.id,
                             document_id=checked_document.id)
            pool[neighbour.id] = candidate
            extra.append(neighbour.id)
            added_by_version[version.id] = added_by_version.get(version.id, 0) + 1
    rows = list(pool.values())
    coverage = evidence_coverage(items, rows, question)
    order = []
    # Reserve one candidate per source/version and one per uncovered subgoal before
    # filling remaining slots. These are lexical candidates, not entailment proofs.
    groups = {}
    def priority(row):
        return (-sum(row['chunk_id'] in refs for refs in coverage.values()),
                -len(wanted & _grams(row['text'])))
    for row in rows:
        groups.setdefault((row['document_id'], row['version_id']), []).append(row)
    for group in groups.values():
        scored = sorted(group, key=priority)
        order.append(scored[0])
    for item in items:
        order.extend(sorted((row for row in rows if row['chunk_id'] in coverage[item]), key=priority))
    order.extend(sorted(rows, key=priority))
    chosen, used, tokens = [], set(), 0
    for row in order:
        cost = token_count(row['text']) + token_count(row['title']) + 100
        if row['chunk_id'] in used or len(chosen) >= 8 or tokens + cost > cfg.context_token_budget:
            continue
        authorize(db, user, row['chunk_id'], active_only=not historical)
        chosen.append(dict(row, id=f'E{len(chosen) + 1}'))
        used.add(row['chunk_id'])
        tokens += cost
    final = evidence_coverage(items, chosen, question)
    return chosen, {
        'method': 'subgoal_sentence_window_v1', 'coverage_type': 'lexical_candidate_only',
        'scanned_chunks': scan_count, 'scan_limit_per_version': cfg.passage_scan_limit,
        'before': initial, 'after': final, 'context_tokens': tokens,
        'added_chunk_ids': [key for key in extra if key in used],
        'uncovered_items': [item for item in items if not final[item]],
    }
