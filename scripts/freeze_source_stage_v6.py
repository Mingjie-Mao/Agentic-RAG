"""Freeze fresh, gold-stratified retrieval challenges before generating answers.

This is a diagnostic challenge set, not a population sample: raw public gold is used
to select retrieval stages. No answer correctness or model output enters selection.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import select

from app.clients import Models, Search
from app.config import settings
from app.db import SessionLocal
from app.models import AgentTask, Answer, User
from app.retrieval import retrieve_authorized
from build_multihop_subset import FILES, fetch, digest
from freeze_source_facts_external import near
from multihop_retrieval_eval import document_id, fact_delivered

OUT = Path('fixtures/source_contract/external-stage-v6.json')
TARGETS = ('candidate_missing', 'quota_blocked', 'fact_proxy_reached')
PER_STAGE = 4


def stage(gold, candidates, admitted, matched_facts, total_facts):
    if gold - candidates:
        return 'candidate_missing'
    if gold - admitted:
        return 'quota_blocked'
    if total_facts and matched_facts == total_facts:
        return 'fact_proxy_reached'
    return None


def used_queries(data, db):
    used = set()
    for path in [*Path('fixtures/multihop').glob('subset*.json'),
                 *Path('fixtures/source_contract').glob('external*.json'),
                 Path('fixtures/verdict/external-oracle-r4-8.json')]:
        used.update(row['query_sha256'] for row in json.loads(path.read_text())['items'])
    used.update(digest(goal) for goal in db.scalars(
        select(AgentTask.goal).where(AgentTask.user_id == 'mh-eval')))
    used.update(digest(question) for question in db.scalars(
        select(Answer.question).where(Answer.user_id == 'mh-eval')))
    return used, [data[key] for key in sorted(used) if key in data]


def main():
    if OUT.exists():
        raise SystemExit('Frozen stage manifest exists; refusing overwrite')
    data = {digest(row['query']): row for row in fetch('MultiHopRAG.json', FILES['MultiHopRAG.json'], False)}
    titles = {row['title'] for row in fetch('corpus.json', FILES['corpus.json'], False)}
    cfg = settings().model_copy(update={
        'retrieval_mode': 'hybrid', 'source_focus_queries': False,
        'source_facet_queries': False, 'source_facet_document_queries': False,
        'article_first_lanes': False, 'source_query_plan': False,
        'passage_rerank': False,
    })
    chosen = {name: [] for name in TARGETS}
    screened = {'previous_gold_combination': 0, 'near_duplicate': 0,
                'missing_corpus_title': 0, 'other_retrieval_stage': 0}
    scanned = 0
    models, search = Models(), Search()
    with SessionLocal() as db:
        user = db.get(User, 'mh-eval')
        if not user:
            raise SystemExit('External corpus is not ingested')
        used, previous = used_queries(data, db)
        previous_signatures = {tuple(sorted(e['title'] for e in row.get('evidence_list', [])))
                               for row in previous}
        for key in sorted(data):
            if all(len(items) == PER_STAGE for items in chosen.values()):
                break
            if key in used:
                continue
            row = data[key]
            evidence = row.get('evidence_list', [])
            if not evidence or not all(e['title'] in titles for e in evidence):
                screened['missing_corpus_title'] += 1
                continue
            signature = tuple(sorted(e['title'] for e in evidence))
            if signature in previous_signatures:
                screened['previous_gold_combination'] += 1
                continue
            comparison = previous + [data[item['query_sha256']]
                                     for items in chosen.values() for item in items]
            similarity = max((near(row['query'], old['query']) for old in comparison), default=0)
            if similarity >= 0.80:
                screened['near_duplicate'] += 1
                continue
            result = retrieve_authorized(db, user, row['query'], cfg=cfg,
                                         models=models, search=search, top_k=6)
            scanned += 1
            gold = {document_id(e['title']) for e in evidence}
            candidate_ids = {c['document_id'] for c in result.candidates}
            admitted_ids = {e['document_id'] for e in result.evidence}
            by_document = {}
            for found in result.evidence:
                by_document.setdefault(found['document_id'], []).append(found['text'])
            matched = sum(fact_delivered(e['fact'], by_document.get(document_id(e['title']), []))
                          for e in evidence)
            label = stage(gold, candidate_ids, admitted_ids, matched, len(evidence))
            if label is None or len(chosen[label]) == PER_STAGE:
                screened['other_retrieval_stage'] += 1
                continue
            chosen[label].append({
                'id': f'SV6-{key[:12]}', 'question_type': row['question_type'],
                'target_stage': label, 'query_sha256': key, 'answer_sha256': digest(row['answer']),
                'gold_document_count': len(gold), 'candidate_gold_document_count': len(gold & candidate_ids),
                'admitted_gold_document_count': len(gold & admitted_ids),
                'gold_fact_count': len(evidence), 'fact_proxy_reached': matched,
                'max_similarity': round(similarity, 4),
            })
            previous_signatures.add(signature)
            print(f"selected {label} {len(chosen[label])}/{PER_STAGE} scanned={scanned}", flush=True)
    if any(len(items) != PER_STAGE for items in chosen.values()):
        raise SystemExit(f'Insufficient fresh cases after {scanned} retrievals: '
                         f'{ {key: len(value) for key, value in chosen.items()} }')
    output = {
        'name': 'external-stage-v6', 'source_files': FILES,
        'scope': ('Gold-stratified diagnostic challenge, not random held-out accuracy. '
                  'No answer model output used for selection; raw public gold may contain optional articles. '
                  'No independent human review.'),
        'selection': ('Unused query hashes and gold combinations; near similarity <0.80. '
                      'Four cases each: gold missing in retrieval candidates, gold in candidates '
                      'but not admitted, all gold docs and fact 4-gram proxy admitted.'),
        'used_question_hashes': sorted(used), 'screened_out_counts': screened,
        'retrievals_scanned': scanned, 'retrieval_top_k': 6,
        'retrieval_code_sha256': hashlib.sha256(Path('app/retrieval.py').read_bytes()).hexdigest(),
        'selection_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'stages': list(TARGETS), 'items': [item for label in TARGETS for item in chosen[label]],
        'scoring': 'Literal v3 answer score separately; document and fact coverage are proxies.',
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'counts': {key: len(value) for key, value in chosen.items()},
                      'retrievals_scanned': scanned, 'screened': screened}))


if __name__ == '__main__':
    main()
