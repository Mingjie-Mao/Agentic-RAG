"""Frozen-manifest, retrieval-only comparison of source-clause queries."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.clients import Models, Search
from app.config import settings
from app.db import SessionLocal
from app.models import User
from app.retrieval import retrieve_authorized
from build_multihop_subset import FILES, fetch, digest
from multihop_retrieval_eval import document_id, fact_delivered

MANIFEST = Path('fixtures/source_contract/external-p0-p2-v1.json')
OUT = Path('artifacts/source-focus-retrieval-development.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--facet', action='store_true',
                        help='Compare baseline to explicit multi-facet source search')
    parser.add_argument('--rerank', action='store_true',
                        help='Compare baseline to local cross-encoder passage reranking')
    parser.add_argument('--facet-document', action='store_true',
                        help='Compare baseline to one distinct document per facet first')
    parser.add_argument('--article-first', action='store_true',
                        help='Compare baseline to document-diverse source lanes')
    parser.add_argument('--plan', action='store_true',
                        help='Compare baseline to bounded local-model search planning')
    parser.add_argument('--bm25', action='store_true',
                        help='Compare hybrid baseline with BM25-only retrieval')
    parser.add_argument('--document-quota-one', action='store_true',
                        help='Compare the default two chunks per document with one')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite source focus experiment')
    manifest = json.loads(args.manifest.read_text())
    data = {digest(row['query']): row for row in fetch('MultiHopRAG.json', FILES['MultiHopRAG.json'], False)}
    cfg = settings().model_copy(update={
        'retrieval_mode': 'hybrid', 'source_focus_queries': False,
        'source_facet_queries': False, 'source_facet_document_queries': False,
        'article_first_lanes': False, 'source_query_plan': False,
        'passage_rerank': False,
    })
    models, search = Models(), Search()
    rows = []
    with SessionLocal() as db:
        user = db.get(User, 'mh-eval')
        for item in manifest['items']:
            question = data[item['query_sha256']]
            if not question.get('evidence_list'):
                continue
            modes = ((('baseline', {}), ('bm25', {'retrieval_mode': 'bm25'}))
                     if args.bm25 else
                     (('baseline', {}), ('document_quota_one', {'document_quota': 1}))
                     if args.document_quota_one else
                     (('baseline', {}), ('source_query_plan', {'source_query_plan': True}))
                     if args.plan else
                     (('baseline', {}), ('article_first', {'article_first_lanes': True}))
                     if args.article_first else
                     (('baseline', {}), ('source_facet_document', {'source_facet_document_queries': True}))
                     if args.facet_document else
                     (('baseline', {}), ('source_facet', {'source_facet_queries': True}))
                     if args.facet else
                     (('baseline', {}), ('rerank', {'passage_rerank': True}))
                     if args.rerank else
                     (('baseline', {}), ('source_clause', {'source_clause_queries': True}),
                      ('source_focus', {'source_focus_queries': True})))
            for mode, changes in modes:
                started = time.monotonic()
                found = retrieve_authorized(
                    db, user, question['query'], cfg=cfg.model_copy(update=changes),
                    models=models, search=search, top_k=6)
                by_doc = {}
                for evidence in found.evidence:
                    by_doc.setdefault(evidence['document_id'], []).append(evidence['text'])
                gold = {document_id(row['title']) for row in question['evidence_list']}
                missing = gold - by_doc.keys()
                candidate_causes = {}
                for document_key in missing:
                    matches = [row for row in found.candidates if row['document_id'] == document_key]
                    candidate_causes[document_key] = (
                        sorted({row['excluded_because'] or 'candidate_not_admitted' for row in matches})
                        if matches else ['not_in_candidates'])
                rows.append({
                    'id': item['id'], 'type': item['question_type'],
                    'target_stage': item.get('target_stage'), 'mode': mode,
                    'all_gold_documents': gold <= by_doc.keys(),
                    'gold_documents_found': len(gold & by_doc.keys()),
                    'gold_document_count': len(gold),
                    'gold_documents_in_candidates': len(gold & {
                        row['document_id'] for row in found.candidates}),
                    'missing_gold_document_causes': sorted({
                        cause for causes in candidate_causes.values() for cause in causes}),
                    'gold_facts_delivered_proxy': sum(fact_delivered(
                        row['fact'], by_doc.get(document_id(row['title']), []))
                        for row in question['evidence_list']),
                    'gold_fact_count': len(question['evidence_list']),
                    'retrieval_ms': round((time.monotonic() - started) * 1000, 1),
                    'planning_ms': found.planning_ms,
                    'planning_tokens': found.planning_tokens,
                    'planning_error': found.planning_error,
                    'planned_queries_count': found.planned_queries_count,
                })
    summary = {}
    for mode in (('baseline', 'bm25') if args.bm25 else
                 ('baseline', 'document_quota_one') if args.document_quota_one else
                 ('baseline', 'source_query_plan') if args.plan else
                 ('baseline', 'article_first') if args.article_first else
                 ('baseline', 'source_facet_document') if args.facet_document else
                 ('baseline', 'source_facet') if args.facet else
                 ('baseline', 'rerank') if args.rerank else
                 ('baseline', 'source_clause', 'source_focus')):
        subset = [row for row in rows if row['mode'] == mode]
        summary[mode] = {
            'tasks': len(subset),
            'all_gold_documents': sum(row['all_gold_documents'] for row in subset),
            'gold_documents_found': sum(row['gold_documents_found'] for row in subset),
            'gold_documents': sum(row['gold_document_count'] for row in subset),
            'gold_documents_in_candidates': sum(row['gold_documents_in_candidates'] for row in subset),
            'gold_facts_delivered_proxy': sum(row['gold_facts_delivered_proxy'] for row in subset),
            'gold_facts': sum(row['gold_fact_count'] for row in subset),
            'p50_retrieval_ms': round(statistics.median(row['retrieval_ms'] for row in subset), 1),
            'p50_planning_ms': round(statistics.median(row['planning_ms'] for row in subset), 1),
            'planning_tokens': sum(row['planning_tokens'] for row in subset),
            'planning_errors': sum(row['planning_error'] is not None for row in subset),
        }
    by_stage = {}
    for label in sorted({row['target_stage'] for row in rows if row['target_stage']}):
        by_stage[label] = {}
        for mode in summary:
            subset = [row for row in rows if row['target_stage'] == label and row['mode'] == mode]
            by_stage[label][mode] = {
                'tasks': len(subset),
                'all_gold_documents': sum(row['all_gold_documents'] for row in subset),
                'gold_documents_found': sum(row['gold_documents_found'] for row in subset),
                'gold_documents_in_candidates': sum(row['gold_documents_in_candidates'] for row in subset),
                'gold_facts_delivered_proxy': sum(row['gold_facts_delivered_proxy'] for row in subset),
                'gold_facts': sum(row['gold_fact_count'] for row in subset),
                'p50_retrieval_ms': round(statistics.median(row['retrieval_ms'] for row in subset), 1),
            }
    result = {
        'scope': ('Retrieval-only; question novelty is recorded in the manifest. '
                  'No generated-answer accuracy or independent-human claim.'),
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'retrieval_code_sha256': hashlib.sha256(Path('app/retrieval.py').read_bytes()).hexdigest(),
        'facet_code_sha256': (hashlib.sha256(Path('app/retrieval_queries.py').read_bytes()).hexdigest()
                              if args.facet or args.facet_document else None),
        'rerank_code_sha256': (hashlib.sha256(Path('app/rerank.py').read_bytes()).hexdigest()
                               if args.rerank else None),
        'planner_code_sha256': (hashlib.sha256(Path('app/query_planner.py').read_bytes()).hexdigest()
                                if args.plan else None),
        'baseline_flags': {'retrieval_mode': cfg.retrieval_mode, 'source_focus_queries': cfg.source_focus_queries,
                           'source_facet_queries': cfg.source_facet_queries,
                           'source_facet_document_queries': cfg.source_facet_document_queries,
                           'article_first_lanes': cfg.article_first_lanes,
                           'source_query_plan': cfg.source_query_plan, 'passage_rerank': cfg.passage_rerank,
                           'document_quota': cfg.document_quota},
        'top_k': 6, 'summary': summary, 'by_stage': by_stage, 'rows': rows,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
