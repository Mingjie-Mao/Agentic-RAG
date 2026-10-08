"""Freeze a new, screened 12-task set, excluding all previous actual model runs."""
import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import select
from app.db import SessionLocal
from app.models import AgentTask, Answer
from build_multihop_subset import FILES, fetch, digest
from freeze_source_facts_external import freeze

OUT = Path('fixtures/source_contract/external-p0-p2-v1.json')


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--smoke', action='store_true', help='Four balanced comparison/temporal tasks, no global quality claim')
    parser.add_argument('--retrieval-focus', action='store_true',
                        help='Two-arm retrieval-focused workflow experiment')
    parser.add_argument('--bm25', action='store_true',
                        help='Two-arm BM25-only versus hybrid retrieval experiment')
    args = parser.parse_args()
    OUT = args.output
    if OUT.exists():
        raise SystemExit('Frozen manifest exists; refusing overwrite')
    data = {digest(row['query']): row for row in fetch('MultiHopRAG.json', FILES['MultiHopRAG.json'], False)}
    titles = {row['title'] for row in fetch('corpus.json', FILES['corpus.json'], False)}
    used = set()
    for path in [*Path('fixtures/multihop').glob('subset*.json'),
                 *Path('fixtures/source_contract').glob('external*.json'),
                 Path('fixtures/verdict/external-oracle-r4-8.json')]:
        manifest = json.loads(path.read_text())
        used.update(row['query_sha256'] for row in manifest['items'])
    with SessionLocal() as db:
        used.update(digest(text) for text in db.scalars(select(AgentTask.goal).where(AgentTask.user_id == 'mh-eval')))
        used.update(digest(text) for text in db.scalars(select(Answer.question).where(Answer.user_id == 'mh-eval')))
    pool = {key for key, row in data.items() if key not in used
            and all(e['title'] in titles for e in row.get('evidence_list', []))}
    items, rejected = freeze(data, used, pool)
    if args.smoke:
        items = [items[index] for index in (0, 2, 4, 6)]
    result = {'name': OUT.stem, 'source_files': FILES,
              'scope': 'Fresh queries, mechanical novelty screening, literal v3. Human review pending; no overall semantic or pretraining independence claim.',
              'arms': (['hybrid', 'bm25'] if args.bm25 else
                       ['legacy', 'retrieval_focus'] if args.retrieval_focus else
                       ['legacy', 'candidate'] if args.smoke else ['legacy', 'candidate', 'partial_facts']),
              'smoke_only': args.smoke,
              'selection': 'Same strata as SF-v1; exclude previous query hashes and gold-document combinations; character/token similarity <0.8.',
              'used_question_hashes': sorted(used), 'items': items, 'screened_out': rejected,
              'gate': {'minimum_answer_gain': 2, 'maximum_latency_ratio': 1.15,
                       'maximum_total_token_ratio': 1.15}}
    if args.bm25:
        result['retrieval_gate'] = {
            'minimum_all_gold_documents_delta': 0,
            'minimum_gold_documents_delta': 1,
            'minimum_gold_fact_proxy_delta': 0,
            'maximum_p50_latency_ratio': 1.0,
            'requires_separate_answer_quality_gate': True,
        }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'selected': len(items), 'excluded': len(used),
                      'screened_out': len(rejected),
                      'max_similarity': max(row['max_similarity'] for row in items)}))


if __name__ == '__main__':
    main()
