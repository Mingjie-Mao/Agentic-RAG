"""Freeze/rebuild a fresh, small, balanced oracle-claims judgment experiment.

Only IDs and hashes are committed. Runtime facts come from the pinned local dataset;
this measures conclusion reasoning when benchmark facts are handed to the model,
not retrieval, generated-claim entailment, or end-to-end Agent accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path

from build_multihop_subset import FILES, fetch, digest

MANIFEST = Path('fixtures/verdict/external-oracle-r4-8.json')
RUNTIME = Path('.runtime/verdict-external-oracle-r4-8.json')
BATCH = Path('fixtures/multihop/subset-r4.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    data = {digest(row['query']): row for row in fetch('MultiHopRAG.json', FILES['MultiHopRAG.json'], False)}
    batch = json.loads(BATCH.read_text())
    if args.freeze:
        if MANIFEST.exists():
            raise SystemExit('Manifest already frozen; refusing to overwrite')
        selected = []
        for kind in ['comparison_query', 'temporal_query']:
            for gold in ['yes', 'no']:
                eligible = [item for item in batch['items']
                            if item['question_type'] == kind
                            and data[item['query_sha256']]['answer'].strip().lower() == gold
                            and 1 <= len(data[item['query_sha256']]['evidence_list']) <= 4]
                chosen = sorted(eligible, key=lambda item: item['query_sha256'])[:2]
                if len(chosen) != 2:
                    raise SystemExit(f'Not enough eligible {kind}/{gold} questions')
                selected.extend(chosen)
        used = set()
        for name in ['subset.json', 'subset-r2.json', 'subset-r3.json']:
            used.update(item['query_sha256'] for item in json.loads((BATCH.parent / name).read_text())['items'])
        if used & {item['query_sha256'] for item in selected}:
            raise SystemExit('Previously used question overlap')
        manifest = {'name': 'external-oracle-claims-r4-8', 'source_sha256': FILES['MultiHopRAG.json'],
                    'batch_sha256': hashlib.sha256(BATCH.read_bytes()).hexdigest(),
                    'selection': 'First two sha256-sorted questions per comparison/temporal and yes/no stratum; 1-4 benchmark facts; no overlap with B1-B3.',
                    'scope': 'Fresh external oracle facts, not retrieved/generated claims. Balanced majority baseline 4/8. No independent human source adjudication.',
                    'items': selected}
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    manifest = json.loads(MANIFEST.read_text())
    if manifest['source_sha256'] != FILES['MultiHopRAG.json'] or manifest['batch_sha256'] != hashlib.sha256(BATCH.read_bytes()).hexdigest():
        raise SystemExit('Pinned input mismatch')
    rows = []
    for item in manifest['items']:
        original = data[item['query_sha256']]
        if digest(original['answer']) != item['answer_sha256']:
            raise SystemExit('Gold answer hash mismatch')
        rows.append({'id': item['id'], 'category': item['question_type'], 'question': original['query'],
                     'expected': original['answer'].strip().lower(),
                     'claims': [{'text': f"Source: {evidence['source']}; Published: {evidence['published_at']}. {evidence['fact']}", 'evidence_ids': [f"oracle-{number}"]}
                                for number, evidence in enumerate(original['evidence_list'], 1)]})
    RUNTIME.write_text(json.dumps({'name': manifest['name'], 'scope': manifest['scope'], 'items': rows}, ensure_ascii=False, indent=2)+'\n')
    print(f'{len(rows)} oracle-claims cases reconstructed in {RUNTIME}; facts are not committed')


if __name__ == '__main__':
    main()
