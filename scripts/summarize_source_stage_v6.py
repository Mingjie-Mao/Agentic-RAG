"""Separate retrieved-candidate, admitted-context and answer failures on frozen v6."""
from collections import Counter
import hashlib
import json
from pathlib import Path

MANIFEST = Path('fixtures/source_contract/external-stage-v6.json')
RUN = Path('artifacts/source-stage-v6-baseline.json')
OUT = Path('artifacts/source-stage-v6-diagnosis.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(manifest, run):
    items = {item['id']: item for item in manifest['items']}
    records = run['records']
    if len(items) != 12 or len(records) != 12 or {row['id'] for row in records} != set(items):
        raise ValueError('Expected the complete 12-case baseline, once per case')
    groups = {}
    diagnostic = []
    for target in manifest['stages']:
        rows = [row for row in records if items[row['id']]['target_stage'] == target]
        if len(rows) != 4:
            raise ValueError(f'Wrong frozen denominator for {target}')
        groups[target] = {
            'tasks': len(rows), 'literal_correct': sum(row['answer_correct'] for row in rows),
            'false_refusals': sum(row['status'] == 'insufficient_evidence' for row in rows),
            'first_context_all_gold_documents': sum(row['first_context_gold_documents'] for row in rows),
            'first_context_gold_facts_delivered_proxy': sum(row['gold_facts_delivered_proxy'] for row in rows),
            'gold_facts': sum(row['gold_facts'] for row in rows),
            'execution_errors': sum(bool(row['execution_error']) for row in rows),
            'total_tokens': sum(row['total_prompt_tokens'] + row['total_completion_tokens'] for row in rows),
        }
        for row in rows:
            facts_complete = (row['first_context_gold_documents']
                              and row['gold_facts_delivered_proxy'] == row['gold_facts'])
            diagnostic.append({
                'id': row['id'], 'target_stage': target,
                'freeze_fact_proxy_reached': items[row['id']]['fact_proxy_reached'],
                'first_context_facts_complete': facts_complete,
                'answer_correct': row['answer_correct'], 'status': row['status'],
                'failure_layer': ('correct' if row['answer_correct'] else
                                  'answer_or_scoring_review' if facts_complete else
                                  'document_or_passage_missing'),
            })
    return {
        'by_frozen_stage': groups,
        'observed_failure_layers': dict(Counter(row['failure_layer'] for row in diagnostic)),
        'fact_complete_but_wrong_ids': [row['id'] for row in diagnostic
                                        if row['failure_layer'] == 'answer_or_scoring_review'],
        'records': diagnostic,
        'note': ('Gold fact delivery uses word-4gram overlap, not semantic support. '
                 'A fact-complete wrong answer still needs source and scoring review; '
                 'it is not automatically a model reasoning error.'),
    }


def main():
    if OUT.exists():
        raise SystemExit('Refusing to overwrite stage diagnosis')
    manifest = json.loads(MANIFEST.read_text())
    run = json.loads(RUN.read_text())
    if run['metadata']['manifest_sha256'] != sha(MANIFEST):
        raise SystemExit('Run used another manifest')
    result = {'manifest_sha256': sha(MANIFEST), 'run_sha256': sha(RUN),
              **summarize(manifest, run)}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'records'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
