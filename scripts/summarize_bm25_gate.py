"""Apply the frozen retrieval and answer gates without rewriting run artifacts."""
import hashlib
import json
from pathlib import Path

MANIFEST = Path('fixtures/source_contract/external-p0-p2-v5-bm25.json')
RETRIEVAL = Path('artifacts/bm25-fresh-v5.json')
ANSWERS = Path('artifacts/p0-p2-bm25-end-to-end-v5.json')
OUT = Path('artifacts/bm25-v5-gate.json')


def evaluate(manifest, retrieval, answers):
    baseline, candidate = (retrieval['summary'][arm] for arm in ('baseline', 'bm25'))
    criteria = manifest['retrieval_gate']
    checks = {
        'all_gold_documents': candidate['all_gold_documents'] - baseline['all_gold_documents']
        >= criteria['minimum_all_gold_documents_delta'],
        'gold_document_count': candidate['gold_documents_found'] - baseline['gold_documents_found']
        >= criteria['minimum_gold_documents_delta'],
        'gold_fact_proxy': candidate['gold_facts_delivered_proxy'] - baseline['gold_facts_delivered_proxy']
        >= criteria['minimum_gold_fact_proxy_delta'],
        'p50_retrieval_latency': candidate['p50_retrieval_ms']
        <= baseline['p50_retrieval_ms'] * criteria['maximum_p50_latency_ratio'],
    }
    quality = answers.get('gates', {}).get('bm25', {})
    return {
        'retrieval_checks': checks,
        'retrieval_passed': all(checks.values()),
        'answer_gate': quality,
        'answer_passed': quality.get('decision') == 'eligible_for_larger_validation',
        'decision': ('eligible_for_larger_validation' if all(checks.values()) and
                     quality.get('decision') == 'eligible_for_larger_validation'
                     else 'keep_default_off'),
        'note': 'Literal v3 answer scoring and lexical fact proxy; project-case independent human review=0. '
                'Even passing both gates would require larger validation before enabling by default.',
    }


def main():
    if OUT.exists():
        raise SystemExit('Refusing to overwrite frozen gate result')
    manifest = json.loads(MANIFEST.read_text())
    retrieval = json.loads(RETRIEVAL.read_text())
    answers = json.loads(ANSWERS.read_text())
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if retrieval['manifest_sha256'] != manifest_sha or answers['metadata']['manifest_sha256'] != manifest_sha:
        raise SystemExit('Input manifests differ')
    if len(answers['records']) != len(manifest['items']) * len(manifest['arms']):
        raise SystemExit('Answer experiment incomplete')
    result = {
        'manifest_sha256': manifest_sha,
        'retrieval_sha256': hashlib.sha256(RETRIEVAL.read_bytes()).hexdigest(),
        'answers_sha256': hashlib.sha256(ANSWERS.read_bytes()).hexdigest(),
        **evaluate(manifest, retrieval, answers),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
