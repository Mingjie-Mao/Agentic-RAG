"""Describe opt-in v6 retrieval probes without a post-hoc promotion gate."""
import hashlib
import json
from pathlib import Path

MANIFEST = Path('fixtures/source_contract/external-stage-v6.json')
PROBES = {
    'source_facet_document': Path('artifacts/source-stage-v6-facet-document.json'),
    'document_quota_one': Path('artifacts/source-stage-v6-document-quota-one.json'),
    'article_first': Path('artifacts/source-stage-v6-article-first.json'),
}
OUT = Path('artifacts/source-stage-v6-ablation-summary.json')


def main():
    if OUT.exists():
        raise SystemExit('Refusing to overwrite v6 ablation summary')
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    result = {
        'scope': ('Gold-stratified development challenge; exploratory treatment probes after baseline. '
                  'No pre-registered promotion gate, answer quality test or independent human review.'),
        'manifest_sha256': manifest_sha, 'probes': {},
    }
    retrieval_hash = None
    for arm, path in PROBES.items():
        artifact = json.loads(path.read_text())
        if artifact['manifest_sha256'] != manifest_sha:
            raise SystemExit(f'Manifest changed for {arm}')
        if retrieval_hash is not None and artifact['retrieval_code_sha256'] != retrieval_hash:
            raise SystemExit('Retrieval code differs between probes')
        retrieval_hash = artifact['retrieval_code_sha256']
        if {mode for mode in artifact['summary']} != {'baseline', arm}:
            raise SystemExit(f'Unexpected arms for {arm}')
        stages = {}
        for name, values in artifact['by_stage'].items():
            before, after = values['baseline'], values[arm]
            if before['tasks'] != 4 or after['tasks'] != 4:
                raise SystemExit(f'Incomplete stage for {arm}/{name}')
            stages[name] = {
                'all_gold_documents': [before['all_gold_documents'], after['all_gold_documents']],
                'gold_documents_found': [before['gold_documents_found'], after['gold_documents_found']],
                'gold_documents_in_candidates': [before['gold_documents_in_candidates'],
                                                 after['gold_documents_in_candidates']],
                'gold_facts_delivered_proxy': [before['gold_facts_delivered_proxy'],
                                               after['gold_facts_delivered_proxy']],
                'p50_retrieval_ms': [before['p50_retrieval_ms'], after['p50_retrieval_ms']],
            }
        result['probes'][arm] = {
            'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'overall': {key: [artifact['summary']['baseline'][key], artifact['summary'][arm][key]]
                        for key in ('all_gold_documents', 'gold_documents_found',
                                    'gold_documents_in_candidates', 'gold_facts_delivered_proxy',
                                    'p50_retrieval_ms')},
            'by_stage': stages,
        }
    result['retrieval_code_sha256'] = retrieval_hash
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['probes'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
