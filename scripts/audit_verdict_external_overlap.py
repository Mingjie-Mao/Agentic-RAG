"""Audit lexical near-duplicates, beyond the manifest's exact query-hash check."""
from difflib import SequenceMatcher
import json
from pathlib import Path

from build_multihop_subset import FILES, fetch, digest


def main():
    data = {digest(row['query']): row for row in fetch('MultiHopRAG.json', FILES['MultiHopRAG.json'], False)}
    used = {}
    for name in ['subset.json', 'subset-r2.json', 'subset-r3.json']:
        for item in json.loads((Path('fixtures/multihop') / name).read_text())['items']:
            used[item['query_sha256']] = item
    manifest = json.loads(Path('fixtures/verdict/external-oracle-r4-8.json').read_text())
    rows = []
    for item in manifest['items']:
        query = data[item['query_sha256']]
        similarities = [(SequenceMatcher(None, query['query'].lower(), data[key]['query'].lower()).ratio(), key)
                        for key in used]
        similarity, key = max(similarities)
        current_docs = {e['title'] for e in query['evidence_list']}
        old_docs = {e['title'] for e in data[key]['evidence_list']}
        rows.append({'id': item['id'], 'nearest_previous_id': used[key]['id'],
                     'query_similarity': round(similarity, 4), 'near_duplicate_at_090': similarity >= 0.90,
                     'shared_gold_documents': len(current_docs & old_docs)})
    output = {'scope': 'Post-run lexical audit only; original scores and selection remain frozen. Exact hash novelty does not ensure semantic independence. A 0.90 character similarity is a screening proxy, not human adjudication.',
              'near_duplicate_count': sum(row['near_duplicate_at_090'] for row in rows), 'total': len(rows), 'rows': rows}
    Path('artifacts/verdict/external-overlap-audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
