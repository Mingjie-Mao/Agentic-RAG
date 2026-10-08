"""Check frozen review packets without inventing independent human labels."""
import hashlib
import json
from pathlib import Path

from prepare_required_document_review import MANIFEST, OUT as DOC_PACKET, review_items

SEMANTIC_SOURCE = Path('fixtures/source_contract/semantic-calibration-v4.json')
SEMANTIC_PACKET = Path('artifacts/source-human-review-p0-p2.json')
OUT = Path('artifacts/source-review-readiness-v6.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise SystemExit('Refusing to overwrite review readiness audit')
    source = json.loads(SEMANTIC_SOURCE.read_text())['items']
    semantic = json.loads(SEMANTIC_PACKET.read_text())
    original_by_id = {row['id']: row for row in source}
    semantic_ids = [row['id'] for row in semantic['items']]
    semantic_unchanged = (semantic['source_sha256'] == sha(SEMANTIC_SOURCE)
                          and len(semantic_ids) == len(original_by_id)
                          and len(set(semantic_ids)) == len(semantic_ids)
                          and all(all(row.get(key) == original_by_id[row['id']][key]
                                      for key in ('category', 'question', 'evidence', 'claim'))
                                  for row in semantic['items']))
    semantic_completed = sum(bool(row.get('entailment') and row.get('relevance')
                                  and row.get('reason')) for row in semantic['items'])

    document = json.loads(DOC_PACKET.read_text())
    expected = review_items(json.loads(MANIFEST.read_text()))
    document_unchanged = (document['manifest_sha256'] == sha(MANIFEST)
                          and len(document['items']) == len(expected)
                          and all(item['id'] == original['id']
                                  and item['question'] == original['question']
                                  and item['gold_answer'] == original['gold_answer']
                                  and len(item['evidence']) == len(original['evidence'])
                                  and all(a['title'] == b['title'] and a['fact'] == b['fact']
                                          for a, b in zip(item['evidence'], original['evidence'], strict=True))
                                  for item, original in zip(document['items'], expected, strict=True)))
    sources = [source for item in document['items'] for source in item['evidence']]
    document_completed = sum(bool(source.get('label') and source.get('reason')) for source in sources)
    result = {
        'scope': 'Structural readiness only; no semantic or required-source labels inferred',
        'semantic': {'items': len(source), 'content_unchanged': semantic_unchanged,
                     'labels_complete': semantic_completed,
                     'independent_human_reviewed': semantic_completed if semantic.get('reviewer')
                     and semantic.get('reviewer_type') == 'human' else 0},
        'required_documents': {'questions': len(expected), 'sources': len(sources),
                               'content_unchanged': document_unchanged,
                               'labels_complete': document_completed,
                               'independent_human_reviewed': document_completed if document.get('reviewer')
                               and document.get('reviewer_type') == 'human' else 0},
    }
    if not semantic_unchanged or not document_unchanged:
        raise SystemExit('Review source content or fingerprint changed')
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
