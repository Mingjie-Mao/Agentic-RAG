"""Make a local blind review packet for required versus optional gold articles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_multihop_subset import FILES, digest, fetch

MANIFEST = Path('fixtures/source_contract/external-p0-p2-v5-bm25.json')
OUT = Path('artifacts/local/required-documents-review-v5.json')
LABELS = {'required', 'supporting_optional', 'irrelevant', 'unclear'}


def review_items(manifest):
    data = {digest(row['query']): row for row in fetch('MultiHopRAG.json', FILES['MultiHopRAG.json'], False)}
    items = []
    for item in manifest['items']:
        question = data[item['query_sha256']]
        if digest(question['answer']) != item['answer_sha256']:
            raise ValueError('Gold answer changed')
        if not question.get('evidence_list'):
            continue
        items.append({'id': item['id'], 'question': question['query'],
                      'gold_answer': question['answer'], 'evidence': [
                          {'title': row['title'], 'fact': row['fact'],
                           'label': None, 'reason': ''} for row in question['evidence_list']]})
    return items


def validate(packet, expected):
    if not packet.get('reviewer') or packet.get('reviewer_type') != 'human':
        raise ValueError('An independent human reviewer is required')
    if len(packet['items']) != len(expected):
        raise ValueError('Review packet is incomplete')
    for item, original in zip(packet['items'], expected, strict=True):
        if any(item.get(key) != original[key] for key in ('id', 'question', 'gold_answer')):
            raise ValueError('A question or gold answer changed')
        if len(item['evidence']) != len(original['evidence']):
            raise ValueError(f"Evidence list changed for {item['id']}")
        for source, source_original in zip(item['evidence'], original['evidence'], strict=True):
            if any(source.get(key) != source_original[key] for key in ('title', 'fact')):
                raise ValueError(f"Source text changed for {item['id']}")
            if source.get('label') not in LABELS or not source.get('reason', '').strip():
                raise ValueError(f"Missing label or reason for {item['id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if args.validate:
        packet = json.loads(OUT.read_text())
        if packet['manifest_sha256'] != hashlib.sha256(MANIFEST.read_bytes()).hexdigest():
            raise SystemExit('Review packet does not match frozen manifest')
        try:
            validate(packet, review_items(manifest))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Validated {len(packet['items'])} questions from {packet['reviewer']}")
        return
    if OUT.exists():
        raise SystemExit('Refusing to overwrite review packet')
    items = review_items(manifest)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        'scope': 'Local third-party text; independent required-source review; no model predictions shown',
        'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        'reviewer': None, 'reviewer_type': None, 'items': items,
    }, ensure_ascii=False, indent=2) + '\n')
    print(f'Prepared {len(items)} answerable questions at {OUT}; human-reviewed=0')


if __name__ == '__main__':
    main()
