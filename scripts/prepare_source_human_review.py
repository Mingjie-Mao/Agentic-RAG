"""Prepare blind independent human review; never fill labels from model predictions."""
import hashlib
import json
from pathlib import Path


def prepare(source, target):
    raw = source.read_bytes()
    fixture = json.loads(raw)
    packet = {'source_sha256': hashlib.sha256(raw).hexdigest(),
              'reviewer': None, 'reviewer_type': 'human', 'status': 'pending',
              'rubric': 'Judge entailment separately from relevance. Preserve speaker, modality, negation scope, compared attribute, time and conclusion. Atomic false claims are unsupported; genuinely mixed claims may be partial.',
              'items': [{key: row[key] for key in ('id', 'category', 'question', 'evidence', 'claim')}
                        | {'entailment': None, 'relevance': None, 'reason': None}
                        for row in fixture['items']]}
    if target.exists():
        raise ValueError('Review packet exists; refusing to overwrite human work')
    target.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + '\n')
    return packet


def audited_labels(packet):
    if packet.get('reviewer_type') != 'human' or not str(packet.get('reviewer') or '').strip():
        raise ValueError('Independent human reviewer identity is required')
    source = Path(__file__).resolve().parents[1] / 'fixtures/source_contract/semantic-calibration-v4.json'
    if packet.get('source_sha256') != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('Human review source fingerprint differs')
    expected = {row['id']: row for row in json.loads(source.read_text())['items']}
    ids = [row.get('id') for row in packet['items']]
    if len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError('Review must cover the entire frozen set exactly once')
    labels = {'supported', 'partial', 'unsupported', 'unclear'}
    for row in packet['items']:
        if any(row.get(key) != expected[row['id']][key]
               for key in ('category', 'question', 'evidence', 'claim')):
            raise ValueError('Human review changed frozen case content')
        if row.get('entailment') not in labels or row.get('relevance') not in {
            'answers', 'related', 'off_topic', 'unclear'} or not row.get('reason'):
            raise ValueError('Every item requires two labels and an explanation')
    return {row['id']: row['entailment'] for row in packet['items']}


if __name__ == '__main__':
    packet = prepare(Path('fixtures/source_contract/semantic-calibration-v4.json'),
                     Path('artifacts/source-human-review-p0-p2.json'))
    print(f"Prepared {len(packet['items'])} blind cases; independent human labels=0")
