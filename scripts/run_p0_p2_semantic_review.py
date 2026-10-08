"""Calibrate a compact shadow rubric; independent human gold is optional and explicit."""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.clients import Models
from app.config import settings
from app.semantic_review import compact_review
from prepare_source_human_review import audited_labels
from run_source_semantic_calibration import metrics, write

FIXTURE = Path('fixtures/source_contract/semantic-calibration-v4.json')
OUT = Path('artifacts/p0-p2-semantic-calibration-v5.json')


def main():
    fixture = json.loads(FIXTURE.read_text())
    metadata = {'fixture_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                'code_sha256': hashlib.sha256(Path('app/semantic_review.py').read_bytes()).hexdigest(),
                'model': settings().chat_model, 'rubric': 'compact-source-entailment-v5',
                'api_cost': 0, 'human_reviewed': 0,
                'scope': 'Previously used authored calibration cases, not unseen generalization. No production semantic blocking.'}
    result = {'metadata': metadata, 'records': []}
    if OUT.exists():
        result = json.loads(OUT.read_text())
        if result['metadata'] != metadata:
            raise ValueError('Frozen code/input changed; do not overwrite experiment')
    done = {row['id'] for row in result['records']}
    model = Models()
    for item in fixture['items']:
        if item['id'] in done:
            continue
        started = time.monotonic()
        label, usage = compact_review(model, item['question'], item['claim'], item['evidence'])
        result['records'].append({'id': item['id'], 'category': item['category'],
            'expected': item['expected'], 'label': label, 'usage': usage,
            'wall_ms': round((time.monotonic() - started) * 1000, 1)})
        write(OUT, result)
        print(item['id'], label['entailment'], flush=True)
    result['summary'] = metrics(result['records'])
    result['by_category'] = {category: metrics([row for row in result['records'] if row['category'] == category])
                             for category in sorted({row['category'] for row in result['records']})}
    packet = json.loads(Path('artifacts/source-human-review-p0-p2.json').read_text())
    try:
        if packet['source_sha256'] != metadata['fixture_sha256']:
            raise ValueError('Human review source mismatch')
        labels = audited_labels(packet)
    except ValueError:
        result['independent_human_calibration'] = {'status': 'pending', 'reviewed': 0}
    else:
        result['independent_human_calibration'] = {
            'status': 'complete', 'reviewed': len(labels),
            'label_agreement': sum(row['label']['entailment'] == labels[row['id']] for row in result['records'])}
    result['decision'] = 'shadow_only; human validation and fresh external quality gate required'
    write(OUT, result)
    print(json.dumps({'summary': result['summary'], 'human': result['independent_human_calibration']}, indent=2))


if __name__ == '__main__':
    main()
