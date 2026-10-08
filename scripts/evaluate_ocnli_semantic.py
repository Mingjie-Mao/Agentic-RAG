"""Freeze and evaluate a human-labeled Chinese NLI slice, separate from project QA."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.clients import Models
from app.config import settings
from app.semantic_review import compact_review

SOURCE = Path('.runtime/ocnli-dev.json')
MANIFEST = Path('fixtures/source_contract/ocnli-hard-gov-news-v1.json')
OUTPUT = Path('artifacts/ocnli-semantic-external-v1.json')
LABELS = ('entailment', 'neutral', 'contradiction')
GENRES = ('gov', 'news')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_rows():
    return {row['id']: row for row in (json.loads(line) for line in SOURCE.open())}


def freeze(rows):
    if MANIFEST.exists():
        raise SystemExit('Manifest already frozen')
    chosen = []
    used_premises = set()
    for genre in GENRES:
        for label in LABELS:
            pool = [row for row in rows.values() if row['genre'] == genre
                    and row['level'] == 'hard' and row['label'] == label
                    and sum(row[f'label{i}'] == label for i in range(5)) >= 4]
            pool.sort(key=lambda row: digest(f"ocnli-v1|{row['id']}".encode()))
            count = 0
            for row in pool:
                if row['prem_id'] in used_premises:
                    continue
                used_premises.add(row['prem_id'])
                chosen.append({'id': row['id'], 'row_sha256': digest(
                    json.dumps(row, ensure_ascii=False, sort_keys=True).encode())})
                count += 1
                if count == 10:
                    break
            if count != 10:
                raise SystemExit(f'Insufficient rows for {genre}/{label}')
    MANIFEST.write_text(json.dumps({
        'name': 'ocnli-hard-gov-news-v1',
        'source_url': 'https://github.com/CLUEbenchmark/OCNLI/blob/main/data/ocnli/dev.json',
        'source_sha256': digest(SOURCE.read_bytes()),
        'selection': '10 per label per genre; hard; >=4/5 annotators agree; unique premise; deterministic id hash',
        'scope': 'Human-majority labels for sentence-pair NLI, not project-specific QA or production accuracy',
        'items': chosen,
    }, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'selected': len(chosen), 'source_sha256': digest(SOURCE.read_bytes())}))


def write(result):
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(OUTPUT)


def summary(records):
    if not records:
        return {}
    matrix = {gold: dict(Counter(row['predicted'] for row in records if row['gold'] == gold))
              for gold in LABELS}
    supported = [row for row in records if row['gold'] == 'entailment']
    unsupported = [row for row in records if row['gold'] != 'entailment']
    return {
        'n': len(records), 'by_label': dict(Counter(row['gold'] for row in records)),
        'supported_recall': round(sum(row['predicted'] == 'supported' for row in supported) / len(supported), 3),
        'not_supported_recall': round(sum(row['predicted'] != 'supported' for row in unsupported) / len(unsupported), 3),
        'false_accepts': sum(row['predicted'] == 'supported' for row in unsupported),
        'false_rejects': sum(row['predicted'] != 'supported' for row in supported),
        'three_label_confusion': matrix,
        'p50_wall_ms': round(statistics.median(row['wall_ms'] for row in records), 1),
        'prompt_tokens': sum(row['usage']['prompt_tokens'] for row in records),
        'completion_tokens': sum(row['usage']['completion_tokens'] for row in records),
    }


def run(rows):
    manifest = json.loads(MANIFEST.read_text())
    if digest(SOURCE.read_bytes()) != manifest['source_sha256']:
        raise SystemExit('Human-labeled source changed')
    metadata = {
        'manifest_sha256': digest(MANIFEST.read_bytes()),
        'rubric_sha256': digest(Path('app/semantic_review.py').read_bytes()),
        'model': settings().chat_model,
        'human_label_source': 'OCNLI dev majority of five annotators, >=4 agree',
        'project_case_human_reviewed': 0,
        'scope': 'External sentence-pair NLI only; potential model pretraining overlap; no QA relevance or citation validation',
    }
    result = {'metadata': metadata, 'records': []}
    if OUTPUT.exists():
        result = json.loads(OUTPUT.read_text())
        if result['metadata'] != metadata:
            raise SystemExit('Cannot resume with different inputs/model/rubric')
    done = {row['id'] for row in result['records']}
    model = Models()
    for entry in manifest['items']:
        if entry['id'] in done:
            continue
        row = rows[entry['id']]
        if digest(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()) != entry['row_sha256']:
            raise SystemExit(f"Source row changed: {entry['id']}")
        started = time.monotonic()
        label, usage = compact_review(model, '证据是否支持这项陈述？', row['sentence2'], row['sentence1'])
        result['records'].append({
            'id': row['id'], 'genre': row['genre'], 'gold': row['label'],
            'predicted': label['entailment'], 'reason': label['reason'],
            'usage': usage, 'wall_ms': round((time.monotonic() - started) * 1000, 1),
        })
        write(result)
        print(f"{len(result['records'])}/{len(manifest['items'])} {row['genre']} {row['label']} -> {label['entailment']}", flush=True)
    result['summary'] = summary(result['records'])
    write(result)
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    rows = source_rows()
    freeze(rows) if args.freeze else run(rows)


if __name__ == '__main__':
    main()
