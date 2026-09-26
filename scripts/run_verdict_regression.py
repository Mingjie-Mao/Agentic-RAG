"""Paired, local claims-only experiment, without retrieval or answer regeneration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.clients import DependencyError, Models
from app.config import settings
from app.qa import answer_verdict


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=Path('fixtures/verdict/synthetic-v1.json'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/verdict-synthetic-v1.json'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--span-mode', choices=['free', 'constrained'], default='constrained')
    args = parser.parse_args()
    raw = args.fixture.read_bytes()
    fixture = json.loads(raw)
    cfg = settings()
    cfg.verdict_span_mode = args.span_mode
    metadata = {'fixture_sha256': hashlib.sha256(raw).hexdigest(), 'model': cfg.chat_model,
                'scope': fixture['scope'], 'protocols': ['legacy', 'structured'], 'api_cost': 0,
                'span_mode': args.span_mode,
                'code_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                for name in ['app/clients.py', 'app/verdict.py', 'app/qa.py']}}
    result = {'metadata': metadata, 'records': []}
    if args.out.exists():
        if not args.resume:
            raise SystemExit('Artifact exists; use --resume or a new output path')
        result = json.loads(args.out.read_text())
        if result['metadata'] != metadata:
            raise SystemExit('Resume configuration or code hash mismatch')
    done = {(row['id'], row['protocol']) for row in result['records']}
    models = Models()
    original = cfg.verdict_protocol
    try:
        for number, item in enumerate(fixture['items']):
            # Alternate which arm runs first; both are warmed by the same model server.
            protocols = ['legacy', 'structured'] if number % 2 == 0 else ['structured', 'legacy']
            for protocol in protocols:
                if (item['id'], protocol) in done:
                    continue
                cfg.verdict_protocol = protocol
                started = time.monotonic()
                try:
                    verdict, usage = answer_verdict(models, item['question'], item['claims'], 'answered')
                    error = None
                except DependencyError as exc:
                    verdict, usage, error = None, {}, exc.stage
                record = {'id': item['id'], 'category': item['category'], 'protocol': protocol,
                          'expected': item['expected'], 'verdict': verdict, 'usage': usage,
                          'latency_ms': round((time.monotonic()-started)*1000, 1), 'error': error,
                          'correct': bool(verdict and verdict['value'] == item['expected'])}
                result['records'].append(record)
                save(args.out, result)
                print(json.dumps({key: record[key] for key in ['id', 'protocol', 'correct', 'latency_ms']}, ensure_ascii=False), flush=True)
        result['summary'] = {}
        for protocol in ['legacy', 'structured']:
            rows = [row for row in result['records'] if row['protocol'] == protocol]
            elapsed = sorted(row['latency_ms'] for row in rows)
            result['summary'][protocol] = {
                'correct': sum(row['correct'] for row in rows), 'total': len(rows),
                'unclear': sum(bool(row['verdict']) and row['verdict']['value'] == 'unclear' for row in rows),
                'missing_verdict': sum(row['verdict'] is None for row in rows),
                'p50_ms': (elapsed[(len(elapsed)-1)//2]+elapsed[len(elapsed)//2])/2,
                'completion_tokens': sum(row['usage'].get('completion_tokens',0) for row in rows),
                'by_category': {category: {'correct': sum(row['correct'] for row in rows if row['category'] == category), 'total': count}
                                for category, count in Counter(row['category'] for row in rows).items()},
            }
        save(args.out, result)
        print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
    finally:
        cfg.verdict_protocol = original


if __name__ == '__main__':
    main()
