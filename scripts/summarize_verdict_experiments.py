"""Report paired judgment-only experiments without treating them as Agent scores."""
import hashlib
import json
from pathlib import Path
import statistics

INPUTS = {
    'synthetic_free_pilot': 'artifacts/verdict-synthetic-v1.json',
    'synthetic_constrained_development': 'artifacts/verdict-synthetic-v2-constrained.json',
    'fresh_external_oracle': 'artifacts/verdict-external-oracle-r4-8.json',
}


def summarize(payload):
    paired = {}
    for row in payload['records']:
        arms = paired.setdefault(row['id'], {})
        if row['protocol'] in arms:
            raise ValueError('Duplicate arm record')
        arms[row['protocol']] = row
    if not paired or any(set(arms) != {'legacy', 'structured'} for arms in paired.values()):
        raise ValueError('Incomplete pair; experiment is still running')
    if any(arms['legacy']['expected'] != arms['structured']['expected'] for arms in paired.values()):
        raise ValueError('Paired gold mismatch')
    result = {'questions': len(paired), 'arms': {}, 'paired': {
        'legacy_only_correct': sum(arms['legacy']['correct'] and not arms['structured']['correct'] for arms in paired.values()),
        'structured_only_correct': sum(arms['structured']['correct'] and not arms['legacy']['correct'] for arms in paired.values()),
        'both_correct': sum(arms['structured']['correct'] and arms['legacy']['correct'] for arms in paired.values()),
        'both_wrong': sum(not arms['structured']['correct'] and not arms['legacy']['correct'] for arms in paired.values()),
    }}
    for protocol in ['legacy', 'structured']:
        rows = [arms[protocol] for arms in paired.values()]
        latency = [row['latency_ms'] for row in rows]
        result['arms'][protocol] = {
            'correct': sum(row['correct'] for row in rows), 'total': len(rows),
            'p50_ms': round(statistics.median(latency), 1),
            'p95_ms': round(statistics.quantiles(latency, n=100, method='inclusive')[94], 1),
            'prompt_tokens': sum(row['usage'].get('prompt_tokens', 0) for row in rows),
            'completion_tokens': sum(row['usage'].get('completion_tokens', 0) for row in rows),
            'structural_rejections': sum(bool((row['verdict'] or {}).get('validation_issues')) for row in rows),
            'missing_verdict': sum(row['verdict'] is None for row in rows),
        }
    return result


def main():
    output = {'scope': 'Judgment-only, local Qwen2.5 7B. Synthetic development and fresh oracle-facts probe are separate. Not end-to-end accuracy or semantic source verification.',
              'api_cost': 0, 'experiments': {}}
    for name, path in INPUTS.items():
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
        output['experiments'][name] = {'path': path, 'artifact_sha256': hashlib.sha256(raw).hexdigest(), **summarize(payload)}
    output['deployment_decision'] = 'keep_legacy_default; structured protocol is experimental and has not met the external quality/latency gate'
    out = Path('artifacts/verdict/experiment-summary.json')
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
