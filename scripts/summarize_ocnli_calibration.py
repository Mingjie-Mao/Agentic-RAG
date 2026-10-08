"""Paired human-label audit of the local 7B judge and small NLI classifier."""
import hashlib
import json
from pathlib import Path
import random

LLM = Path('artifacts/ocnli-semantic-external-v1.json')
NLI = Path('artifacts/ocnli-local-nli-v1.json')
OUT = Path('artifacts/ocnli-human-calibration-summary-v1.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rates(rows, arm):
    supported = [row for row in rows if row['gold'] == 'entailment']
    other = [row for row in rows if row['gold'] != 'entailment']
    return {'supported_recall': sum(row[arm] for row in supported) / len(supported),
            'false_accept_rate': sum(row[arm] for row in other) / len(other),
            'binary_accuracy': sum(row[arm] == (row['gold'] == 'entailment') for row in rows) / len(rows)}


def main():
    if OUT.exists():
        raise SystemExit('Refusing to overwrite paired calibration')
    llm = json.loads(LLM.read_text())
    nli = json.loads(NLI.read_text())
    if llm['metadata']['manifest_sha256'] != nli['metadata']['manifest_sha256']:
        raise SystemExit('Arms used different human-labeled cases')
    by_nli = {row['id']: row for row in nli['records']}
    rows = [{'id': row['id'], 'gold': row['gold'],
             'llm': row['predicted'] == 'supported',
             'nli': by_nli[row['id']]['predicted'] == 'entailment'}
            for row in llm['records']]
    if len(rows) != 60 or len(by_nli) != 60:
        raise SystemExit('Incomplete 60-case evaluation')
    rng = random.Random(42)
    differences = []
    for _ in range(2000):
        sample = [rng.choice(rows) for _ in rows]
        if any(row['gold'] == 'entailment' for row in sample) and any(
            row['gold'] != 'entailment' for row in sample):
            differences.append(rates(sample, 'nli')['binary_accuracy'] -
                               rates(sample, 'llm')['binary_accuracy'])
    differences.sort()
    result = {'scope': ('OCNLI hard gov/news human-majority sentence pairs only; '
                        'not enterprise QA, provenance checking or project-case human review'),
              'source': 'https://github.com/CLUEbenchmark/OCNLI',
              'sha256': {'llm_artifact': sha(LLM), 'nli_artifact': sha(NLI)},
              'n': len(rows), 'llm': rates(rows, 'llm'), 'nli': rates(rows, 'nli'),
              'paired_binary_accuracy_difference_nli_minus_llm': round(
                  rates(rows, 'nli')['binary_accuracy'] - rates(rows, 'llm')['binary_accuracy'], 3),
              'paired_bootstrap_difference_ci95': [round(differences[50], 3),
                                                   round(differences[1949], 3)],
              'llm_p50_ms': llm['summary']['p50_wall_ms'],
              'nli_total_ms': nli['summary']['elapsed_ms'],
              'nli_mean_ms_per_pair': round(nli['summary']['elapsed_ms'] / len(rows), 1),
              'project_cases_independent_human_reviewed': 0}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
