"""Retrospective rule audit of authored v5 labels; never a production quality claim."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.answer_contract import source_audit


def main():
    fixture_path = Path('fixtures/source_contract/semantic-calibration-v4.json')
    labels_path = Path('artifacts/p0-p2-semantic-calibration-v5.json')
    output_path = Path('artifacts/p0-p2-semantic-postcheck-v7.json')
    if output_path.exists():
        raise SystemExit('Refusing to overwrite postcheck artifact')
    authored = {row['id']: row for row in json.loads(fixture_path.read_text())['items']}
    records = []
    for row in json.loads(labels_path.read_text())['records']:
        case = authored[row['id']]
        issues = source_audit(case['claim'], [case['evidence']], check_polarity=True)
        before = row['label']['entailment']
        # Candidate veto is intentionally narrow; other diagnostics are recorded
        # but cannot be assumed to prove that a supported claim is false.
        veto = any(issue in {'unbounded_document_absence', 'possible_speaker_attribution_loss',
                             'explicit_source_negation_lost'} for issue in issues)
        after = 'unsupported' if before in {'supported', 'unclear'} and veto else before
        records.append({'id': row['id'], 'category': row['category'],
                        'expected_assistant_label': case['expected'],
                        'before': before, 'candidate_after': after, 'issues': issues})
    result = {
        'metadata': {
            'scope': ('Previously used assistant-authored cases; retrospective rule audit. '
                      'No independent human label, unseen generalization, or production semantic gate.'),
            'human_reviewed': 0,
            'fixture_sha256': hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            'v5_sha256': hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            'code_sha256': {
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in ('app/answer_contract.py', 'app/claim_consistency.py')
            },
        },
        'summary': {
            'cases': len(records),
            'before_agreement': sum(r['before'] == r['expected_assistant_label'] for r in records),
            'candidate_after_agreement': sum(r['candidate_after'] == r['expected_assistant_label'] for r in records),
            'before_false_accepts': sum(r['before'] == 'supported' and r['expected_assistant_label'] != 'supported' for r in records),
            'candidate_false_accepts': sum(r['candidate_after'] == 'supported' and r['expected_assistant_label'] != 'supported' for r in records),
            'before_false_rejects': sum(r['before'] != 'supported' and r['expected_assistant_label'] == 'supported' for r in records),
            'candidate_false_rejects': sum(r['candidate_after'] != 'supported' and r['expected_assistant_label'] == 'supported' for r in records),
        },
        'records': records,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
