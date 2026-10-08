"""Prepare a local blind review of v6 fact-complete, literal-score failures."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db import SessionLocal
from app.models import AgentTask
from build_multihop_subset import FILES, fetch, digest

MANIFEST = Path('fixtures/source_contract/external-stage-v6.json')
RUN = Path('artifacts/source-stage-v6-baseline.json')
DIAGNOSIS = Path('artifacts/source-stage-v6-diagnosis.json')
OUT = Path('artifacts/local/source-stage-v6-verdict-review.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise SystemExit('Refusing to overwrite local human review work')
    manifest = json.loads(MANIFEST.read_text())
    run = json.loads(RUN.read_text())
    diagnosis = json.loads(DIAGNOSIS.read_text())
    if run['metadata']['manifest_sha256'] != sha(MANIFEST) or diagnosis['run_sha256'] != sha(RUN):
        raise SystemExit('Frozen artifacts differ')
    by_id = {item['id']: item for item in manifest['items']}
    by_result = {row['id']: row for row in run['records']}
    data = {digest(row['query']): row for row in fetch('MultiHopRAG.json', FILES['MultiHopRAG.json'], False)}
    items = []
    with SessionLocal() as db:
        for case_id in diagnosis['fact_complete_but_wrong_ids']:
            item, result = by_id[case_id], by_result[case_id]
            question = data[item['query_sha256']]
            task = db.get(AgentTask, result['task_id'])
            if not task or digest(question['answer']) != item['answer_sha256']:
                raise SystemExit(f'Missing task or changed gold for {case_id}')
            claims = [{'text': claim.get('text'), 'quotes': claim.get('quotes', [])}
                      for claim in task.result.get('claims', [])]
            items.append({
                'id': case_id, 'question': question['query'], 'gold_answer': question['answer'],
                'gold_evidence': question['evidence_list'], 'actual_claims': claims,
                'actual_verdict': task.result.get('verdict'),
                'gold_validity': None, 'answer_semantics': None, 'reason': None,
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        'scope': 'Local third-party text, blind human adjudication of two literal-score failures',
        'manifest_sha256': sha(MANIFEST), 'run_sha256': sha(RUN),
        'reviewer': None, 'reviewer_type': None,
        'rubric': ('Judge whether the original yes/no gold follows from the quoted evidence, '
                  'then whether the actual claims and verdict answer the full question. '
                  'Do not equate an unclear verdict with a false factual claim.'),
        'items': items,
    }, ensure_ascii=False, indent=2) + '\n')
    print(f'Prepared {len(items)} local cases; independent human labels=0')


if __name__ == '__main__':
    main()
