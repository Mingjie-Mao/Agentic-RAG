import pytest
import json
from types import SimpleNamespace
from app.execution_budget import bounded_model
import scripts.run_external_answer_review as review
from test_external_validation import fixture
from scripts.run_external_answer_review import validate_decision, project_item


def item():
    return {'id': 'opaque', 'task': {'id': 'task', 'goal': 'What is the limit?',
        'required_facts': [{'id': 'f', 'text': 'nine'}], 'gold_evidence': [{'quote': 'private large source'}],
        'expected_status': 'answered', 'gold_answer': 'nine'},
        'answer': {'status': 'answered', 'claims': [{'text': 'nine'}],
                   'citations': [{'chunk_id': 'c', 'text': 'nine'}]},
        'retrieved_evidence': [{'chunk_id': 'c', 'text': 'nine'}]}


def test_projection_preserves_actual_evidence_once_and_hides_gold_source_and_method():
    projected = project_item(item())
    assert 'private large source' not in str(projected)
    assert projected['retrieved_evidence'][0]['text'] == 'nine'
    assert 'text' not in projected['answer']['citations'][0]
    assert 'method' not in projected


def test_incomplete_semantic_verdicts_are_rejected():
    with pytest.raises(ValueError):
        validate_decision({'fact_verdicts': {}, 'retrieved_fact_verdicts': {}}, item())


def test_interrupted_judge_retains_inflight_cost_and_checkpoint(tmp_path, monkeypatch):
    class InterruptedModels:
        def __init__(self):
            self.events = []

        @bounded_model('judge')
        def _judge_chat(self, payload):
            self.events.append({'status': 'failed', 'wall_ms': 12})
            raise KeyboardInterrupt()

    suite, raw, _ = fixture()
    raw.update(valid=True, split='dev')
    packet = review.answer_review_packet(raw, suite)
    inputs = {'packet': packet, 'raw': raw,
              'suite': suite, 'guidance': {}, 'registration': {'sealed': True}}
    paths = {}
    for name, value in inputs.items():
        path = tmp_path / (name+'.json')
        path.write_text(json.dumps(value))
        paths[name] = str(path)
    args = SimpleNamespace(**paths, private_root=str(tmp_path), output=str(tmp_path/'public.json'))
    monkeypatch.setattr(review, 'registration_for', lambda _: {'sealed': True})
    monkeypatch.setattr(review, 'private_root', lambda _: tmp_path)
    monkeypatch.setattr(review, 'MeteredModels', InterruptedModels)
    with pytest.raises(KeyboardInterrupt):
        review.execute(args, {'run_id': 'interrupted'})
    observed = json.loads((tmp_path/'interrupted/observations.json').read_text())
    assert observed['errors'] == [{'item_id': packet['items'][0]['id'], 'type': 'KeyboardInterrupt', 'interrupted': True}]
    assert observed['rows'][0]['business_budget']['calls']['judge']['attempted'] == 1
    assert observed['rows'][0]['business_budget']['calls']['judge']['failed'] == 1
    assert observed['rows'][0]['physical_judge']['attempted'] == 1
    assert (tmp_path/'interrupted/reviewed.json').exists()


def test_review_code_identity_binds_every_registered_file(tmp_path, monkeypatch):
    names = ('review.py', 'scoring.py')
    monkeypatch.setattr(review, 'code_fingerprints', lambda: {
        name: review.sha((tmp_path/name).read_bytes()) for name in names})
    for name in names:
        (tmp_path/name).write_text(name)
    before = review.code_identity()
    (tmp_path/'scoring.py').write_text('changed scoring')
    after = review.code_identity()
    assert set(before) == {'review.py', 'scoring.py'}
    assert before['review.py'] == after['review.py']
    assert before['scoring.py'] != after['scoring.py']


def test_actual_review_code_fingerprints_are_available_without_service_calls():
    fingerprints = review.code_identity()
    assert 'scripts/run_external_answer_review.py' in fingerprints
    assert 'scripts/benchmark_package_scoring.py' in fingerprints
    assert 'app/qa.py' in fingerprints
    assert all(len(value) == 64 for value in fingerprints.values())
