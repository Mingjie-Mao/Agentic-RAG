import json
from pathlib import Path
import sys

import pytest
from fastapi import HTTPException

from app.clients import Models, evidence_spans
from app.config import settings
from app.date_comparison import date_comparison
from app.models import Chunk, Document
from app.passages import prepare_passages
from app.routing import execution_route
from app.source_facts import PartialFactAnswer, validate_facts
from app.temporal import effective_interval, select_effective_versions
from test_agent import agent_db
from test_source_facts import evidence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_source_human_review import audited_labels


def test_window_adds_missing_exact_field_with_separate_citation_and_budget(monkeypatch):
    db, user = agent_db()
    db.add(Chunk(id='c3', version_id='v2', ordinal=1,
                 text='回滚阈值为 2%；event_id 为 OPS-87。', locator={}))
    db.commit()
    cfg = settings()
    rows = [dict(evidence('RPO 为 15 分钟。'), chunk_id='c2', document_id='doc-a', version_id='v2')]
    selected, trace = prepare_passages(db, user, '回滚阈值和 event_id 是多少？', rows, cfg)
    assert 'c3' in trace['added_chunk_ids']
    assert any('2%' in row['text'] and row['chunk_id'] == 'c3' for row in selected)
    assert trace['context_tokens'] <= cfg.context_token_budget
    monkeypatch.setattr(cfg, 'context_token_budget', 1)
    assert prepare_passages(db, user, '阈值', rows, cfg)[0] == []


def test_window_does_not_cross_version_and_rechecks_revocation():
    db, user = agent_db()
    rows = [dict(evidence('RPO 为 15 分钟。'), chunk_id='c2', document_id='doc-a', version_id='v2')]
    selected, _ = prepare_passages(db, user, 'RPO', rows, settings())
    assert all(row['version_id'] == 'v2' for row in selected)
    db.get(Document, 'doc-a').owner_id = 'other'
    db.get(Document, 'doc-a').read_groups = ['finance']
    db.commit()
    with pytest.raises(HTTPException):
        prepare_passages(db, user, 'RPO', rows, settings())


def test_optional_metadata_cannot_erase_valid_literal_value_or_invent_time():
    rows = [evidence('Policy limit is 15 minutes.')]
    sources, _ = evidence_spans(rows)
    parsed = PartialFactAnswer(answerable=True, facts=[{
        'source_id': 'E1:S1', 'attribute': 'limit', 'value': '15 minutes', 'kind': 'number',
        'subject': 'Invented department', 'time_source_id': 'E1:S1',
        'time_value': '2026-01-01', 'time_role': 'effective'}])
    facts, rejected = validate_facts(parsed, sources, rows, partial=True)
    assert len(facts) == 1 and not rejected
    assert facts[0]['subject'] == '' and facts[0]['time']['iso'] is None
    assert facts[0]['quote'] == rows[0]['text'] and facts[0]['field_warnings']
    parsed.facts[0].value = '50'
    facts, rejected = validate_facts(parsed, sources, rows, partial=True)
    assert not facts and rejected


def test_partial_extraction_preserves_reported_uncertain_source():
    rows = [evidence('Analyst Dana said costs may fall.')]
    sources, _ = evidence_spans(rows)
    parsed = PartialFactAnswer(answerable=True, facts=[{
        'source_id': 'E1:S1', 'attribute': 'costs', 'value': 'fall', 'kind': 'text'}])
    facts, rejected = validate_facts(parsed, sources, rows, partial=True)
    assert not rejected and facts[0]['modality'] == 'uncertain'
    assert 'Dana said' in facts[0]['claim']['text']


def test_failed_extraction_falls_back_once_and_accounts_for_both_calls(monkeypatch):
    monkeypatch.setattr(settings(), 'source_facts_enabled', True)
    monkeypatch.setattr(settings(), 'source_facts_protocol', 'partial')
    monkeypatch.setattr(settings(), 'source_facts_fallback_enabled', True)
    calls = []
    def chat(body):
        calls.append(body)
        content = {'answerable': False, 'facts': []} if len(calls) == 1 else {
            'status': 'answered', 'claims': [{'text': 'Policy limit is 15 minutes.',
                                             'source_ids': ['E1:S1']}]}
        return {'message': {'content': json.dumps(content)}, 'prompt_eval_count': 10, 'eval_count': 5}
    answer, usage = Models(chat).generate('What is the policy limit?', [evidence('Policy limit is 15 minutes.')])
    assert answer.answerable and len(calls) == 2 and usage['fallback_calls'] == 1
    assert usage['prompt_tokens'] == 20 and usage['completion_tokens'] == 10


def test_extraction_fallback_keeps_date_relation_and_scope_audit_metadata(monkeypatch):
    for key, value in {'source_facts_enabled': True, 'source_facts_protocol': 'partial',
                       'source_facts_fallback_enabled': True, 'focused_generation_enabled': True}.items():
        monkeypatch.setattr(settings(), key, value)
    calls = []
    def chat(body):
        calls.append(body)
        content = {'answerable': False, 'facts': []} if len(calls) == 1 else {
            'status': 'answered', 'claims': [{'text': 'The opening dates differ.',
                                             'source_ids': ['E1:S1', 'E2:S1']}]}
        return {'message': {'content': json.dumps(content)}, 'prompt_eval_count': 10, 'eval_count': 5}
    rows = [evidence('East store opens June 4, 2026 at 09:00.', 1),
            evidence('East store opens June 4, 2026 at 10:00.', 2)]
    answer, usage = Models(chat).generate('Are the opening dates consistent despite different operating hours?', rows)
    assert len(calls) == 2 and answer.answerable
    assert usage['deterministic_date_comparison']['value'] == 'yes'
    assert 'scope_repairs' in usage and usage['fallback_calls'] == 1
    assert usage['prompt_tokens'] == 20 and usage['completion_tokens'] == 10


def test_date_equality_ignores_hours_but_rejects_speaker_or_subject_mismatch():
    question = 'Are the opening dates consistent despite different operating hours?'
    rows = [evidence('East store opens June 4, 2026 at 09:00.', 1),
            evidence('East store opens June 4, 2026 at 10:00.', 2)]
    sources, _ = evidence_spans(rows)
    assert date_comparison(question, sources, rows)['value'] == 'yes'
    rows[1]['text'] = 'West store opens June 4, 2026.'
    assert date_comparison(question, evidence_spans(rows)[0], rows) is None
    rows[1]['text'] = 'Analyst Dana claimed East store opens June 4, 2026.'
    assert date_comparison(question, evidence_spans(rows)[0], rows) is None


def test_explicit_interval_exclusive_end_and_no_publication_fallback():
    old = effective_interval('Published January 1, 2026; effective March 1, 2026 until April 1, 2026.')
    new = effective_interval('Published March 1, 2026; effective April 1, 2026 through May 1, 2026.')
    assert old['valid_from'] == '2026-03-01' and not old['end_inclusive']
    versions = [{'version_id': 'v1', 'effective_interval': old},
                {'version_id': 'v2', 'effective_interval': new}]
    result = select_effective_versions('What applied on April 1, 2026?', versions)
    assert result['selections'][0]['version_id'] == 'v2'
    assert select_effective_versions('What applied on January 2, 2026?', versions)['status'] == 'unresolved'
    assert effective_interval('Published March 1, 2026.')['status'] == 'unknown'
    assert effective_interval('Effective March 1, 2026.')['status'] == 'unknown'


def test_overlapping_intervals_are_unresolved_not_newest_wins():
    interval = effective_interval('Effective March 1, 2026 through May 1, 2026.')
    rows = [{'version_id': value, 'effective_interval': interval} for value in ('v1', 'v2')]
    assert select_effective_versions('What applied on April 1, 2026?', rows)['status'] == 'unresolved'


def test_route_simple_compound_history_and_observation_dependency():
    assert execution_route('What is the RPO?')['direct']
    assert execution_route('What is the RPO and what is the RTO?')['reason'] == 'fixed_compound_workflow'
    assert execution_route('Compare historical versions')['reason'] == 'explicit_version_workflow'
    assert execution_route('If search finds a component then inspect its policy')['mode'] == 'hybrid'
    assert execution_route('Are opening dates consistent across these two notices?')['reason'] == 'fixed_compound_workflow'


def test_subject_release_noun_is_not_a_second_date_role():
    rows = [evidence('Cedar release opens June 4, 2026 at 09:00.', 1),
            evidence('Cedar release opens June 4, 2026 at 10:00.', 2)]
    source, _ = evidence_spans(rows)
    result = date_comparison('Are the opening dates consistent across these two notices?', source, rows)
    assert result['value'] == 'yes'


def test_unlabelled_or_assistant_review_cannot_claim_human_calibration():
    with pytest.raises(ValueError):
        audited_labels({'reviewer': None, 'reviewer_type': 'human', 'items': []})
    with pytest.raises(ValueError):
        audited_labels({'reviewer': 'assistant', 'reviewer_type': 'assistant', 'items': []})


def test_targeted_passage_scan_finds_far_field_without_reading_new_documents():
    db, user = agent_db()
    db.add(Chunk(id='c9', version_id='v2', ordinal=9,
                 text='event_id 是 OPS-901；回滚阈值是 2%。', locator={}))
    db.commit()
    rows = [dict(evidence('RPO 为 15 分钟。'), chunk_id='c2', document_id='doc-a', version_id='v2')]
    result, trace = prepare_passages(db, user, 'event_id 和回滚阈值是多少？', rows, settings())
    assert 'c9' in trace['added_chunk_ids']
    assert {row['document_id'] for row in result} == {'doc-a'}
    assert trace['scanned_chunks'] <= settings().passage_scan_limit


def test_repeated_publication_and_effective_date_and_inclusive_single_day():
    interval = effective_interval('Published March 1, 2026. Effective March 1, 2026 through March 1, 2026.')
    assert interval['status'] == 'explicit'
    assert interval['valid_from'] == interval['valid_to'] == '2026-03-01'
    interval = effective_interval('Effective March 1, 2026 until April 1, 2026 inclusive.')
    assert interval['end_inclusive']


def test_event_date_does_not_filter_publication_date_scope():
    from app.retrieval import route_sources
    from test_source_routing import article
    rows = [article('a', 'Example News', '2026-03-01T08:00:00Z'),
            article('b', 'Example News', '2026-03-02T08:00:00Z')]
    groups = route_sources('What did Example News say about the incident on March 1, 2026?', rows, strict_dates=True)
    assert not groups[0]['dated'] and groups[0]['key'] == ['a', 'b']
    groups = route_sources('What did Example News report on March 1, 2026?', rows, strict_dates=True)
    assert groups[0]['dated'] and groups[0]['key'] == ['a']


def test_numeric_binding_rejects_intervening_value():
    rows = [evidence('Policy limit is 30 minutes, while the reserve is 15 minutes.')]
    source, _ = evidence_spans(rows)
    answer = PartialFactAnswer(answerable=True, facts=[{
        'source_id': 'E1:S1', 'attribute': 'limit', 'value': '15 minutes', 'kind': 'number'}])
    assert not validate_facts(answer, source, rows, partial=True)[0]


def test_two_dates_cannot_settle_explicit_three_source_comparison():
    rows = [evidence('East store opens June 4, 2026.', 1),
            evidence('East store opens June 4, 2026.', 2)]
    source, _ = evidence_spans(rows)
    assert date_comparison('Are the opening dates in all reports consistent?', source, rows) is None
    assert date_comparison('Are the opening dates in Alpha, Beta and Gamma consistent?', source, rows) is None
    assert date_comparison('Are the opening dates in Three notices consistent?', source, rows) is None
    assert date_comparison('Are the opening dates in all documents consistent?', source, rows) is None


def test_targeted_field_replaces_low_relevance_row_when_context_has_eight_slots():
    db, user = agent_db()
    chunks = [db.get(Chunk, 'c2')]
    for ordinal in range(1, 8):
        part = Chunk(id=f'near-{ordinal}', version_id='v2', ordinal=ordinal,
                     text='RPO 为 15 分钟。', locator={})
        db.add(part)
        chunks.append(part)
    db.add(Chunk(id='far-id', version_id='v2', ordinal=40,
                 text='event_id 是 OPS-901。', locator={}))
    db.commit()
    rows = [dict(evidence(part.text), chunk_id=part.id, document_id='doc-a', version_id='v2')
            for part in chunks]
    result, trace = prepare_passages(db, user, 'event_id 是什么？', rows, settings())
    assert len(result) == 8
    assert 'far-id' in trace['added_chunk_ids']
    assert any(row['chunk_id'] == 'far-id' for row in result)


def test_small_extra_budget_is_shared_between_observed_versions():
    db, user = agent_db()
    for version in ('v1', 'v2'):
        db.add_all([Chunk(id=f'{version}-adjacent', version_id=version, ordinal=1,
                          text='邻段说明。', locator={}),
                    Chunk(id=f'{version}-field', version_id=version, ordinal=30,
                          text='event_id 是 OPS-901。', locator={})])
    db.commit()
    rows = [dict(evidence('RPO 为 15 分钟。'), chunk_id=chunk,
                 document_id='doc-a', version_id=version)
            for chunk, version in [('c1', 'v1'), ('c2', 'v2')]]
    result, trace = prepare_passages(db, user, 'event_id 是什么？', rows, settings(), historical=True)
    assert set(trace['added_chunk_ids']) == {'v1-field', 'v2-field'}
    assert {row['version_id'] for row in result} == {'v1', 'v2'}
