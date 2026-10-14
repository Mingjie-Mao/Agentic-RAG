from copy import deepcopy
import json

import pytest

from scripts.prepare_source_human_review import (
    compare_external_reviews, prepare_external, validate_external_review,
)


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / '.runtime/source.txt'
    source.parent.mkdir()
    source.write_text('Original source: exactly nine widgets.')
    suite = {'split': 'external', 'tasks': [{'id': 'system-method-99', 'question': 'How many widgets?',
        'gold_documents': ['doc'], 'gold_answer': 'old label MUST NOT LEAK', 'required_facts': [],
        'upstream_gold_evidence': [{'title': 'Widget source', 'fact': 'old label MUST NOT LEAK'}]}]}
    spec = {'documents': [{'document_id': 'doc', 'title': 'Widget source',
                          'versions': [{'path': '.runtime/source.txt', 'effective_from': '2024-01-01'}]}]}
    for name, value in [('suite', suite), ('spec', spec)]:
        (tmp_path / (name + '.json')).write_text(json.dumps(value))
    target = tmp_path / '.runtime/external-review'
    result = prepare_external(tmp_path / 'suite.json', tmp_path / 'spec.json', target, root=tmp_path)
    packets = [json.loads((target / slot / 'packet.json').read_text()) for slot in ('reviewer-1', 'reviewer-2')]
    key = json.loads((target / 'coordinator/key.json').read_text())
    return target, result, packets, key


def completed(packet, name='Independent reviewer'):
    value = deepcopy(packet)
    value.update(status='complete', reviewer=name, reviewer_type='human', independent=True,
                 completed_at='2026-10-14T00:00:00Z', declaration=dict.fromkeys(packet['declaration'], False))
    sid = packet['sources'][0]['source_id']
    value['items'][0]['annotation'].update(status='approved', answerable=True,
        reference_answer='Nine widgets.', required_facts=[{'text': 'Nine widgets.',
        'citations': [{'source_id': sid, 'quote': 'exactly nine widgets'}]}],
        conditions='none', temporal_scope='source date', source_requirements='one source', reason='Source gives the count.')
    return value


def test_first_pass_hides_old_labels_method_identity_and_separates_coordinator(prepared):
    target, result, packets, _ = prepared
    assert result['independent_human_labels'] == 0
    for packet in packets:
        assert 'old label MUST NOT LEAK' not in json.dumps(packet)
        assert 'system-method-99' not in json.dumps(packet)
        assert packet['items'][0]['annotation']['reference_answer'] is None
        assert packet['sources'][0]['text'].startswith('Original source')
    assert packets[0]['items'][0]['id'] != packets[1]['items'][0]['id']
    assert 'old label MUST NOT LEAK' in (target / 'coordinator/original-annotations.json').read_text()


@pytest.mark.parametrize('change', ['model', 'not_independent', 'exposed', 'forged_quote', 'changed_source', 'duplicate', 'blank_reason', 'missing_condition'])
def test_invalid_or_non_independent_labels_cannot_be_accepted(prepared, change):
    _, _, packets, _ = prepared
    original = packets[0]
    value = completed(original)
    if change == 'model':
        value['reviewer_type'] = 'model'
    if change == 'not_independent':
        value['independent'] = False
    if change == 'exposed':
        value['declaration']['saw_existing_annotations'] = True
    if change == 'forged_quote':
        value['items'][0]['annotation']['required_facts'][0]['citations'][0]['quote'] = 'ten widgets'
    if change == 'changed_source':
        value['sources'][0]['text'] = 'Different source'
    if change == 'duplicate':
        value['items'].append(deepcopy(value['items'][0]))
    if change == 'blank_reason':
        value['items'][0]['annotation']['reason'] = ' '
    if change == 'missing_condition':
        value['items'][0]['annotation']['conditions'] = None
    with pytest.raises(ValueError):
        validate_external_review(original, value)


def test_unclear_is_preserved_and_agreement_does_not_automatically_create_gold(prepared):
    _, _, packets, key = prepared
    a, b = completed(packets[0], 'Reviewer One'), completed(packets[1], 'Reviewer Two')
    a['items'][0]['annotation'].update(status='unclear', answerable=None, reference_answer=None,
                                     required_facts=[], reason='Time relationship unresolved')
    assert next(iter(validate_external_review(packets[0], a).values()))['status'] == 'unclear'
    comparison = compare_external_reviews(a, b, key, originals=packets)
    assert comparison['status'] == 'awaiting_adjudication'
    assert comparison['promoted_gold_labels'] == 0
    assert comparison['disagreements'][0]['different_fields']


def test_same_reviewer_is_not_two_independent_passes_and_no_overwrites(prepared):
    target, _, packets, key = prepared
    with pytest.raises(ValueError, match='distinct'):
        compare_external_reviews(completed(packets[0]), completed(packets[1]), key, originals=packets)
    with pytest.raises(ValueError, match='preserve'):
        prepare_external(target.parent.parent / 'suite.json', target.parent.parent / 'spec.json', target,
                         root=target.parent.parent)


def test_third_party_packet_must_be_local(tmp_path):
    with pytest.raises(ValueError, match='local'):
        prepare_external(tmp_path / 'suite.json', tmp_path / 'spec.json', tmp_path / 'public', root=tmp_path)
