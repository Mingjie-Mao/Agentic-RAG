from copy import deepcopy
from types import SimpleNamespace

from scripts.run_supplement_contract_dev import scope_keys, choose_matrix_seed, MATRIX


def test_scope_pairs_do_not_zip_independently_ordered_ids_and_versions():
    docs = [SimpleNamespace(id='z', active_version_id='v-z', metadata_json={'source':'Publisher P'}),
            SimpleNamespace(id='a', active_version_id='v-a', metadata_json={'source':'Publisher P'})]
    assert scope_keys('Publisher P report', docs, SimpleNamespace(focused_generation_enabled=False)) == [
        frozenset({('z','v-z'), ('a','v-a')})]


def test_matrix_choice_uses_native_reference_no_loss_guard():
    row = {'basis': 'source_atom_exact_proxy', 'eligible': True, 'error': None,
           'phases': {'context': {'matched_reference_ids': ['atom']}}}
    report = {'valid': True, 'rows': {p: [deepcopy(row) for _ in range(47)] for p in MATRIX},
        'summaries': {p: {'by_basis': {
            'upstream_word4_proxy': {'phases': {'context': {'delivered': 15+i}}},
            'source_atom_exact_proxy': {'phases': {'context': {'delivered': 45}}}}}
            for i,p in enumerate(MATRIX)}}
    for rows in report['rows'].values():
        rows[-1].update(eligible=False, phases={'context': None})
    report['rows']['slot_literal'][0]['phases']['context']['matched_reference_ids'] = []
    assert choose_matrix_seed(report) == 'slot_source'
    report['rows']['slot_source'][0]['phases']['context']['matched_reference_ids'] = []
    assert choose_matrix_seed(report) == 'relevant_doc'


def test_bridge_hash_uses_the_graphs_plain_string_fingerprint():
    from scripts.run_supplement_contract_dev import verify_bridge_binding
    from agent.evidence_supplement import _hash
    trace = {'bridge_chunk_sha256': _hash('source'), 'bridge_source_sha256': 'source-sha'}
    verify_bridge_binding(trace, [{'chunk_id':'source', 'source_sha256':'source-sha'}])
    import pytest
    with pytest.raises(ValueError):
        verify_bridge_binding(trace, [{'chunk_id':'source', 'source_sha256':'different'}])
