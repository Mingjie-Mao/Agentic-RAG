from copy import deepcopy

import pytest

from scripts import run_reviewed_supplement_dev as runner
from scripts.research_review import digest
from scripts.run_revised_evidence_dev import RELEVANCE_PROFILES
from scripts.dev_evidence_annotations import compile_annotation

import test_dev_evidence_annotations as annotation_tests

inputs = annotation_tests.inputs


def packet():
    tasks = [{'id': f'D{i}'} for i in range(47)]
    suite, parent, manifest = {'tasks': tasks}, {'source': 'bound'}, {'overlay': 'bound'}
    rows = [dict(task_id=t['id'], eligible=i < 36, basis='source_atom_exact_proxy'
                 if i < 26 else 'upstream_word4_proxy', error=None,
                 phases={'context': {'matched_reference_ids': [f'a{i}']}})
            for i, t in enumerate(tasks)]
    seed = {'version': 'revised-evidence-dev-v1', 'valid': True, 'status': 'complete',
            'selected_task_ids': [t['id'] for t in tasks], 'chosen_seed_profile': 'legacy_rerank',
            'identity': {'stage': 'relevance', 'parent_sha256': digest(parent),
                         'annotation_manifest': manifest, 'profiles': list(RELEVANCE_PROFILES)},
            'rows': {p: deepcopy(rows) for p in RELEVANCE_PROFILES},
            'summaries': {p: {'by_basis': {
                'source_atom_exact_proxy': {'phases': {'context': {'delivered': 45}}},
                'upstream_word4_proxy': {'phases': {'context': {'delivered': 13}}}}}
                          for p in RELEVANCE_PROFILES}}
    return seed, parent, suite, manifest


@pytest.mark.parametrize('change', ['overlay', 'parent', 'partial', 'choice', 'drift', 'duplicate'])
def test_reviewed_seed_rejects_forged_or_partial_identity(change):
    seed, parent, suite, manifest = packet()
    if change == 'overlay':
        manifest = {'overlay': 'changed'}
    elif change == 'parent':
        parent['source'] = 'changed'
    elif change == 'partial':
        seed['rows']['relevant_doc'].pop()
    elif change == 'choice':
        seed['chosen_seed_profile'] = 'relevant_both'
    elif change == 'drift':
        seed['drift'] = {'code': True}
    else:
        seed['rows']['relevant_doc'][1]['task_id'] = 'D0'
    with pytest.raises(ValueError):
        runner.validate_seed(seed, parent, suite, manifest)


def test_complete_frozen_seed_is_accepted():
    assert runner.validate_seed(*packet()) == 'legacy_rerank'


@pytest.mark.parametrize('change', ['input', 'budget', 'extra', 'profile'])
def test_registration_is_exact_and_not_self_rebuilt(change):
    inputs = [b'one', b'two']
    registration = {'version': 'reviewed-supplement-input-registration-v1', 'split': 'dev',
        'input_bytes_sha256': list(map(runner.sha, inputs)), 'seed_profile': 'relevant_doc',
        'limits': deepcopy(runner.LIMITS), 'union': 'old_new_round_robin_rank_only',
        'control_order': 'original_task_index_parity'}
    runner.validate_registration(registration, inputs, 'relevant_doc')
    if change == 'input':
        inputs[0] = b'changed'
    elif change == 'budget':
        registration['limits']['shared_policy_calls'] = 3
    elif change == 'extra':
        registration['private'] = 'must not escape'
    else:
        registration['seed_profile'] = 'legacy_rerank'
    with pytest.raises(ValueError):
        runner.validate_registration(registration, inputs, 'relevant_doc')


def test_missing_reviewed_arm_preserves_unknown_phases_and_physical_usage(inputs):
    suite, draft, review, root, suite_bytes, draft_sha = inputs
    overlay = compile_annotation(suite, draft, review, root,
        draft_sha256=draft_sha, suite_bytes_sha256=runner.sha(suite_bytes))
    row = {'error': {'type': 'Interrupted', 'stage': 'search'}, 'telemetry': None,
           'usage': {'search_attribution': 'unknown', 'additional_retrieval_tools': None}}
    scored = runner.rescore(None, None, suite['tasks'][0], row, {}, overlay['records'][0])
    assert scored['eligible'] and scored['status'] == 'error'
    assert all(value is None for value in scored['phases'].values())
    assert scored['losses'] is None
    assert scored['usage']['additional_retrieval_tools'] is None
