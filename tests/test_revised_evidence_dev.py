from copy import deepcopy

import pytest

from scripts import run_revised_evidence_dev as runner
from scripts.research_review import digest
import test_evidence_selection_dev as replay_tests

frozen = replay_tests.frozen


def history(parent):
    return {'valid': True, 'status': 'complete', 'split': 'dev', 'subset': False,
            'selected_task_ids': parent['selected_task_ids'], 'error': None,
            'identity': {'parent_sha256': digest(parent)},
            'rows': {'old': deepcopy(parent['rows']['depth50_rerank'])}}


@pytest.mark.parametrize('change', ['partial', 'error', 'source', 'ids', 'invalid', 'duplicate'])
def test_historical_contexts_fail_closed(frozen, change):
    h = history(frozen.parent)
    if change == 'partial':
        h['rows']['old'].pop()
    elif change == 'error':
        h['rows']['old'][0]['error'] = {'type': 'Private'}
    elif change == 'source':
        h['identity']['parent_sha256'] = 'wrong'
    elif change == 'ids':
        h['selected_task_ids'] = h['selected_task_ids'][::-1]
    elif change == 'invalid':
        h['valid'] = False
    else:
        h['rows']['legacy_rerank'] = deepcopy(h['rows']['old'])
        h['rows']['legacy_rerank'][0]['telemetry']['context'] = []
    with pytest.raises(ValueError):
        runner.cached_contexts(frozen.parent, [h])


def test_cached_contexts_keep_every_task_and_identical_duplicate(frozen):
    h = history(frozen.parent)
    sources = runner.cached_contexts(frozen.parent, [h, deepcopy(h)])
    assert list(sources) == ['legacy_rerank', 'old']
    assert len(sources['old']) == 47


def test_candidate_pool_change_cannot_be_scored(monkeypatch, frozen):
    row = frozen.parent['rows']['depth50_rerank'][0]
    monkeypatch.setattr(runner, 'rehydrate_task', lambda *args: frozen.phases)
    monkeypatch.setattr(runner, 'verify_legacy', lambda *args: None)
    monkeypatch.setattr(runner, 'rebind_rows', lambda *args: [])
    with pytest.raises(ValueError, match='candidate pool'):
        runner.score_context(None, None, frozen.suite['tasks'][0], row, row,
                             frozen.corpus, {})


def test_seed_rule_cannot_trade_native_reference_loss_for_upstream_gain():
    rows = [dict(task_id=f'D{i}', eligible=i < 36, basis='source_atom_exact_proxy'
                 if i < 26 else 'upstream_word4_proxy', error=None,
                 phases={'context': {'matched_reference_ids': [f'atom-{i}']}}) for i in range(47)]
    report = {'valid': True, 'identity': {'profiles': list(runner.RELEVANCE_PROFILES)},
              'rows': {p: deepcopy(rows) for p in runner.RELEVANCE_PROFILES},
              'summaries': {p: {'by_basis': {
                  'source_atom_exact_proxy': {'phases': {'context': {'delivered': 45}}},
                  'upstream_word4_proxy': {'phases': {'context': {'delivered': 13}}}}}
                            for p in runner.RELEVANCE_PROFILES}}
    report['rows']['relevant_both'][0]['phases']['context']['matched_reference_ids'] = []
    report['summaries']['relevant_both']['by_basis']['upstream_word4_proxy']['phases']['context']['delivered'] = 19
    assert runner.choose_seed(report) == 'legacy_rerank'
    report['summaries']['relevant_doc']['by_basis']['upstream_word4_proxy']['phases']['context']['delivered'] = 15
    assert runner.choose_seed(report) == 'relevant_doc'
