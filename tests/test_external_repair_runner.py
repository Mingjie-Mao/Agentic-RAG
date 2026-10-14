from scripts.run_external_repair_dev import focus_proposal, public_trace

import pytest


def test_focus_is_question_only_and_prefers_missing_late_topic():
    from app.supplement_contract import build_supplement_contract
    c = build_supplement_contract("Discuss Ari's memoir, divorce views, and roof commitment.", [])
    p = focus_proposal(None, c, [], [0])
    assert p.mode == 'independent'
    assert 'roof commitment' in p.query
    assert p.target_slot == 0


def test_focus_never_silently_clips_a_long_query():
    from app.supplement_contract import build_supplement_contract
    c = build_supplement_contract('word ' * 60, [])
    p = focus_proposal(None, c, [], [0])
    assert p.mode == 'unknown'


def test_public_trace_contains_hashes_not_question_or_source():
    trace = {'aspect_ranking': [{'query': 'private question', 'scores': [{'chunk_id': 'c', 'rerank_score': 1}]}],
             'ranker_calls': 1, 'ranker_pairs': 1, 'ranker_wall_ms': 2}
    out = public_trace(trace)
    assert 'private question' not in str(out)
    assert out['ranker_pairs'] == 1
    assert len(out['trace_sha256']) == 64


def test_unknown_route_target_is_not_eligible():
    from app.supplement_contract import build_supplement_contract
    c = build_supplement_contract('Discuss Ari.', [])
    assert focus_proposal(None, c, [], []).mode == 'unknown'


def test_private_outputs_reject_tracked_folder_and_symlink_escape(tmp_path, monkeypatch):
    from scripts import run_external_repair_dev as runner
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    (tmp_path / '.runtime').mkdir()
    (tmp_path / '.runtime' / 'escape').symlink_to(tmp_path)
    assert runner.private_root(tmp_path / '.runtime' / 'valid') == tmp_path / '.runtime' / 'valid'
    with pytest.raises(ValueError, match='ignored'):
        runner.private_root(tmp_path / 'artifacts')
    with pytest.raises(ValueError, match='ignored'):
        runner.private_root(tmp_path / '.runtime' / 'escape' / 'artifacts')


def test_native_no_loss_is_per_reference_not_aggregate():
    from scripts.run_external_repair_dev import choose, PROFILES
    native = {'eligible': True, 'basis': 'source_atom_exact_proxy', 'error': None,
              'phases': {'context': {'matched_reference_ids': ['a', 'b']}}}
    exchanged = dict(native, phases={'context': {'matched_reference_ids': ['a', 'c', 'd']}})
    rows = {p: [native if p == 'baseline' else exchanged] for p in PROFILES}
    summaries = {p: {'by_basis': {'upstream_word4_proxy': {'phases': {'context': {'delivered':
        15 if p == 'baseline' else 19}}}, 'source_atom_exact_proxy': {'phases': {'context':
        {'delivered': 2 if p == 'baseline' else 3}}}}} for p in PROFILES}
    assert choose({'rows': rows, 'summaries': summaries}) == 'baseline'


def test_backbone_preserves_one_content_witness_per_existing_source_without_gold():
    from scripts.run_external_repair_dev import backbone_ids
    from types import SimpleNamespace
    c = SimpleNamespace(source_requests=[], authorized_bindings=[('a', 'v', 'h'), ('b', 'v', 'h')])
    rows = [dict(chunk_id=cid, document_id=did, version_id='v', source_sha256='h',
        text=text, excluded_because=None) for cid, did, text in
        [('header', 'a', '# Story\nSource: Publisher · Published: 2026-01-01'),
         ('a1', 'a', 'A contested policy and its correction.'),
         ('a2', 'a', 'Related background.'), ('b1', 'b', 'The second source position.')]]
    assert backbone_ids(c, rows, rows) == ['a1', 'b1']


def test_rank_cache_reuses_only_identical_observed_query_source_bytes():
    from scripts.run_external_repair_dev import cached_rank
    from types import SimpleNamespace
    calls, cache = [], {}
    budget = SimpleNamespace(check=lambda: None)
    rows = [dict(chunk_id=cid, document_id=cid, version_id='v', source_sha256='h',
                 text=text, title='', rank=rank) for rank, (cid, text) in enumerate([('a', 'nine'), ('b', 'other')], 1)]
    def ranker(query, supplied):
        calls.append([r['chunk_id'] for r in supplied])
        return sorted([dict(r, rerank_score=1.0 if query in r['text'] else 0.0) for r in supplied],
                      key=lambda r: -r['rerank_score'])
    event = {}
    first = cached_rank('nine', rows, cache, ranker, budget, event)
    second = cached_rank('nine', rows, cache, ranker, budget, {})
    assert first == second and calls == [['a', 'b']]
    changed = [dict(rows[0], text='new nine'), rows[1]]
    event = {}
    cached_rank('nine', changed, cache, ranker, budget, event)
    assert calls[-1] == ['a'] and event['fresh_pairs'] == 1 and event['cached_pairs'] == 1
    cached_rank('other', changed, cache, ranker, budget, {})
    assert calls[-1] == ['a', 'b']
