from copy import deepcopy
import math

import pytest

from app.evidence_selection import select_relevant_evidence


def row(cid, score, rank, doc=None, **extra):
    return dict(chunk_id=cid, document_id=doc or cid, version_id='v',
                source_sha256='source', title='title', text='Complete relevant sentence.',
                rerank_score=score, rank=rank, parent_rank=rank,
                lane_type='global', lane_sha256='global', **extra)


def ids(result):
    return [r['chunk_id'] for r in result.evidence]


def test_relevance_outranks_novel_background_and_preserves_input():
    rows = [row('background', -4, 1), row('good', 3, 8)]
    before = deepcopy(rows)
    assert ids(select_relevant_evidence(rows, limit=1)) == ['good']
    assert rows == before


def test_source_floor_reserved_inside_budget_and_infeasible_is_explicit():
    rows = [row('a', 10, 1), row('b', 9, 2), row('c', -2, 3)]
    result = select_relevant_evidence(rows, limit=2, source_groups={'source-c': ['c'], 'empty': []})
    assert set(ids(result)) == {'a', 'c'}
    assert result.trace['source_floor'][1]['status'] == 'no_eligible_candidate'
    assert len(result.evidence) <= 2 and result.context_tokens <= 5000


@pytest.mark.parametrize('score', [None, True, math.inf, math.nan, '2'])
def test_unknown_or_invalid_relevance_fails_closed(score):
    with pytest.raises(ValueError):
        select_relevant_evidence([row('a', score, 1)])


def test_hard_exclusions_stay_blocked_and_required_cannot_override():
    rows = [row('a', 10, 1, excluded_because='boilerplate_only'), row('b', 0, 2)]
    assert ids(select_relevant_evidence(rows)) == ['b']
    with pytest.raises(ValueError):
        select_relevant_evidence(rows, required_ids=['a'])


def test_real_alternate_lane_can_satisfy_source_cap():
    a = row('a', 10, 1)
    a.update(lane_type='source', lane_sha256='x')
    b = row('b', 9, 2)
    b.update(lane_type='source', lane_sha256='x')
    assert ids(select_relevant_evidence([a, b], source_quota=1)) == ['a']
    assert ids(select_relevant_evidence([a, b, row('b', 9, 3)], source_quota=1)) == ['a', 'b']


def test_document_cap_relaxation_and_exact_cost_boundary():
    rows = [row('a', 10, 1, 'd'), row('b', 9, 2, 'd'), row('c', 8, 3, 'd')]
    assert len(select_relevant_evidence(rows, document_quota=2).evidence) == 2
    result = select_relevant_evidence(rows, document_quota=None)
    assert len(result.evidence) == 3
    assert len(select_relevant_evidence(rows[:1], token_budget=result.context_tokens // 3).evidence) == 1
    assert not select_relevant_evidence(rows[:1], token_budget=result.context_tokens // 3 - 1).evidence


def test_same_source_reservation_can_reuse_required_chunk_and_ties_are_stable():
    rows = [row('b', 1, 1), row('a', 1, 1)]
    result = select_relevant_evidence(rows, limit=1, required_ids=['a'], source_groups={'g': ['a']})
    assert ids(result) == ['a']
    assert result.trace['source_floor'][0]['status'] == 'reserved'
