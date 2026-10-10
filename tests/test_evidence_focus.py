from copy import deepcopy
import re
from types import SimpleNamespace

import pytest

from app.chunking import token_count
from app.evidence_focus import question_aspects, select_aspect_evidence
from app.supplement_contract import build_supplement_contract


def document(did, source='Publisher P', day='2026-01-03'):
    return SimpleNamespace(id=did, active_version_id=did+'-v1', source_sha256=did+'-sha',
                           metadata_json={'source': source, 'published_at': day})


def row(cid, text='Background.', rank=1, did=None, **kwargs):
    did = did or cid
    return dict(chunk_id=cid, document_id=did, version_id=did+'-v1', source_sha256=did+'-sha',
                title='', text=text, parent_rank=rank, lane_type='global', lane_sha256='global', **kwargs)


def score_by_text(query, rows):
    scores = [(len(set(query.lower().split()) & set(r['text'].lower().split())), r) for r in rows]
    return [dict(r, rerank_score=score) for score, r in sorted(scores, key=lambda p: -p[0])]


def test_final_topics_keep_original_entity_and_content_negation():
    q = ("According to Publisher P, discuss Ari's memoir, relationship work, separation at an awards event, "
         "pre-marital divorce views, and a commitment to remain under the same roof without separating.")
    aspects = question_aspects(build_supplement_contract(q, [document('p')]))
    assert 2 <= len(aspects) <= 8
    assert any('divorce views' in a.query for a in aspects)
    assert any('same roof without separating' in a.query for a in aspects)
    assert all('Ari' in a.query and 'Publisher P' not in a.query for a in aspects)
    assert all(a.document_versions == (('p', 'p-v1'),) for a in aspects)
    terms = set(re.findall(r"[\w'-]+", q))
    assert all(set(re.findall(r"[\w'-]+", a.query)) <= terms | {'Ari'} for a in aspects)


def test_generic_conjunction_topics_and_chinese_slots_preserve_units():
    c = build_supplement_contract("Describe Ari's memoir and divorce views and roof commitment", [])
    assert len(question_aspects(c)) == 3
    c = build_supplement_contract('白泽账务的超时阈值和青鸾网关的超时阈值分别是多少秒？', [])
    aspects = question_aspects(c)
    assert len(aspects) == 2
    assert any('白泽账务' in a.query for a in aspects)
    assert any('青鸾网关' in a.query for a in aspects)
    assert all('秒' in a.query for a in aspects)
    assert all(a.document_versions is None for a in aspects)


def test_single_word_comma_topics_keep_written_subject_context():
    c = build_supplement_contract('Discuss Product Z durability, repairability, and warranty length.', [])
    aspects = question_aspects(c)
    assert len(aspects) == 3
    assert all('Product Z' in a.query for a in aspects)
    assert any('repairability' in a.query for a in aspects)
    assert any('warranty length' in a.query for a in aspects)


def test_trailing_publisher_keeps_all_governed_content_and_exact_source_scope():
    q = "Discuss Ari's memoir and divorce views according to Publisher P."
    c = build_supplement_contract(q, [document('p')])
    aspects = question_aspects(c)
    assert aspects
    assert any('memoir' in a.query for a in aspects)
    assert any('divorce views' in a.query for a in aspects)
    assert all('Ari' in a.query and 'Publisher P' not in a.query for a in aspects)
    assert all(a.document_versions == (('p', 'p-v1'),) for a in aspects)


@pytest.mark.parametrize('q', [
    "Discuss Ari's decision not to relocate and separate.",
    "Discuss Ari's decision never to relocate and separate.",
    "Discuss Ari's decision to avoid relocating and separating.",
    "Discuss Ari's promise that the couple wouldn't relocate and separate.",
    "Discuss Ari's promise that the couple wouldn’t relocate and separate.",
    "Discuss Ari's claim that the couple isn't relocating and separating.",
    "Discuss Ari's claim that the couple isn’t relocating and separating.",
])
def test_shared_negation_is_preserved_as_one_clause(q):
    aspects = question_aspects(build_supplement_contract(q, []))
    assert len(aspects) == 1
    assert aspects[0].query == q


def test_coordinated_reports_reserve_two_dates_without_event_date_loss():
    q = ('Publisher P reports on January 3, 2026 that Ari discussed divorce views, '
         'and then on February 8, 2026, reported that Ari promised the same roof.')
    c = build_supplement_contract(q, [document('jan'), document('feb', day='2026-02-08')])
    aspects = question_aspects(c)
    assert [a.document_versions for a in aspects] == [(('jan', 'jan-v1'),), (('feb', 'feb-v1'),)]
    rows = [row('jan-topic', 'Ari discussed divorce views', did='jan'),
            row('feb-topic', 'Ari promised the same roof', did='feb')]
    chosen = select_aspect_evidence(c, rows, ranker=score_by_text, limit=2)
    assert {r['chunk_id'] for r in chosen.evidence} == {'jan-topic', 'feb-topic'}
    assert all(f['status'] == 'reserved' for f in chosen.trace['aspect_floor'])
    event = 'Publisher P reports on January 3, 2026 that Ari did not separate on May 2, 2025.'
    # Ambiguous metadata attachment stays an empty scope; content date remains query text.
    assert 'May 2, 2025' in question_aspects(build_supplement_contract(event, [document('jan')]))[0].query


def test_unknown_and_missing_sources_do_not_call_ranker_or_widen():
    c = build_supplement_contract('Publisher P article published February 8, 2026 about Ari.', [document('jan')])
    calls = []
    chosen = select_aspect_evidence(c, [row('other')], ranker=lambda *args: calls.append(args))
    assert chosen.evidence == [] and not calls
    assert all(a.document_versions == () for a in question_aspects(c))


def test_aspect_floor_preserves_whole_rows_skips_oversize_and_interleaves_scores():
    c = build_supplement_contract("Describe Ari's memoir and divorce views", [])
    rows = [row('big', 'memoir '*1000, 1), row('memoir', 'memoir full literal ' + 'x '*200, 2),
            row('divorce', 'divorce views', 3), row('fill-a', 'memoir', 4), row('fill-b', 'divorce', 5)]
    original = deepcopy(rows)
    calls = []
    def ranker(q, rs):
        calls.append((q, len(rs)))
        factor = 10000 if 'memoir' in q else 1
        return [dict(r, rerank_score=r['rerank_score']*factor) for r in score_by_text(q, rs)]
    chosen = select_aspect_evidence(c, rows, ranker=ranker, limit=4, token_budget=800)
    assert {r['chunk_id'] for r in chosen.evidence} == {'memoir', 'divorce', 'fill-a', 'fill-b'}
    assert rows == original
    for selected in chosen.evidence:
        assert selected['text'] == next(r['text'] for r in rows if r['chunk_id'] == selected['chunk_id'])
    assert chosen.context_tokens == sum(token_count(r['text']) + 100 for r in chosen.evidence)
    assert chosen.trace['ranker_calls'] == 2
    assert chosen.trace['ranker_pairs'] == sum(n for _, n in calls)
    assert chosen.trace['ranker_wall_ms'] >= 0
    assert chosen.trace['coverage_type'] == 'relevance_ranking_only'


def test_hard_exclusions_versions_sha_and_required_ids_fail_closed():
    c = build_supplement_contract('Publisher P about roof commitment', [document('p')])
    rows = [row('good', 'roof commitment', did='p'), row('stale', 'roof', did='p'),
            row('wrong-sha', 'roof', did='p'), row('blocked', 'roof', did='p', excluded_because='tenant_scope')]
    rows[1]['version_id'] = 'old'
    rows[2]['source_sha256'] = 'wrong'
    chosen = select_aspect_evidence(c, rows, ranker=score_by_text)
    assert [r['chunk_id'] for r in chosen.evidence] == ['good']
    for required in ['stale', 'wrong-sha', 'blocked', 'missing']:
        with pytest.raises(ValueError):
            select_aspect_evidence(c, rows, ranker=score_by_text, required_ids=[required])
    with pytest.raises(ValueError, match='Infeasible required'):
        select_aspect_evidence(c, rows, ranker=score_by_text, required_ids=['good'], token_budget=1)


@pytest.mark.parametrize('mutation', ['text', 'version_id', 'duplicate', 'missing', 'score', 'order', 'input'])
def test_ranker_integrity_is_validated(mutation):
    c = build_supplement_contract('Discuss the ordinary topic', [])
    rows = [row('a'), row('b', rank=2)]
    def bad(q, rs):
        out = [dict(r, rerank_score=2-i) for i, r in enumerate(rs)]
        if mutation in {'text', 'version_id'}:
            out[0][mutation] = 'invented'
        elif mutation == 'duplicate':
            out[1] = out[0]
        elif mutation == 'missing':
            out.pop()
        elif mutation == 'score':
            out[0]['rerank_score'] = float('nan')
        elif mutation == 'order':
            out.reverse()
        else:
            rs[0]['text'] = 'tampered'
        return out
    with pytest.raises(ValueError, match='ranker'):
        select_aspect_evidence(c, rows, ranker=bad)
    assert rows[0]['text'] == 'Background.'


def test_cap_and_query_length_fail_without_silent_clipping():
    c = build_supplement_contract("Describe Ari's " + ', '.join('topic '+str(i) for i in range(9)), [])
    with pytest.raises(ValueError, match='aspect'):
        question_aspects(c)
    c = build_supplement_contract('x '*260, [])
    with pytest.raises(ValueError, match='query'):
        question_aspects(c)


def test_ties_are_deterministic_and_document_quota_is_hard():
    c = build_supplement_contract("Describe Ari's memoir and divorce views", [])
    rows = [row('z', 'memoir divorce', rank=1, did='same'),
            row('a', 'memoir divorce', rank=1, did='same'), row('b', 'memoir divorce', rank=2)]
    chosen = select_aspect_evidence(c, rows, ranker=score_by_text, document_quota=1, limit=2)
    assert [r['chunk_id'] for r in chosen.evidence] == ['a', 'b']


def test_aspect_floor_uses_best_feasible_order_even_when_another_chunk_is_already_selected():
    c = build_supplement_contract("Describe Ari's memoir and divorce views", [])
    rows = [row('memoir', 'memoir', 1), row('divorce', 'divorce views', 2)]
    chosen = select_aspect_evidence(c, rows, ranker=score_by_text, limit=2)
    assert [f['chunk_id'] for f in chosen.trace['aspect_floor']] == ['memoir', 'divorce']


def test_resolved_inactive_conditional_slots_are_not_ranked_or_reserved():
    c = build_supplement_contract('回滚服务的阈值是多少？如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。', [])
    calls = []
    def ranker(q, rows):
        calls.append(q)
        return score_by_text(q, rows)
    chosen = select_aspect_evidence(c, [row('operand', '回滚服务的阈值为 9 秒。')], ranker=ranker)
    assert len(calls) == 2
    assert all('日志服务' not in q for q in calls)
    assert len(chosen.trace['inactive_slot_ids']) == 1
    calls.clear()
    chosen = select_aspect_evidence(c, [row('unknown', '回滚服务阈值待确认。')], ranker=ranker)
    assert len(calls) == 3 and chosen.trace['inactive_slot_ids'] == []


def test_excluded_condition_operand_cannot_deactivate_a_branch():
    c = build_supplement_contract('回滚服务的阈值是多少？如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。', [])
    rows = [row('blocked', '回滚服务的阈值为 9 秒。', excluded_because='tenant_scope'),
            row('other', '回滚服务阈值待确认。')]
    chosen = select_aspect_evidence(c, rows, ranker=score_by_text)
    assert chosen.trace['inactive_slot_ids'] == []


def test_condition_operand_witness_survives_a_higher_ranked_unrelated_candidate():
    from app.supplement_contract import assess_slot_presence
    c = build_supplement_contract('回滚服务的阈值是多少？如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。', [])
    rows = [row('operand', '回滚服务的阈值为 9 秒。', rank=2), row('unrelated', '无关背景。', rank=1)]
    def ranker(q, rs):
        return [dict(r, rerank_score=2 if r['chunk_id'] == 'unrelated' else 1)
                for r in sorted(rs, key=lambda r: r['chunk_id'] != 'unrelated')]
    chosen = select_aspect_evidence(c, rows, ranker=ranker, limit=1)
    assert [r['chunk_id'] for r in chosen.evidence] == ['operand']
    assert chosen.trace['inactive_slot_ids'] == ['slot_2']
    assert assess_slot_presence(c, chosen.evidence)['slots'][1]['state'] == 'inactive'
    assert chosen.trace['condition_witness_ids'] == ['operand']


@pytest.mark.parametrize('caps', [{'limit': 1}, {'token_budget': 120}, {'document_quota': 1}])
def test_infeasible_condition_witness_does_not_deactivate_branches(caps):
    c = build_supplement_contract('回滚服务的阈值是多少？如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。', [])
    rows = [row('operand', '回滚服务的阈值为 9 秒。', rank=2, did='same'),
            row('required', '无关背景。', rank=1, did='same')]
    calls = []
    def ranker(q, rs):
        calls.append(q)
        return score_by_text(q, rs)
    chosen = select_aspect_evidence(c, rows, ranker=ranker, required_ids=['required'], **caps)
    assert [r['chunk_id'] for r in chosen.evidence] == ['required']
    assert chosen.trace['inactive_slot_ids'] == []
    assert chosen.trace['condition_witness_ids'] == []
    assert len(calls) == 3


def test_conflicting_version_subset_cannot_invalidate_a_claimed_inactive_branch():
    from app.supplement_contract import assess_slot_presence
    c = build_supplement_contract('回滚服务的阈值是多少？如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。', [])
    rows = [row('good', '回滚服务的阈值为 9 秒。', rank=1),
            row('bad20', '回滚服务的阈值为 20 秒。', rank=2, did='bad'),
            row('bad30', '回滚服务的阈值为 30 秒。', rank=3, did='bad')]
    def ranker(q, rs):
        return [dict(r, rerank_score=4-r['parent_rank']) for r in rs]
    assert assess_slot_presence(c, rows)['slots'][1]['state'] == 'inactive'
    chosen = select_aspect_evidence(c, rows, ranker=ranker, limit=2)
    assert [r['chunk_id'] for r in chosen.evidence] == ['good', 'bad20']
    assert chosen.trace['inactive_slot_ids'] == []
    assert len(chosen.trace['aspect_ranking']) == 3
    assert all(s['reason'] == 'condition_unknown' for s in assess_slot_presence(c, chosen.evidence)['slots'])


def test_source_quota_and_required_overflow_remain_hard():
    c = build_supplement_contract("Describe Ari's memoir and divorce views", [])
    rows = [row('a', 'memoir', lane_name='Publisher P'), row('b', 'divorce views', rank=2)]
    for r in rows:
        r.update(lane_type='source', lane_sha256='publisher-p')
    chosen = select_aspect_evidence(c, rows, ranker=score_by_text, source_quota=1)
    assert [r['chunk_id'] for r in chosen.evidence] == ['a']
    assert chosen.context_tokens <= 5000
    with pytest.raises(ValueError, match='Infeasible required'):
        select_aspect_evidence(c, rows, ranker=score_by_text, source_quota=1, required_ids=['a', 'b'])
