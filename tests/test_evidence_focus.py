from copy import deepcopy
import re
from types import SimpleNamespace

import pytest

from app.chunking import token_count
from app.evidence_focus import question_aspects, select_aspect_evidence
from app.supplement_contract import build_supplement_contract


def rank_by_parent(query, rows):
    return [dict(r, rerank_score=-r['parent_rank']) for r in rows]


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


@pytest.mark.parametrize('selection_policy', ['rank_floor', 'guarded'])
def test_hard_exclusions_versions_sha_and_required_ids_fail_closed(selection_policy):
    c = build_supplement_contract('Publisher P about roof commitment', [document('p')])
    rows = [row('good', 'roof commitment', did='p'), row('stale', 'roof', did='p'),
            row('wrong-sha', 'roof', did='p'), row('blocked', 'roof', did='p', excluded_because='tenant_scope')]
    rows[1]['version_id'] = 'old'
    rows[2]['source_sha256'] = 'wrong'
    chosen = select_aspect_evidence(c, rows, ranker=score_by_text, selection_policy=selection_policy)
    assert [r['chunk_id'] for r in chosen.evidence] == ['good']
    for required in ['stale', 'wrong-sha', 'blocked', 'missing']:
        with pytest.raises(ValueError):
            select_aspect_evidence(c, rows, ranker=score_by_text, required_ids=[required],
                                   selection_policy=selection_policy)
    with pytest.raises(ValueError, match='Infeasible required'):
        select_aspect_evidence(c, rows, ranker=score_by_text, required_ids=['good'], token_budget=1,
                               selection_policy=selection_policy)


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


@pytest.mark.parametrize('selection_policy', ['rank_floor', 'guarded'])
def test_resolved_inactive_conditional_slots_are_not_ranked_or_reserved(selection_policy):
    c = build_supplement_contract('回滚服务的阈值是多少？如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。', [])
    calls = []
    def ranker(q, rows):
        calls.append(q)
        return score_by_text(q, rows)
    chosen = select_aspect_evidence(c, [row('operand', '回滚服务的阈值为 9 秒。')], ranker=ranker,
                                   selection_policy=selection_policy)
    assert len(calls) == 2
    assert all('日志服务' not in q for q in calls)
    assert len(chosen.trace['inactive_slot_ids']) == 1
    calls.clear()
    chosen = select_aspect_evidence(c, [row('unknown', '回滚服务阈值待确认。')], ranker=ranker,
                                   selection_policy=selection_policy)
    assert len(calls) == 3 and chosen.trace['inactive_slot_ids'] == []


def test_excluded_condition_operand_cannot_deactivate_a_branch():
    c = build_supplement_contract('回滚服务的阈值是多少？如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。', [])
    rows = [row('blocked', '回滚服务的阈值为 9 秒。', excluded_because='tenant_scope'),
            row('other', '回滚服务阈值待确认。')]
    chosen = select_aspect_evidence(c, rows, ranker=score_by_text)
    assert chosen.trace['inactive_slot_ids'] == []


@pytest.mark.parametrize('selection_policy', ['rank_floor', 'guarded'])
def test_condition_operand_witness_survives_a_higher_ranked_unrelated_candidate(selection_policy):
    from app.supplement_contract import assess_slot_presence
    c = build_supplement_contract('回滚服务的阈值是多少？如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。', [])
    rows = [row('operand', '回滚服务的阈值为 9 秒。', rank=2), row('unrelated', '无关背景。', rank=1)]
    def ranker(q, rs):
        return [dict(r, rerank_score=2 if r['chunk_id'] == 'unrelated' else 1)
                for r in sorted(rs, key=lambda r: r['chunk_id'] != 'unrelated')]
    chosen = select_aspect_evidence(c, rows, ranker=ranker, limit=1, selection_policy=selection_policy)
    assert [r['chunk_id'] for r in chosen.evidence] == ['operand']
    assert chosen.trace['inactive_slot_ids'] == ['slot_2']
    assert assess_slot_presence(c, chosen.evidence)['slots'][1]['state'] == 'inactive'
    assert chosen.trace['condition_witness_ids'] == ['operand']


@pytest.mark.parametrize('caps', [{'limit': 1}, {'token_budget': 120}, {'document_quota': 1}])
@pytest.mark.parametrize('selection_policy', ['rank_floor', 'guarded'])
def test_infeasible_condition_witness_does_not_deactivate_branches(caps, selection_policy):
    c = build_supplement_contract('回滚服务的阈值是多少？如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。', [])
    rows = [row('operand', '回滚服务的阈值为 9 秒。', rank=2, did='same'),
            row('required', '无关背景。', rank=1, did='same')]
    calls = []
    def ranker(q, rs):
        calls.append(q)
        return score_by_text(q, rs)
    chosen = select_aspect_evidence(c, rows, ranker=ranker, required_ids=['required'],
                                   selection_policy=selection_policy, **caps)
    assert [r['chunk_id'] for r in chosen.evidence] == ['required']
    assert chosen.trace['inactive_slot_ids'] == []
    assert chosen.trace['condition_witness_ids'] == []
    assert len(calls) == 3


@pytest.mark.parametrize('selection_policy', ['rank_floor', 'guarded'])
def test_conflicting_version_subset_cannot_invalidate_a_claimed_inactive_branch(selection_policy):
    from app.supplement_contract import assess_slot_presence
    c = build_supplement_contract('回滚服务的阈值是多少？如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。', [])
    rows = [row('good', '回滚服务的阈值为 9 秒。', rank=1),
            row('bad20', '回滚服务的阈值为 20 秒。', rank=2, did='bad'),
            row('bad30', '回滚服务的阈值为 30 秒。', rank=3, did='bad')]
    def ranker(q, rs):
        return [dict(r, rerank_score=4-r['parent_rank']) for r in rs]
    assert assess_slot_presence(c, rows)['slots'][1]['state'] == 'inactive'
    chosen = select_aspect_evidence(c, rows, ranker=ranker, limit=2, selection_policy=selection_policy)
    assert [r['chunk_id'] for r in chosen.evidence] == ['good', 'bad20']
    assert chosen.trace['inactive_slot_ids'] == []
    assert len(chosen.trace['aspect_ranking']) == 3
    assert all(s['reason'] == 'condition_unknown' for s in assess_slot_presence(c, chosen.evidence)['slots'])


@pytest.mark.parametrize('selection_policy', ['rank_floor', 'guarded'])
def test_source_quota_and_required_overflow_remain_hard(selection_policy):
    c = build_supplement_contract("Describe Ari's memoir and divorce views", [])
    rows = [row('a', 'memoir', lane_name='Publisher P'), row('b', 'divorce views', rank=2)]
    for r in rows:
        r.update(lane_type='source', lane_sha256='publisher-p')
    chosen = select_aspect_evidence(c, rows, ranker=score_by_text, source_quota=1,
                                   selection_policy=selection_policy)
    assert [r['chunk_id'] for r in chosen.evidence] == ['a']
    assert chosen.context_tokens <= 5000
    with pytest.raises(ValueError, match='Infeasible required'):
        select_aspect_evidence(c, rows, ranker=score_by_text, source_quota=1, required_ids=['a', 'b'],
                               selection_policy=selection_policy)
def test_metadata_only_predicate_keeps_mixed_article_content():
    from app.evidence_focus import metadata_only
    assert metadata_only('# Story\n\nSource: Publisher P · Published: 2026-01-01')
    assert not metadata_only('# Story\n\nSource: Publisher P · Published: 2026-01-01\nA fact was contested.')
    assert not metadata_only('## Budget is nine units')


def test_guarded_protects_two_numeric_witnesses_below_background_ranks():
    c = build_supplement_contract('白泽账务的超时阈值和青鸾网关的超时阈值分别是多少秒？', [])
    rows = [row('background-a', '白泽账务背景。', 1), row('background-b', '青鸾网关背景。', 2),
            row('left', '白泽账务的超时阈值为 9 秒。', 3),
            row('right', '青鸾网关的超时阈值为 12 秒。', 4)]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=2, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['left', 'right']
    assert all(f['status'] == 'reserved' and f['selected_state'] == 'supported'
               for f in chosen.trace['aspect_floor'])
    default = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=2)
    explicit = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=2, selection_policy='rank_floor')
    assert default.evidence == explicit.evidence
    assert default.trace['aspect_floor'] == explicit.trace['aspect_floor']
    assert [r['chunk_id'] for r in default.evidence] == ['background-a', 'background-b']


def test_guarded_unknown_first_aspect_cannot_consume_known_witness_capacity():
    c = build_supplement_contract('白泽账务的超时阈值和青鸾网关的超时阈值分别是多少秒？', [])
    rows = [row('unknown-background', '白泽账务的超时阈值待确认。', 1),
            row('known', '青鸾网关的超时阈值为 12 秒。', 2)]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['known']
    assert chosen.trace['aspect_floor'][0]['status'] == 'unverified'
    assert chosen.trace['aspect_floor'][0]['chunk_id'] is None
    assert chosen.trace['aspect_floor'][1]['selected_state'] == 'supported'


def test_guarded_unknown_background_is_fill_without_mechanical_reservation():
    c = build_supplement_contract("Describe Ari's memoir and divorce views", [])
    chosen = select_aspect_evidence(c, [row('background', 'Ari memoir divorce views')],
                                   ranker=rank_by_parent, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['background']
    assert all(f['status'] == 'unverified' and f['chunk_id'] is None and not f['witness_ids']
               for f in chosen.trace['aspect_floor'])


def test_guarded_one_literal_chunk_can_cover_multiple_slots():
    c = build_supplement_contract('白泽账务的超时阈值和青鸾网关的超时阈值分别是多少秒？', [])
    rows = [row('both', '白泽账务的超时阈值为 9 秒。青鸾网关的超时阈值为 12 秒。')]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1, selection_policy='guarded')
    assert [f['chunk_id'] for f in chosen.trace['aspect_floor']] == ['both', 'both']


def test_guarded_exact_duplicate_background_does_not_use_an_extra_slot():
    c = build_supplement_contract('Describe ordinary context', [])
    rows = [row('a', 'Identical background.', 1, did='same'),
            row('b', 'Identical background.', 2, did='same'), row('other', 'Distinct context.', 3)]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=2, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['a', 'other']
    assert chosen.trace['candidate_count'] == 3
    rejection = next(r for r in chosen.trace['rejected'] if r['chunk_id'] == 'b')
    assert rejection['reasons'] == ['exact_duplicate_background']
    assert rejection['representative_id'] == 'a'


@pytest.mark.parametrize('different', ['document', 'version', 'source_sha'])
def test_guarded_identical_text_keeps_different_document_version_source_identity(different):
    c = build_supplement_contract('Describe ordinary context', [])
    rows = [row('a', 'Identical background.', 1, did='same'),
            row('b', 'Identical background.', 2, did='same')]
    rows[1][{'document': 'document_id', 'version': 'version_id', 'source_sha': 'source_sha256'}[different]] = 'other'
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=2, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['a', 'b']


def test_guarded_mandatory_duplicate_ids_are_kept():
    c = build_supplement_contract('Describe ordinary context', [])
    rows = [row('a', 'Identical background.', 1, did='same'),
            row('b', 'Identical background.', 2, did='same')]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, required_ids=['a', 'b'],
                                   selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['a', 'b']


def test_guarded_duplicate_can_use_real_lane_after_earlier_copy_is_infeasible():
    c = build_supplement_contract('Describe ordinary context', [])
    rows = [row('required', 'Required context.', 1, lane_name='P'),
            row('a', 'Identical background.', 2, did='same', lane_name='P'),
            row('b', 'Identical background.', 3, did='same')]
    for r in rows[:2]:
        r.update(lane_type='source', lane_sha256='publisher-p')
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, required_ids=['required'],
                                   source_quota=1, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['required', 'b']
    assert chosen.evidence[1]['lane_type'] == 'global'


@pytest.mark.parametrize('policy', ['', 'unsupported', None, [], True])
def test_bad_selection_policy_fails_before_ranker(policy):
    calls = []
    c = build_supplement_contract('Describe ordinary context', [])
    with pytest.raises(ValueError, match='policy'):
        select_aspect_evidence(c, [row('a')], ranker=lambda *args: calls.append(args), selection_policy=policy)
    assert not calls


@pytest.mark.parametrize('bad_row', [None, {}, {'text': 3}])
def test_guarded_bad_candidate_schema_fails_before_ranker(bad_row):
    calls = []
    c = build_supplement_contract('Describe ordinary context', [])
    with pytest.raises(ValueError, match='candidate'):
        select_aspect_evidence(c, [bad_row], ranker=lambda *args: calls.append(args), selection_policy='guarded')
    assert not calls


def test_guarded_value_only_table_chunk_cannot_claim_support():
    c = build_supplement_contract('Product A 的超时阈值是多少？', [])
    rows = [row('header', '| 产品 | 超时阈值 |\n| --- | --- |', 2, did='table'),
            row('value', '| Product A | 9 秒 |', 1, did='table')]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['value']
    assert chosen.trace['aspect_floor'][0]['status'] == 'unverified'
    assert chosen.trace['aspect_floor'][0]['selected_state'] != 'supported'
    complete = row('complete', '| 产品 | 超时阈值 |\n| --- | --- |\n| Product A | 9 秒 |', 3, did='table')
    chosen = select_aspect_evidence(c, [*rows, complete], ranker=rank_by_parent, limit=1, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['complete']
    assert chosen.trace['aspect_floor'][0]['selected_state'] == 'supported'


def test_guarded_fill_keeps_contradiction_visible_and_downgrades_selected_support():
    c = build_supplement_contract('Product A 的超时阈值是多少秒？', [])
    rows = [row('coherent', 'Product A 的超时阈值为 9 秒。', 3),
            row('conflict-a', 'Product A 的超时阈值为 20 秒。', 1, did='conflicting'),
            row('conflict-b', 'Product A 的超时阈值为 30 秒。', 2, did='conflicting')]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=2, selection_policy='guarded')
    assert {r['chunk_id'] for r in chosen.evidence} == {'coherent', 'conflict-a'}
    assert chosen.trace['aspect_floor'][0]['selected_state'] == 'unknown'
    assert chosen.trace['aspect_floor'][0]['status'] == 'unverified'


def test_guarded_equivalent_literal_copy_can_replace_an_oversize_pool_witness():
    c = build_supplement_contract('Product A 的超时阈值是多少秒？', [])
    rows = [row('big', 'Product A 的超时阈值为 9 秒。' + ' Background.' * 1000, 1, did='same'),
            row('background', 'Ordinary context.', 2),
            row('short', 'Product A 的超时阈值为 9 秒。', 3, did='same')]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1,
                                   token_budget=300, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['short']
    assert chosen.trace['aspect_floor'][0]['selected_state'] == 'supported'


@pytest.mark.parametrize('caps', [{'limit': 9}, {'token_budget': 5001},
                                 {'document_quota': 0}, {'source_quota': False}])
def test_guarded_invalid_caps_fail_before_ranker(caps):
    calls = []
    c = build_supplement_contract('Describe ordinary context', [])
    with pytest.raises(ValueError, match='Invalid'):
        select_aspect_evidence(c, [row('a')], ranker=lambda *args: calls.append(args),
                               selection_policy='guarded', **caps)
    assert not calls


@pytest.mark.parametrize('caps', [{'limit': 1}, {'token_budget': 120}, {'document_quota': 1},
                                 {'source_quota': 1}])
def test_guarded_literal_protection_cannot_relax_allocator_caps(caps):
    c = build_supplement_contract('Product A 的超时阈值是多少秒？', [])
    rows = [row('required', 'Required context.', 1, did='same'),
            row('literal', 'Product A 的超时阈值为 9 秒。', 2, did='same')]
    for r in rows:
        r.update(lane_type='source', lane_sha256='publisher-p')
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, required_ids=['required'],
                                   selection_policy='guarded', **caps)
    assert [r['chunk_id'] for r in chosen.evidence] == ['required']
    assert chosen.trace['aspect_floor'][0]['status'] == 'unverified'
    assert chosen.trace['aspect_floor'][0]['candidate_state'] == 'supported'
    expected_reason = {'limit': 'top_k_full', 'token_budget': 'context_budget_exhausted',
                       'document_quota': 'document_quota', 'source_quota': 'source_quota'}
    assert chosen.trace['aspect_floor'][0]['reservation_reason'] == expected_reason[next(iter(caps))]


def test_guarded_negated_numeric_background_cannot_be_reserved_as_literal():
    c = build_supplement_contract('Product A 的超时阈值是多少秒？', [])
    rows = [row('negation', 'Product A 的超时阈值不是 9 秒。')]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, selection_policy='guarded')
    assert [r['text'] for r in chosen.evidence] == [rows[0]['text']]
    assert chosen.trace['aspect_floor'][0]['selected_state'] == 'unknown'
    assert chosen.trace['aspect_floor'][0]['status'] == 'unverified'


def test_guarded_literal_protection_uses_only_real_recorded_lane_assignments():
    c = build_supplement_contract('Product A 的超时阈值是多少秒？', [])
    rows = [row('required', 'Required context.', 1),
            row('literal', 'Product A 的超时阈值为 9 秒。', 2)]
    for r in rows:
        r.update(lane_type='source', lane_sha256='publisher-p')
    alternate = dict(rows[1], lane_type='global', lane_sha256='global', parent_rank=3)
    chosen = select_aspect_evidence(c, [*rows, alternate], ranker=rank_by_parent,
                                   required_ids=['required'], source_quota=1, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['required', 'literal']
    assert chosen.evidence[1]['lane_type'] == 'global'
    assert chosen.evidence[1]['parent_rank'] == 3


def test_guarded_heading_value_dependency_requires_the_complete_original_chunk():
    c = build_supplement_contract('Product A 的超时阈值是多少秒？', [])
    rows = [row('value', '超时阈值为 9 秒。', 1, did='section'),
            row('heading', '# Product A', 2, did='section')]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1, selection_policy='guarded')
    assert chosen.trace['aspect_floor'][0]['status'] == 'unverified'
    assert not chosen.trace['aspect_floor'][0]['witness_ids']
    complete = row('complete', '# Product A\n超时阈值为 9 秒。', 3, did='section')
    chosen = select_aspect_evidence(c, [*rows, complete], ranker=rank_by_parent, limit=1,
                                   selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['complete']
    assert chosen.trace['aspect_floor'][0]['selected_state'] == 'supported'


def test_guarded_infeasible_complete_table_keeps_value_only_selected_subset_unverified():
    c = build_supplement_contract('Product A 的超时阈值是多少秒？', [])
    rows = [row('value', '| Product A | 9 秒 |', 1, did='table'),
            row('header', '| 产品 | 超时阈值 |\n| --- | --- |', 2, did='table'),
            row('complete', '| 产品 | 超时阈值 |\n| --- | --- |\n| Product A | 9 秒 |', 3, did='table')]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1,
                                   required_ids=['value'], selection_policy='guarded')
    assert chosen.trace['aspect_floor'][0]['status'] == 'unverified'
    assert chosen.trace['aspect_floor'][0]['selected_state'] != 'supported'
    assert not chosen.trace['aspect_floor'][0]['witness_ids']
    assert chosen.trace['aspect_floor'][0]['reservation_reason'] == 'top_k_full'


@pytest.mark.parametrize('text', [
    'Product A 讨论了 Product B 的超时阈值为 9 秒。',
    'Product A 的超时阈值示例为 9 秒，实际值待确认。',
])
def test_guarded_cooccurrence_and_example_values_are_unverified(text):
    c = build_supplement_contract('Product A 的超时阈值是多少秒？', [])
    rows = [row('background', 'General context.', 1), row('nonliteral', text, 2)]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['background']
    floor = chosen.trace['aspect_floor'][0]
    assert floor['candidate_state'] == 'unknown'
    assert floor['status'] == 'unverified' and floor['basis'] == 'unverified_relevance_only'
    assert floor['chunk_id'] is None and not floor['witness_ids']
    # Even a mandatory original row must not be reported as guarded support.
    mandatory = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1,
                                      required_ids=['nonliteral'], selection_policy='guarded')
    assert mandatory.evidence[0]['text'] == text
    assert mandatory.trace['aspect_floor'][0]['selected_state'] == 'unknown'
    assert mandatory.trace['aspect_floor'][0]['status'] == 'unverified'
    default = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1)
    assert [r['chunk_id'] for r in default.evidence] == ['background']
    assert default.trace['aspect_floor'][0]['basis'] == 'scoped_relevance_only'


def test_guarded_unverified_cooccurrence_operand_cannot_deactivate_conditional_branches():
    c = build_supplement_contract('Product A 的超时阈值是多少？如果超过 10 秒，再查日志服务的阈值；'
                                  '否则查通知服务的阈值。', [])
    rows = [row('background', 'General context.', 1),
            row('nonliteral', 'Product A 讨论了 Product B 的超时阈值为 9 秒。', 2)]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['background']
    assert chosen.trace['condition_witness_ids'] == []
    assert chosen.trace['inactive_slot_ids'] == []
    assert len(chosen.trace['aspect_ranking']) == 3


@pytest.mark.parametrize('text', [
    'Product A 的超时阈值为 9 秒。',
    '| 产品 | 超时阈值 |\n| --- | --- |\n| Product A | 9 秒 |',
    '# Product A\n超时阈值为 9 秒。',
])
def test_guarded_direct_table_and_heading_literals_still_protect(text):
    c = build_supplement_contract('Product A 的超时阈值是多少秒？', [])
    rows = [row('background', 'General context.', 1), row('literal', text, 2)]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['literal']
    assert chosen.evidence[0]['text'] == text
    floor = chosen.trace['aspect_floor'][0]
    assert floor['status'] == 'reserved' and floor['basis'] == 'coherent_literal'
    assert floor['selected_state'] == 'supported'


@pytest.mark.parametrize('text', [
    '假定 Product A 的超时阈值为 9 秒。',
    'Suppose Product A 的超时阈值为 9 秒。',
    '假定，Product A 的超时阈值为 9 秒。',
    'Product A 的超时阈值为 9 秒，作为演算选取。',
    'Product A 的超时阈值为 9 秒，尚未确认。',
    'Product A 的超时阈值为 9 秒 as a counterfactual.',
])
@pytest.mark.parametrize('conditional', [False, True])
def test_guarded_unknown_literal_prefix_or_suffix_cannot_establish_support_or_condition(text, conditional):
    question = 'Product A 的超时阈值是多少秒？'
    if conditional:
        question += '如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。'
    c = build_supplement_contract(question, [])
    rows = [row('background', 'General context.', 1), row('nonliteral', text, 2)]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['background']
    assert chosen.trace['condition_witness_ids'] == []
    assert chosen.trace['inactive_slot_ids'] == []
    assert all(f['candidate_state'] == 'unknown' and f['status'] == 'unverified'
               for f in chosen.trace['aspect_floor'])
    mandatory = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1,
                                      required_ids=['nonliteral'], selection_policy='guarded')
    assert mandatory.evidence[0]['text'] == text
    assert mandatory.trace['condition_witness_ids'] == []
    assert mandatory.trace['inactive_slot_ids'] == []
    assert all(f['selected_state'] == 'unknown' and not f['witness_ids']
               for f in mandatory.trace['aspect_floor'])


@pytest.mark.parametrize('text', [
    '假设\n# Product A\n超时阈值为 9 秒。',
    '假设\n| 产品 | 超时阈值 |\n| --- | --- |\n| Product A | 9 秒 |',
    '| 产品 | 超时阈值 |\n| --- | --- |\n| Product A | 9 秒（假定） |',
    '# Product A\n超时阈值为 9 秒 as a counterfactual.',
    '# Product A\n超时阈值为 9 秒，作为演算选取。',
    '| 产品 | 超时阈值 |\n| --- | --- |\n| Product A | 9 秒 as a counterfactual |',
    # A valid number in another row cannot certify the target's qualified cell.
    '| 产品 | 超时阈值 |\n| --- | --- |\n| Product B | 8 秒 |\n| Product A | 9 秒（假定） |',
    # An exact subject in a note cell is still cooccurrence, not the row subject.
    '| 产品 | 超时阈值 | 备注 |\n| --- | --- | --- |\n| Product B | 9 秒 | Product A |',
])
@pytest.mark.parametrize('conditional', [False, True])
def test_guarded_qualified_scope_and_numeric_cells_remain_unverified(text, conditional):
    question = 'Product A 的超时阈值是多少秒？'
    if conditional:
        question += '如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。'
    c = build_supplement_contract(question, [])
    rows = [row('background', 'General context.', 1), row('nonliteral', text, 2)]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['background']
    assert chosen.trace['condition_witness_ids'] == []
    assert chosen.trace['inactive_slot_ids'] == []
    assert all(f['candidate_state'] == 'unknown' and f['status'] == 'unverified'
               for f in chosen.trace['aspect_floor'])
    mandatory = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1,
                                      required_ids=['nonliteral'], selection_policy='guarded')
    assert mandatory.evidence[0]['text'] == text
    assert mandatory.trace['condition_witness_ids'] == []
    assert mandatory.trace['inactive_slot_ids'] == []
    assert all(f['selected_state'] == 'unknown' and not f['witness_ids']
               for f in mandatory.trace['aspect_floor'])


@pytest.mark.parametrize('text', [
    '# Product A\n超时阈值为 9 秒。',
    '| 产品 | 超时阈值 |\n| --- | --- |\n| Product A | 9 秒 |',
])
def test_guarded_simple_scoped_literals_keep_the_target_condition_witness(text):
    c = build_supplement_contract('Product A 的超时阈值是多少秒？如果超过 10 秒，'
                                  '再查日志服务的阈值；否则查通知服务的阈值。', [])
    rows = [row('background', 'General context.', 1), row('literal', text, 2)]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['literal']
    assert chosen.evidence[0]['text'] == text
    assert chosen.trace['condition_witness_ids'] == ['literal']
    assert chosen.trace['inactive_slot_ids'] == ['slot_2']
    assert chosen.trace['aspect_floor'][0]['selected_state'] == 'supported'


def test_guarded_multirow_table_protects_only_the_exact_subject_value_cell():
    c = build_supplement_contract('Product A 的超时阈值是多少秒？', [])
    text = '| 产品 | 超时阈值 |\n| --- | --- |\n| Product B | 18 秒 |\n| Product A | 9 秒 |'
    rows = [row('background', 'General context.', 1), row('literal', text, 2)]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['literal']
    assert chosen.trace['aspect_floor'][0]['selected_state'] == 'supported'


@pytest.mark.parametrize('text', [
    'Product A 的超时阈值为 9 秒。该阈值尚未确认。',
    '以下数据均为假定值。\nProduct A 的超时阈值为 9 秒。',
    'Assumption: the following settings are provisional.\nProduct A 的超时阈值为 9 秒。',
    'Suppose the following settings apply.\nProduct A 的超时阈值为 9 秒。',
])
@pytest.mark.parametrize('conditional', [False, True])
def test_guarded_explicit_whole_row_qualifiers_cannot_be_lost_outside_quote(text, conditional):
    question = 'Product A 的超时阈值是多少秒？'
    if conditional:
        question += '如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。'
    c = build_supplement_contract(question, [])
    rows = [row('background', 'General context.', 1), row('nonliteral', text, 2)]
    chosen = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1, selection_policy='guarded')
    assert [r['chunk_id'] for r in chosen.evidence] == ['background']
    assert chosen.trace['condition_witness_ids'] == []
    assert chosen.trace['inactive_slot_ids'] == []
    assert all(f['candidate_state'] == 'unknown' and f['status'] == 'unverified'
               for f in chosen.trace['aspect_floor'])
    mandatory = select_aspect_evidence(c, rows, ranker=rank_by_parent, limit=1,
                                      required_ids=['nonliteral'], selection_policy='guarded')
    assert mandatory.evidence[0]['text'] == text
    assert mandatory.trace['condition_witness_ids'] == []
    assert mandatory.trace['inactive_slot_ids'] == []
    assert all(f['selected_state'] == 'unknown' and not f['witness_ids']
               for f in mandatory.trace['aspect_floor'])


def test_literal_generation_strict_guard_keeps_reviewed_selector_default():
    from app.evidence_focus import _literal_guard_reason
    from app.task_contract import _slot
    spec = _slot('日志服务的状态是什么？', 'status')
    quote = '日志服务的状态未启用'
    # The reviewed selector retains its historical broad enum predicate shape.
    assert _literal_guard_reason(spec, quote, source_text=quote) is None
    assert _literal_guard_reason(spec, quote, source_text=quote, strict=True)
