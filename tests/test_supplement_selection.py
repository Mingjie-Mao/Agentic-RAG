from app.supplement_contract import build_supplement_contract, assess_slot_presence, diagnose_slot_loss
from app.evidence_selection import select_slot_evidence


def row(cid, text, rank):
    return dict(chunk_id=cid, document_id=cid, version_id=cid+'-v1', source_sha256=cid+'-sha',
                title='', text=text, parent_rank=rank, lane_type='global', lane_sha256='global')


def test_critical_witness_replaces_weak_ranked_background_within_cap():
    c = build_supplement_contract('Product A 的超时阈值是多少？', [])
    rows = [row('weak', 'Product A background.', 1), row('needed', 'Product A 的超时阈值为 9 秒。', 2)]
    chosen = select_slot_evidence(c, rows, limit=1, token_budget=500, strategy='literal')
    assert [r['chunk_id'] for r in chosen.evidence] == ['needed']
    p = assess_slot_presence(c, rows)
    loss = diagnose_slot_loss(p, assess_slot_presence(c, [rows[0]]))
    assert loss[0]['status'] == 'selection_loss' and loss[0]['avoidable'] is None
    loss = diagnose_slot_loss(p, assess_slot_presence(c, [rows[0]]), feasible_slots={'slot_1'})
    assert loss[0]['avoidable'] is True


def test_oversize_witness_is_not_forced_through_budget():
    c = build_supplement_contract('Product A 的超时阈值是多少？', [])
    rows = [row('weak', 'Background', 1), row('big', 'Product A 的超时阈值为 9 秒。'+'long '*1000, 2)]
    chosen = select_slot_evidence(c, rows, limit=1, token_budget=150, strategy='literal')
    assert chosen.context_tokens <= 150
    assert chosen.trace['slot_floor'][0]['status'] == 'infeasible'


def test_unknown_presence_is_not_candidate_absence():
    c = build_supplement_contract('What is the position on privacy?', [])
    p = assess_slot_presence(c, [])
    assert diagnose_slot_loss(p, p)[0]['status'] == 'unresolved'


def test_candidate_absence_means_inspected_pool_only():
    c = build_supplement_contract('Product A 的超时阈值是多少？', [])
    p = assess_slot_presence(c, [])
    loss = diagnose_slot_loss(p, p)[0]
    assert loss['status'] == 'candidate_absent'
    assert loss['basis'] == 'inspected_pool_only'


def test_focused_subquestion_ranks_scoped_candidates_without_certifying_semantics():
    c = build_supplement_contract('What does the privacy article say?', [])
    rows = [row('background', 'What does the privacy article say? Background on privacy.', 1),
            row('critical', 'The filmmaker received approval to film.', 2)]
    selected = select_slot_evidence(c, rows, limit=1, token_budget=500,
        focus_queries={'slot_1': 'filmmaker approval to film'})
    assert selected.evidence[0]['chunk_id'] == 'critical'
    assert assess_slot_presence(c, selected.evidence)['slots'][0]['state'] == 'unknown'


def test_focused_query_cannot_invent_slot_or_budget():
    import pytest
    c = build_supplement_contract('What does the privacy article say?', [])
    with pytest.raises(ValueError):
        select_slot_evidence(c, [], focus_queries={'gold_slot': 'answer'})
