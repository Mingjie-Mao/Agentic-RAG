"""Synthetic-only component checks. No benchmark assets or services are used."""
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from agent.tools import SearchArgs
from agent.supplement_repair import run_repair
from app.execution_budget import BudgetExceeded, ExecutionBudget
from app.evidence_focus import select_aspect_evidence
from app.supplement_contract import build_supplement_contract


def doc(name='p', source='Publisher P'):
    return SimpleNamespace(id=name, active_version_id=name+'-v1', source_sha256=name+'-sha',
                           metadata_json={'source': source, 'published_at': '2026-02-03'})


def row(name='p', text='Order Z-7 carrier Zephyr. Phone unknown.', *, cid=None, rank=1):
    return dict(chunk_id=cid or name, document_id=name, version_id=name+'-v1',
                source_sha256=name+'-sha', title='', text=text, parent_rank=rank,
                lane_type='global', lane_sha256='global')


def budget(**limits):
    return ExecutionBudget(SimpleNamespace(agent_task_timeout_seconds=180,
        agent_policy_max_calls=2, agent_generation_max_calls=0,
        agent_judge_max_calls=0, agent_judge_token_budget=0), saved={'limits': {
            'policy': 2, 'search': 1, 'embedding': 1, 'ranker': 8, 'generation': 0,
            'judge': 0, 'judge_tokens': 0, **limits}})


def rank(q, rows):
    return sorted([dict(r, rerank_score=-r['parent_rank']) for r in rows],
                  key=lambda r: -r['rerank_score'])


class Policy:
    def __init__(self, proposal=None, entity='Zephyr'):
        self.proposal = proposal or {'mode': 'independent', 'target_slot': 0,
                                    'query': 'Z-7 carrier phone'}
        self.bodies = []
        self.entity = entity

    def _post(self, path, body, **kwargs):
        self.bodies.append(body)
        return {'message': {'content': json.dumps(self.proposal) if len(self.bodies) == 1
                            else self.entity}}


def test_focused_rank_uses_query_without_mutating_original_source_contract():
    c = build_supplement_contract('Discuss Zephyr durability and warranty according to Publisher P.',
                                  [doc(), doc('q', 'Publisher Q')])
    original = deepcopy(c)
    seen = []
    def recording(query, rows):
        seen.append((query, deepcopy(rows)))
        return rank(query, rows)
    result = select_aspect_evidence(c, [row(), row('q')], ranker=recording,
                                   focus_queries={'slot_1': 'Zephyr warranty'})
    assert seen and all(q == 'Zephyr warranty' for q, _ in seen)
    assert all({r['document_id'] for r in rs} == {'p'} for _, rs in seen)
    assert c == original
    assert all(r['query'] == 'Zephyr warranty' and r['original_query'] != r['query']
               for r in result.trace['aspect_ranking'])
    assert all(r['document_versions'] == (('p', 'p-v1'),)
               for r in result.trace['aspect_ranking'])


@pytest.mark.parametrize('invalid', [{'slot_77': 'safe query'}, {'slot_1': ''},
    {'slot_1': 'x'}, {'slot_1': 'x'*201}, {'slot_1': '{entity}'}, {'slot_1': 42}, []])
def test_bad_focus_is_rejected_before_any_rank(invalid):
    c = build_supplement_contract('What is order Z-7 carrier phone?', [])
    seen = []
    with pytest.raises(ValueError, match='focus'):
        select_aspect_evidence(c, [row()], ranker=lambda *args: seen.append(args),
                               focus_queries=invalid)
    assert not seen


def test_empty_focus_keeps_default_trace_and_behavior():
    c = build_supplement_contract('What is order Z-7 carrier phone?', [])
    a = select_aspect_evidence(c, [row()], ranker=rank)
    b = select_aspect_evidence(c, [row()], ranker=rank, focus_queries={})
    assert a.evidence == b.evidence
    assert [r['query'] for r in a.trace['aspect_ranking']] == [r['query'] for r in b.trace['aspect_ranking']]
    assert all('original_query' not in r for r in b.trace['aspect_ranking'])


def single_run(model=None, **kwargs):
    c = build_supplement_contract('What is order Z-7 carrier phone?', [])
    queries = []
    allocation = budget()
    model = model or Policy()
    result = run_repair(c, [row()], [row()], models=model, budget=allocation,
        reauthorize=kwargs.pop('reauthorize', deepcopy),
        validate_scope=kwargs.pop('validate_scope', lambda old, new: True),
        search=lambda args: queries.append(args.query) or [row('fresh')],
        active_arms=('treatment',), **kwargs)
    return result, queries, allocation, model


def test_single_treatment_search_and_scope_only_validate_active_query():
    result, queries, allocation, model = single_run(validate_scope=lambda old, new: new == 'Z-7 carrier phone')
    assert result.trace['status'] == 'completed'
    assert list(result.arms) == ['treatment']
    assert queries == ['Z-7 carrier phone']
    assert allocation.state['calls']['search']['attempted'] == 1
    assert allocation.state['calls']['policy']['attempted'] == 1
    assert allocation.state['limits']['policy'] == 2
    assert len(model.bodies) == 1


def test_same_query_is_valid_for_one_active_arm():
    result, queries, _, _ = single_run(Policy({'mode': 'independent', 'target_slot': 0,
                                             'query': 'What is order Z-7 carrier phone?'}))
    assert result.trace['status'] == 'completed'
    assert queries == ['What is order Z-7 carrier phone?']
    assert result.trace['controlled_query_equals_original'] is True


def test_single_bridge_has_literal_witness_and_two_policy_calls():
    model = Policy({'mode': 'entity_bridge', 'target_slot': 0, 'first_query': 'Z-7 carrier',
        'entity_description': 'carrier', 'followup_prefix': '', 'followup_suffix': ' phone'})
    result, queries, allocation, _ = single_run(model)
    assert queries == ['Zephyr phone']
    assert allocation.state['calls']['policy']['attempted'] == 2
    assert allocation.state['calls']['search']['attempted'] == 1
    assert 'p' in [r['chunk_id'] for r in result.arms['treatment']['evidence']]


def test_single_bridge_refresh_rejects_changed_literal_before_search():
    model = Policy({'mode': 'entity_bridge', 'target_slot': 0, 'first_query': 'Z-7 carrier',
        'entity_description': 'carrier', 'followup_prefix': '', 'followup_suffix': ' phone'})
    count = 0
    def auth(rows):
        nonlocal count
        count += 1
        return [dict(r, text='Revoked content') for r in rows] if count == 4 else deepcopy(rows)
    result, queries, _, _ = single_run(model, reauthorize=auth)
    assert not queries and result.trace['status'] == 'operational_failure'


def test_embedding_adapter_counts_real_embed_before_http(monkeypatch):
    from app import clients
    from scripts.run_policy_ablation_dev import BudgetedEmbedding
    cfg = SimpleNamespace(embed_model='synthetic-embed', embed_dimension=2,
                          ollama_url='http://synthetic.invalid', model_timeout_seconds=99)
    monkeypatch.setattr(clients, 'settings', lambda: cfg)
    seen = []
    def post(url, **kwargs):
        seen.append((url, kwargs))
        return httpx.Response(200, json={'embeddings': [[0.2, 0.8]]},
                              request=httpx.Request('POST', url))
    monkeypatch.setattr(clients.httpx, 'post', post)
    allocation = budget()
    model = BudgetedEmbedding(clients.Models(), allocation)
    assert model.embed(['synthetic query']) == [[0.2, 0.8]]
    with pytest.raises(BudgetExceeded, match='embedding_call_budget_exhausted'):
        model.embed(['second query'])
    assert len(seen) == 1 and seen[0][1]['timeout'] <= 99
    assert allocation.state['calls']['embedding']['attempted'] == 1
    assert not allocation.state['calls'].get('policy')


@pytest.mark.parametrize('cap', [0, 1])
def test_embedding_zero_or_failed_attempt_never_releases_cap(monkeypatch, cap):
    from app import clients
    from scripts.run_policy_ablation_dev import BudgetedEmbedding
    monkeypatch.setattr(clients, 'settings', lambda: SimpleNamespace(embed_model='fake', embed_dimension=2,
        ollama_url='http://synthetic.invalid', model_timeout_seconds=1))
    seen = []
    def failing(url, **kwargs):
        seen.append(url)
        raise httpx.ConnectError('synthetic failure')
    monkeypatch.setattr(clients.httpx, 'post', failing)
    allocation = budget(embedding=cap)
    model = BudgetedEmbedding(clients.Models(), allocation)
    if cap:
        with pytest.raises(clients.DependencyError):
            model.embed(['query'])
        assert allocation.state['calls']['embedding']['failed'] == 1
    with pytest.raises(BudgetExceeded):
        model.embed(['query'])
    assert len(seen) == cap


def test_ranker_positive_cap_and_task_local_exact_identity_cache():
    from scripts.run_policy_ablation_dev import cached_rank
    cache, events, calls = {}, [], []
    allocation = budget(ranker=1)
    def real(q, rs):
        calls.append((q, deepcopy(rs)))
        return rank(q, rs)
    for arm in ('original_supp', 'controlled_supp'):
        cached_rank('Zephyr phone', [row()], cache, real, allocation, events, arm=arm)
    assert len(calls) == 1
    assert events[1]['cached_pairs'] == 1 and events[1]['physical_ranker_calls'] == 0
    assert events[1]['cache_shared_across_arms'] is True
    with pytest.raises(BudgetExceeded):
        cached_rank('Different query', [row()], cache, real, allocation, events)
    assert len(calls) == 1
    assert allocation.state['calls']['ranker']['attempted'] == 1


def test_three_arms_keep_seed_and_search_original_independently_of_decline():
    from scripts.run_policy_ablation_dev import run_components
    c = build_supplement_contract('What is order Z-7 carrier phone?', [doc()])
    seed = [row()]
    searches = []
    result = run_components(c, seed, seed, models=Policy({'mode': 'unknown', 'reason': 'no_grounded_query'}),
        reauthorize=deepcopy, validate_scope=lambda *args: True,
        search=lambda args: searches.append(args.query) or [row(text='Zephyr phone 555.', cid='fresh')],
        ranker=rank, cfg=SimpleNamespace(agent_task_timeout_seconds=180, agent_policy_max_calls=2,
            agent_generation_max_calls=0, agent_judge_max_calls=0, agent_judge_token_budget=0))
    assert list(result.arms) == ['no_supp', 'original_supp', 'controlled_supp']
    assert result.arms['no_supp']['evidence'] == seed
    assert searches == [c.question]
    assert result.arms['original_supp']['status'] == 'selected'
    assert result.arms['controlled_supp']['status'] == 'skipped'
    assert result.trace['controlled_supp']['decline_is_unverified_model_belief'] is True
    assert result.trace['budgets']['no_supp']['calls'] == {}
    assert result.trace['budgets']['original_supp']['calls']['search']['attempted'] == 1


def component_call(*, question='What is order Z-7 carrier phone?', seed=None, docs=None,
                   proposal=None, search=None, validate_scope=None, reauthorize=deepcopy, ranker=rank):
    from scripts.run_policy_ablation_dev import run_components
    seed = seed if seed is not None else [row()]
    c = build_supplement_contract(question, docs if docs is not None else [doc()])
    queries = []
    result = run_components(c, seed, seed, models=Policy(proposal), reauthorize=reauthorize,
        search=search or (lambda args: queries.append(args.query) or [row(cid='fresh')]),
        validate_scope=validate_scope or (lambda *args: True), ranker=ranker,
        cfg=SimpleNamespace(agent_task_timeout_seconds=180, agent_policy_max_calls=2,
            agent_generation_max_calls=0, agent_judge_max_calls=0, agent_judge_token_budget=0))
    return result, queries


def test_private_rank_queries_are_separate_from_publishable_component_trace():
    result, queries = component_call()
    assert queries == ['What is order Z-7 carrier phone?', 'Z-7 carrier phone']
    assert 'What is order Z-7 carrier phone?' not in json.dumps(result.trace)
    assert result.private['ranking_traces']['controlled_supp'][0]['aspect_ranking'][0]['query'] == 'Z-7 carrier phone'


@pytest.mark.parametrize('alter', ['query', 'version', 'hash', 'title', 'text', 'locator', 'metadata'])
def test_cache_never_reuses_changed_source_identity_or_query(alter):
    from scripts.run_policy_ablation_dev import cached_rank
    cache, events, calls = {}, [], []
    allocation = budget(ranker=8)
    def record(q, rows):
        calls.append(q)
        return rank(q, rows)
    cached_rank('Zephyr phone', [row()], cache, record, allocation, events)
    r, q = row(), 'Zephyr phone'
    field = {'version':'version_id', 'hash':'source_sha256'}.get(alter, alter)
    if alter == 'query':
        q = 'Zephyr carrier'
    else:
        r[field] = {'changed': True} if field in ('metadata', 'locator') else 'changed'
    cached_rank(q, [r], cache, record, allocation, events)
    assert len(calls) == 2 and events[-1]['cached_pairs'] == 0


def test_rank_failure_remains_prepaid_and_does_not_populate_cache():
    from scripts.run_policy_ablation_dev import cached_rank
    cache, events, calls = {}, [], []
    allocation = budget(ranker=1)
    def fail(*args):
        calls.append(args)
        raise RuntimeError('failure')
    with pytest.raises(RuntimeError):
        cached_rank('Zephyr phone', [row()], cache, fail, allocation, events)
    with pytest.raises(BudgetExceeded):
        cached_rank('Zephyr phone', [row()], cache, fail, allocation, events)
    assert len(calls) == 1 and not cache
    assert allocation.state['calls']['ranker']['failed'] == 1
    assert events[0]['provider_server_ms'] is None


def test_ranker_eight_cap_is_enforced_before_ninth_dispatch():
    from scripts.run_policy_ablation_dev import cached_rank
    allocation, cache, events, calls = budget(), {}, [], []
    def record(q, rows):
        calls.append(q)
        return rank(q, rows)
    for i in range(8):
        cached_rank(f'Zephyr phone {i}', [row()], cache, record, allocation, events)
    with pytest.raises(BudgetExceeded):
        cached_rank('Zephyr phone nine', [row()], cache, record, allocation, events)
    assert len(calls) == allocation.state['calls']['ranker']['attempted'] == 8


@pytest.mark.parametrize('reason', ['empty', 'unresolved', 'scope', 'revoked'])
def test_component_safety_stop_prevents_all_source_dispatch(reason):
    kwargs = {}
    if reason == 'empty':
        kwargs['seed'] = []
    elif reason == 'unresolved':
        kwargs['question'] = 'Publisher P article published February 7, 2026'
    elif reason == 'scope':
        kwargs['validate_scope'] = lambda *args: False
    else:
        kwargs['reauthorize'] = lambda rows: [dict(r, authorized=False) for r in rows]
    result, queries = component_call(**kwargs)
    assert not queries
    assert all(not b['calls'] for b in result.trace['budgets'].values())
    assert result.trace['status'] in {'empty_seed', 'original_scope_unresolved', 'scope_mismatch', 'seed_or_scope_invalid'}


def test_independent_written_entity_is_usable_when_missing_from_seed():
    result, queries = component_call(question='What is Zephyr phone?',
        seed=[row(text='Background only.')], proposal={'mode': 'independent', 'target_slot': 0, 'query': 'Zephyr phone'})
    assert queries == ['What is Zephyr phone?', 'Zephyr phone']
    assert result.arms['controlled_supp']['status'] == 'selected'


def test_controlled_query_cannot_invent_entity_from_nowhere():
    result, queries = component_call(proposal={'mode':'independent', 'target_slot':0,
                                              'query':'Imaginary phone'})
    assert queries == ['What is order Z-7 carrier phone?']
    assert result.trace['controlled_supp']['status'] == 'proposal_rejected'
    assert result.trace['controlled_supp']['rejections'][0]['codes'] == ['no_grounded_query']


def test_bridge_query_cannot_insert_unwritten_followup_subject():
    result, queries = component_call(proposal={'mode': 'entity_bridge', 'target_slot': 0,
        'first_query': 'Z-7 carrier', 'entity_description': 'carrier',
        'followup_prefix': 'Imaginary ', 'followup_suffix': ' phone'})
    assert queries == ['What is order Z-7 carrier phone?']
    assert result.trace['controlled_supp']['status'] == 'proposal_rejected'


def test_bridge_and_backbone_have_four_total_unique_document_version_witnesses():
    seed = [row(), *[row(f'd{i}', text='Background only.', rank=i+2) for i in range(5)]]
    result, queries = component_call(seed=seed, docs=[doc(), *[doc(f'd{i}') for i in range(5)]],
        proposal={'mode': 'entity_bridge', 'target_slot':0, 'first_query':'Z-7 carrier',
                  'entity_description':'carrier', 'followup_prefix':'', 'followup_suffix':' phone'})
    trace = result.private['ranking_traces']['controlled_supp'][0]
    required = trace['required_ids']
    assert set(required) == {'p', 'd0', 'd1', 'd2'}
    assert trace['mandatory_witness_order'] == ['p', 'd0', 'd1', 'd2']
    assert len(required) == 4 and queries[-1] == 'Zephyr phone'
    assert result.trace['caps']['document_quota'] is None


def test_authorized_retrieval_transport_uses_frozen_question_scope_and_budget(monkeypatch):
    from app import clients, retrieval
    from app.config import Settings
    from scripts import run_supplement_contract_dev as contracts
    from scripts import run_evidence_supplement_dev as supplement
    from scripts import run_evidence_selection_dev as replay
    from scripts.run_policy_ablation_dev import authorized_search
    cfg = Settings(_env_file=None, embed_dimension=2)
    documents = [doc(), doc('q', 'Publisher Q')]
    c = build_supplement_contract('Discuss Zephyr phone according to Publisher P.', documents)
    monkeypatch.setattr(contracts, 'contract_for', lambda *args: c)
    monkeypatch.setattr(contracts, 'repair_scope_validator', lambda *args: (lambda *args: True, []))
    monkeypatch.setattr('app.security.readable_documents', lambda *args: documents)
    monkeypatch.setattr(supplement, 'live_candidates', lambda *args: [row()])
    monkeypatch.setattr(replay, 'rebind_rows', lambda db, user, rows, corpus: deepcopy(rows))
    seen = []
    def retrieve(db, user, query, **kwargs):
        seen.append((query, kwargs, retrieval._TASK_GOAL.get(), retrieval._FUSE_GOAL.get()))
        assert [d.id for d in kwargs['readable_documents_fn'](db, user)] == ['p']
        assert type(kwargs['models']).__name__ == 'BudgetedEmbedding'
        return SimpleNamespace(blocked_reason=None)
    monkeypatch.setattr(retrieval, 'retrieve_authorized', retrieve)
    api = authorized_search(object(), object(), c, cfg=cfg, models=clients.Models(), backend=object(), corpus={})
    allocation = budget()
    from app.execution_budget import use_budget
    with use_budget(allocation):
        assert api.search(SearchArgs(query='Zephyr phone', top_k=8)) == [row()]
    assert seen[0][0] == 'Zephyr phone' and seen[0][2] == c.question and seen[0][3] is False
    used = seen[0][1]['cfg']
    assert used.rewrite_mode == used.rerank_mode == 'off'
    assert used.source_query_plan is False and used.retrieval_candidate_depth == 50
    assert api.validate_scope(c.question, 'Zephyr phone') is True


def test_bridge_preserves_all_four_backbone_witnesses_plus_separate_literal_id():
    seed = [row(text='Background only.', cid='old'), row(cid='bridge', rank=2),
            *[row(f'd{i}', text='Background only.', rank=i+3) for i in range(4)]]
    result, _ = component_call(seed=seed, docs=[doc(), *[doc(f'd{i}') for i in range(4)]],
        proposal={'mode':'entity_bridge', 'target_slot':0, 'first_query':'Z-7 carrier',
            'entity_description':'carrier', 'followup_prefix':'', 'followup_suffix':' phone'})
    trace = result.private['ranking_traces']['controlled_supp'][0]
    assert set(trace['required_ids']) == {'bridge', 'old', 'd0', 'd1', 'd2'}
    assert trace['mandatory_witness_order'] == ['bridge', 'old', 'd0', 'd1', 'd2']


def test_arm_rotation_starts_each_source_deadline_when_its_arm_starts(monkeypatch):
    from scripts import run_policy_ablation_dev as component
    wall = [1000.0]
    created = []
    def allocation(cfg, saved):
        value = ExecutionBudget(cfg, saved=saved, clock=lambda: wall[0])
        created.append(value.state['deadline'])
        return value
    monkeypatch.setattr(component, 'ExecutionBudget', allocation)
    queries = []
    def search(args):
        queries.append(args.query)
        wall[0] += 120
        return [row(cid='fresh')]
    c = build_supplement_contract('What is order Z-7 carrier phone?', [doc()])
    result = component.run_components(c, [row()], [row()], models=Policy(), reauthorize=deepcopy,
        validate_scope=lambda *args: True, search=search, ranker=rank,
        cfg=SimpleNamespace(agent_task_timeout_seconds=180, agent_policy_max_calls=2,
            agent_generation_max_calls=0, agent_judge_max_calls=0, agent_judge_token_budget=0),
        order=('controlled_supp', 'original_supp', 'no_supp'))
    assert result.trace['order'] == ['controlled_supp', 'original_supp', 'no_supp']
    assert queries == ['Z-7 carrier phone', c.question]
    assert created == [1180, 1300, 1420]
    assert result.trace['status'] == 'completed'


@pytest.mark.parametrize('order', [('original_supp',), ('no_supp','original_supp','original_supp'),
                                  ('no_supp','original_supp','unknown')])
def test_invalid_arm_order_prevents_every_dispatch(order):
    from scripts.run_policy_ablation_dev import run_components
    with pytest.raises(ValueError, match='order'):
        run_components(None, [], [], models=None, reauthorize=None, search=None,
                       validate_scope=None, ranker=None, cfg=None, order=order)



def test_same_arm_cache_reuse_is_not_reported_as_cross_arm_sharing():
    from scripts.run_policy_ablation_dev import cached_rank
    cache, events, allocation = {}, [], budget()
    for _ in range(2):
        cached_rank('Zephyr phone', [row()], cache, rank, allocation, events, arm='controlled_supp')
    assert events[1]['cached_pairs'] == 1
    assert events[1]['cache_shared_across_arms'] is False


def test_scope_change_is_failure_without_usable_seed_or_claimed_success():
    result, queries = component_call(validate_scope=lambda *args: False)
    assert not queries
    assert all(a['status'] == 'failed' and a['evidence'] is None for a in result.arms.values())
    assert result.trace['errors'] and all(e['reason'] == 'scope_changed' for e in result.trace['errors'])


def test_policy_exhaustion_has_explicit_reason_and_does_not_become_decline(monkeypatch):
    from scripts import run_policy_ablation_dev as component
    original = ExecutionBudget
    def allocation(cfg, saved):
        saved = deepcopy(saved)
        saved['limits']['policy'] = 0
        return original(cfg, saved=saved)
    monkeypatch.setattr(component, 'ExecutionBudget', allocation)
    result, queries = component_call()
    assert queries == ['What is order Z-7 carrier phone?']
    assert result.arms['controlled_supp']['status'] == 'failed'
    assert result.arms['controlled_supp']['evidence'] is None
    assert result.trace['controlled_supp']['errors'][0]['reason'] == 'policy_call_budget_exhausted'
    assert 'decline_reason' not in result.trace['controlled_supp']


def test_failed_original_transport_is_not_successful_baseline_fallback():
    def fail(args):
        if args.query.startswith('What'):
            raise RuntimeError('synthetic transport down')
        return [row(cid='fresh')]
    result, _ = component_call(search=fail)
    assert result.arms['original_supp']['status'] == 'failed'
    assert result.arms['original_supp']['evidence'] is None
    assert result.arms['controlled_supp']['status'] == 'selected'
    assert result.trace['budgets']['original_supp']['calls']['search']['failed'] == 1


def test_decline_keeps_reverified_seed_and_unverified_model_belief():
    result, queries = component_call(proposal={'mode':'unknown', 'reason':'evidence_appears_sufficient'})
    assert queries == ['What is order Z-7 carrier phone?']
    assert result.arms['controlled_supp']['status'] == 'skipped'
    assert result.arms['controlled_supp']['evidence'] == [row()]
    assert result.trace['controlled_supp']['status'] == 'route_unknown'
    assert result.trace['controlled_supp']['decline_is_unverified_model_belief'] is True
    assert result.trace['answer_correctness'] is None


def test_equal_original_and_controlled_query_reuses_only_observed_task_scores():
    calls = []
    def recording(query, rows):
        calls.append(query)
        return rank(query, rows)
    q = 'What is order Z-7 carrier phone'
    result, queries = component_call(question=q, proposal={'mode':'independent', 'target_slot':0, 'query':q}, ranker=recording)
    assert queries == [q, q]
    assert result.arms['no_supp']['evidence'] == [row()]
    assert len(calls) == 1
    assert result.trace['cache_shared_across_arms'] is True
    assert result.trace['budgets']['original_supp']['calls']['ranker']['attempted'] == 1
    assert result.trace['budgets']['controlled_supp']['calls'].get('ranker') is None
    for allocation in result.trace['budgets'].values():
        assert not allocation['calls'].get('generation') and not allocation['calls'].get('judge')


def test_authorized_adapter_exposes_real_backend_and_embedding_dispatch_deltas(monkeypatch):
    from app import retrieval, security
    from app.config import Settings
    from app.execution_budget import use_budget
    from scripts import run_supplement_contract_dev as contracts
    from scripts import run_evidence_supplement_dev as supplement
    from scripts.run_policy_ablation_dev import authorized_search
    c = build_supplement_contract('What is order Z-7 carrier phone?', [doc()])
    monkeypatch.setattr(contracts, 'contract_for', lambda *args: c)
    monkeypatch.setattr(contracts, 'repair_scope_validator', lambda *args: (lambda *args: True, []))
    monkeypatch.setattr(security, 'readable_documents', lambda *args: [doc()])
    monkeypatch.setattr(supplement, 'live_candidates', lambda *args: [row()])
    models, backend = SimpleNamespace(events=[]), SimpleNamespace(events=[])
    def retrieve(*args, **kwargs):
        backend.events.extend([{'status':'ok', 'request_sha256':'synthetic'},
                               {'status':'ok', 'request_sha256':'synthetic-other-lane'}])
        models.events.append({'status':'ok', 'role':'embedding'})
        return SimpleNamespace(blocked_reason=None)
    monkeypatch.setattr(retrieval, 'retrieve_authorized', retrieve)
    api = authorized_search(None, None, c, cfg=Settings(_env_file=None), models=models,
                            backend=backend, corpus={})
    with use_budget(budget()):
        api.search(SearchArgs(query=c.question, top_k=8))
    assert api.events[0]['backend_requests'] == 2
    assert api.events[0]['embedding_requests'] == 1
    assert api.search.backend_events is backend.events
    assert api.search.events is api.events


def test_scope_revocation_during_model_decline_cannot_return_usable_seed():
    from scripts.run_policy_ablation_dev import run_components
    current = [True]
    model = Policy({'mode':'unknown', 'reason':'evidence_appears_sufficient'})
    original_post = model._post
    def post(*args, **kwargs):
        result = original_post(*args, **kwargs)
        current[0] = False
        return result
    model._post = post
    c = build_supplement_contract('What is order Z-7 carrier phone?', [doc()])
    result = run_components(c, [row()], [row()], models=model, reauthorize=deepcopy,
        validate_scope=lambda *args: current[0], search=lambda args: [row(cid='fresh')], ranker=rank,
        cfg=SimpleNamespace(agent_task_timeout_seconds=180, agent_policy_max_calls=2,
            agent_generation_max_calls=0, agent_judge_max_calls=0, agent_judge_token_budget=0))
    assert result.arms['controlled_supp']['status'] == 'failed'
    assert result.arms['controlled_supp']['evidence'] is None
    assert result.trace['errors'][-1]['reason'] == 'scope_changed'


def test_returned_fresh_rows_cannot_widen_original_publisher_scope():
    result, _ = component_call(question='Discuss Zephyr phone according to Publisher P.',
        docs=[doc(), doc('q', 'Publisher Q')], search=lambda args: [row('q')],
        proposal={'mode':'independent', 'target_slot':0, 'query':'Zephyr phone'})
    assert result.arms['original_supp']['status'] == 'failed'
    assert result.arms['controlled_supp']['status'] == 'failed'
    assert result.arms['no_supp']['evidence'] == [row()]
    assert result.trace['budgets']['original_supp']['calls']['search']['failed'] == 1


def test_source_changed_after_successful_search_cannot_reuse_stale_seed():
    current = [True]
    def search(args):
        current[0] = False
        return [row(cid='fresh')]
    result, _ = component_call(search=search, validate_scope=lambda *args: current[0])
    assert result.arms['original_supp']['status'] == 'failed'
    assert result.arms['original_supp']['evidence'] is None
    assert result.arms['controlled_supp']['evidence'] is None


@pytest.mark.parametrize('kind', ['unresolved', 'history', 'long', 'empty'])
def test_every_normal_stop_still_rejects_changed_current_original_scope(kind):
    from dataclasses import replace
    from scripts.run_policy_ablation_dev import run_components
    q = 'What is order Z-7 carrier phone?'
    if kind == 'unresolved':
        q = 'Publisher P article published February 7, 2026'
    elif kind == 'long':
        q += ' background'*60
    c = build_supplement_contract(q, [doc(), doc('q', 'Publisher Q')])
    if kind == 'history':
        c = replace(c, reason='unsupported_history')
    seed = [] if kind == 'empty' else [row('q')]
    result = run_components(c, seed, seed, models=Policy(), reauthorize=deepcopy,
        validate_scope=lambda *args: False, search=lambda args: pytest.fail('no search'),
        ranker=rank, cfg=SimpleNamespace(agent_task_timeout_seconds=180, agent_policy_max_calls=2,
            agent_generation_max_calls=0, agent_judge_max_calls=0, agent_judge_token_budget=0))
    assert all(a['status'] == 'failed' and a['evidence'] is None for a in result.arms.values())
    assert all(e['reason'] == 'scope_changed' for e in result.trace['errors'])


def test_unchanged_unresolved_source_retains_only_authorized_immutable_seed():
    result, queries = component_call(question='Publisher P article published February 7, 2026')
    assert not queries
    assert result.trace['status'] == 'original_scope_unresolved'
    assert result.arms['no_supp']['evidence'] == [row()]
    assert result.trace['retained_seed_scope'] == 'authorized_frozen_baseline_only'


def test_unresolved_stop_cannot_retain_seed_with_wrong_authorized_hash():
    result, queries = component_call(question='Publisher P article published February 7, 2026',
                                     seed=[dict(row(), source_sha256='wrong')])
    assert not queries
    assert all(a['status'] == 'failed' and a['evidence'] is None for a in result.arms.values())


@pytest.mark.parametrize('first', ['original_supp', 'controlled_supp'])
def test_task_cancellation_stops_every_subsequent_arm_and_discloses_prepaid_cost(first):
    from app.execution_budget import TaskCancelled
    from scripts.run_policy_ablation_dev import run_components
    queries = []
    def cancel(args):
        queries.append(args.query)
        raise TaskCancelled('synthetic private message')
    order = (first, *[name for name in ('no_supp','original_supp','controlled_supp') if name != first])
    c = build_supplement_contract('What is order Z-7 carrier phone?', [doc()])
    result = run_components(c, [row()], [row()], models=Policy(), reauthorize=deepcopy,
        validate_scope=lambda *args: True, search=cancel, ranker=rank, order=order,
        cfg=SimpleNamespace(agent_task_timeout_seconds=180, agent_policy_max_calls=2,
            agent_generation_max_calls=0, agent_judge_max_calls=0, agent_judge_token_budget=0))
    assert len(queries) == 1
    assert result.trace['status'] == 'interrupted'
    assert result.trace['interruption_reason'] == 'task_cancelled'
    assert result.arms[first]['status'] == 'interrupted'
    assert result.trace['budgets'][first]['calls']['search']['failed'] == 1
    assert all(result.arms[n]['status'] == 'not_run' and result.trace['budgets'][n] is None for n in order[1:])
    assert 'synthetic private message' not in json.dumps(result.trace)


def test_keyboard_interrupt_preserves_consumed_component_result_without_next_dispatch():
    from scripts.run_policy_ablation_dev import run_components
    queries = []
    def interrupt(args):
        queries.append(args.query)
        raise KeyboardInterrupt()
    c = build_supplement_contract('What is order Z-7 carrier phone?', [doc()])
    with pytest.raises(KeyboardInterrupt) as caught:
        run_components(c, [row()], [row()], models=Policy(), reauthorize=deepcopy,
            validate_scope=lambda *args: True, search=interrupt, ranker=rank,
            cfg=SimpleNamespace(agent_task_timeout_seconds=180, agent_policy_max_calls=2,
                agent_generation_max_calls=0, agent_judge_max_calls=0, agent_judge_token_budget=0))
    assert len(queries) == 1
    result = caught.value.component_result
    assert result.trace['status'] == 'interrupted'
    assert result.trace['budgets']['original_supp']['calls']['search']['failed'] == 1
    assert result.arms['controlled_supp']['status'] == 'not_run'


def test_no_supp_reauthorization_and_stop_checks_obey_own_source_deadline(monkeypatch):
    from scripts import run_policy_ablation_dev as component
    wall = [1000.0]
    def allocate(cfg, saved):
        return ExecutionBudget(cfg, saved=saved, clock=lambda: wall[0])
    monkeypatch.setattr(component, 'ExecutionBudget', allocate)
    def slow_auth(rows):
        wall[0] += 400
        return deepcopy(rows)
    result, queries = component_call(reauthorize=slow_auth)
    assert not queries
    assert result.arms['no_supp']['status'] == 'failed'
    assert result.arms['no_supp']['evidence'] is None
    assert result.trace['errors'][0]['reason'] == 'task_deadline_exceeded'


def test_factory_distinguishes_unchanged_original_unresolved_scope_from_query_admissibility(monkeypatch):
    from app import security
    from app.config import Settings
    from scripts import run_supplement_contract_dev as contracts
    from scripts.run_policy_ablation_dev import authorized_search
    docs = [doc()]
    c = build_supplement_contract('Publisher P article published February 7, 2026', docs)
    monkeypatch.setattr(contracts, 'contract_for', lambda *args: c)
    monkeypatch.setattr(contracts, 'repair_scope_validator', lambda *args: (lambda *args: False, []))
    monkeypatch.setattr(contracts, 'scope_keys', lambda *args: [])
    monkeypatch.setattr(security, 'readable_documents', lambda *args: docs)
    api = authorized_search(None, None, c, cfg=Settings(_env_file=None), models=Policy(), backend=None, corpus={})
    assert api.validate_scope(c.question, c.question) is True
    assert api.validate_scope(c.question, 'Different query') is False
    docs.append(doc('q', 'Publisher Q'))
    assert api.validate_scope(c.question, c.question) is False


def test_scope_change_during_final_stop_rebind_is_checked_after_rebind():
    current, count = [True], [0]
    def auth(rows):
        count[0] += 1
        if count[0] == 2:
            current[0] = False
        return deepcopy(rows)
    result, queries = component_call(question='Publisher P article published February 7, 2026',
        reauthorize=auth, validate_scope=lambda *args: current[0])
    assert not queries
    assert all(a['status'] == 'failed' and a['evidence'] is None for a in result.arms.values())
    assert all(e['reason'] == 'scope_changed' for e in result.trace['errors'])


def test_cancellation_in_no_supp_rebind_prevents_all_models_and_searches():
    from app.execution_budget import TaskCancelled
    def cancel(rows):
        raise TaskCancelled('private')
    result, queries = component_call(reauthorize=cancel)
    assert not queries
    assert result.trace['status'] == 'interrupted'
    assert result.trace['interrupted_arm'] == 'no_supp'
    assert result.trace['budgets']['no_supp']['calls'] == {}
    assert result.trace['budgets']['original_supp'] is None
    assert result.trace['budgets']['controlled_supp'] is None


def test_cancellation_in_policy_transport_consumes_policy_without_search_or_later_arm():
    from app.execution_budget import TaskCancelled
    from scripts.run_policy_ablation_dev import run_components
    model = Policy()
    def cancel(*args, **kwargs):
        raise TaskCancelled('private')
    model._post = cancel
    c = build_supplement_contract('What is order Z-7 carrier phone?', [doc()])
    result = run_components(c, [row()], [row()], models=model, reauthorize=deepcopy,
        validate_scope=lambda *args: True, search=lambda args: pytest.fail('no search'),
        ranker=rank, order=('controlled_supp','original_supp','no_supp'),
        cfg=SimpleNamespace(agent_task_timeout_seconds=180, agent_policy_max_calls=2,
            agent_generation_max_calls=0, agent_judge_max_calls=0, agent_judge_token_budget=0))
    assert result.trace['status'] == 'interrupted'
    assert result.trace['budgets']['controlled_supp']['calls']['policy']['failed'] == 1
    assert 'search' not in result.trace['budgets']['controlled_supp']['calls']
    assert result.trace['budgets']['original_supp'] is None


def test_source_component_never_accepts_seed_without_frozen_authorized_binding():
    result, queries = component_call(docs=[])
    assert not queries
    assert all(a['status'] == 'failed' and a['evidence'] is None for a in result.arms.values())
    assert all(e['reason'] == 'source_identity_or_acl_changed' for e in result.trace['errors'])
