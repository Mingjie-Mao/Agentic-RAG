"""Experiment-only source-bound supplementation using official LangGraph.

Raw arms stay in evaluator memory. The trace contains only program enums/hashes.
No labels, answer generation, retries or production execution are accepted here.
"""
from copy import deepcopy
import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent.evidence_supplement import (
    SupplementResult, _IntegrityError, _NoBridge, _authorize, _hash, _merge, _norm,
)
from agent.planned import extract_value, focus_document, literal_values, ungrounded_terms
from agent.supplement_proposal import (
    EntityBridgeProposal, FocusProposal, UnknownProposal, make_proposal, parse_proposal, program_bridge,
)
from agent.tools import SearchArgs
from app.chunking import token_count
from app.evidence_selection import select_slot_evidence
from app.execution_budget import use_budget
from app.supplement_contract import assess_slot_presence, public_presence


class RepairState(TypedDict, total=False):
    stopped: bool
    proposal: object
    source: str
    queries: dict


def run_repair(contract, seed, candidates, *, models, reauthorize, search,
               validate_scope, budget, caps=None, order=('control', 'treatment'),
               proposal_provider=None, evidence_selector=None):
    """Run the shared graph with optional trusted experiment adapters.

    Injected providers/selectors own their model accounting. The graph continues
    to enforce scope, authorization refresh and the shared search/policy limits.
    """
    if tuple(order) not in {('control', 'treatment'), ('treatment', 'control')}:
        raise ValueError('Invalid arm order')
    caps = {'limit': 8, 'token_budget': 5000, 'document_quota': 2, 'source_quota': None, **(caps or {})}
    if (set(caps) != {'limit', 'token_budget', 'document_quota', 'source_quota'}
            or type(caps['limit']) is not int or not 1 <= caps['limit'] <= 8
            or type(caps['token_budget']) is not int or not 1 <= caps['token_budget'] <= 5000
            or any(v is not None and (type(v) is not int or v < 1)
                   for v in (caps['document_quota'], caps['source_quota']))):
        raise ValueError('Invalid context caps')
    seed, candidates = deepcopy(seed), deepcopy(candidates)
    arms = {a: {'status': 'pending', 'evidence': None, 'candidates': None, 'query': None} for a in order}
    trace = {'status': 'pending', 'mode': contract.mode, 'contract': contract.public_summary(),
             'order': list(order), 'errors': [], 'rejections': [], 'arms': {},
             'shared_policy_cost': 'charge_each_arm_in_runner', 'physical_cost': 'delegated_to_caller'}
    before = budget.snapshot()
    budget.state['deadline'] = min(budget.state['deadline'], budget.clock() + 180)
    for role in ('policy', 'search'):
        budget.state['limits'][role] = min(budget.state['limits'].get(role, 2), 2)
    if budget.persist:
        budget.persist(budget.snapshot())

    def stop(status, rejection=None):
        _authorize(seed, reauthorize)
        trace['status'] = status
        if rejection:
            trace['rejections'].append(rejection.public())
        for arm in arms.values():
            arm.update(status='skipped', evidence=deepcopy(seed))
        return {'stopped': True}

    def fail(stage, exc, name=None):
        trace['errors'].append({'stage': stage, 'type': type(exc).__name__, 'arm': name})
        for a in ([name] if name else arms):
            if arms[a]['status'] == 'pending':
                arms[a].update(status='failed', evidence=None)
        if name is None:
            trace['status'] = 'operational_failure'
        return {'stopped': name is None}

    def gate(state):
        try:
            _authorize(seed, reauthorize)
            if (len(seed) > caps['limit'] or len({r['chunk_id'] for r in seed}) != len(seed)
                    or sum(token_count(r['text']) + token_count(r['title']) + 100 for r in seed) > caps['token_budget']):
                raise _IntegrityError()
            if not seed:
                return stop('empty_seed')
            if len(contract.question) > 500:
                return stop('query_too_long')
            if any(r.resolution != 'supported' for r in contract.source_requests):
                return stop('original_scope_unresolved')
            if contract.reason == 'unsupported_history':
                return stop('history_not_applicable')
            presence = assess_slot_presence(contract, seed)
            trace['seed_presence'] = public_presence(presence)
            if not presence['gate']:
                return stop('structurally_complete')
            return {'stopped': False}
        except Exception as exc:
            return fail('gate', exc)

    def planning(state):
        try:
            current = _authorize(seed, reauthorize)
            needed = [i for i, r in enumerate(assess_slot_presence(contract, current)['slots'])
                      if r['state'] in {'missing', 'unknown'}]
            trace['proposal_attempted'] = True
            proposal = (proposal_provider or make_proposal)(models, contract, current, needed)
            if type(proposal) not in {FocusProposal, EntityBridgeProposal, UnknownProposal}:
                raise _NoBridge(('proposal_schema',), stage='schema')
            proposal = parse_proposal(proposal.model_dump_json(), contract.mode, needed)
            if proposal.mode == 'unknown':
                trace['decline_reason'] = proposal.reason
                trace['decline_is_unverified_model_belief'] = True
                return stop('route_unknown')
            trace.update(mode=proposal.mode, target_slot=proposal.target_slot,
                         proposal_sha256=_hash(proposal.model_dump()), proposal_accepted=True)
            return {'proposal': proposal}
        except _NoBridge as exc:
            return stop('proposal_rejected', exc)
        except Exception as exc:
            return fail('proposal', exc)

    def independent(state):
        p = state['proposal']
        return {'queries': {'control': contract.question, 'treatment': p.query.strip()}}

    def bridge(state):
        try:
            current = _authorize(seed, reauthorize)
            first = program_bridge(state['proposal']).steps[0]
            document, _ = focus_document(first.query, current)
            if ungrounded_terms(first.query, current, document):
                return stop('binding_rejected', _NoBridge(('subject_focus',), stage='focus'))
            focused = [r for r in current if r['document_id'] == document]
            # Preflight before the decorated operation so oversize input dispatches
            # neither a policy operation nor a physical model request.
            payload = {'question': contract.question, 'step': first.model_dump(), 'passages': focused}
            if len(json.dumps(payload, ensure_ascii=False)) > 11000:
                return stop('binding_rejected', _NoBridge(('payload_too_large',), stage='extract_payload'))
            answer = extract_value(models, contract.question, first, focused, complete_passages=True)
            _authorize(current, reauthorize)
            if not answer or not any(c.isalpha() for c in answer) or '{' in answer or '}' in answer:
                return stop('binding_rejected', _NoBridge(('entity_invalid',), stage='literal'))
            values, source, quote = literal_values(answer, focused, 'entity', first.query)
            actual = next((r for r in current if r['chunk_id'] == source), None)
            if (not values or len(values) != 1 or not quote or actual is None
                    or _norm(values[0]) not in _norm(quote) or _norm(quote) not in _norm(actual['text'])):
                return stop('binding_rejected', _NoBridge(('literal_binding_mismatch',), stage='literal'))
            p = state['proposal']
            trace.update(bridge_chunk_sha256=_hash(source), bridge_source_sha256=actual['source_sha256'],
                         value_sha256=_hash(values[0]), quote_sha256=_hash(quote))
            return {'source': source, 'queries': {
                'control': (p.followup_prefix + p.followup_suffix).strip(),
                'treatment': (p.followup_prefix + values[0] + p.followup_suffix).strip()}}
        except _NoBridge as exc:
            return stop('binding_rejected', exc)
        except Exception as exc:
            return fail('extract', exc)

    def validation(state):
        try:
            queries = state['queries']
            if any(not q.strip() or len(q) > 500 or '{' in q or '}' in q for q in queries.values()):
                return stop('query_rejected', _NoBridge(('query_length_or_placeholder',), stage='query'))
            if queries['control'] == queries['treatment']:
                return stop('indistinguishable_pair')
            if not all(validate_scope(contract.question, queries[a]) is True for a in order):
                return stop('scope_mismatch', _NoBridge(('scope_changed',), stage='scope'))
            return {}
        except Exception as exc:
            return fail('scope', exc)

    def dispatch(name):
        def node(state):
            try:
                _authorize(seed, reauthorize)
                q = state['queries'][name]
                if validate_scope(contract.question, q) is not True:
                    raise _IntegrityError()
                arms[name]['query'] = q
                trace.setdefault('query_sha256', {})[name] = _hash(q)
                with budget.call('search'):
                    fresh = search(SearchArgs(query=q, top_k=8))
                    if not isinstance(fresh, list):
                        raise _IntegrityError()
                arms[name]['fresh'] = deepcopy(fresh)
            except Exception as exc:
                fail('search', exc, name)
            return {}
        return node

    def selection(state):
        for a in order:
            if arms[a]['status'] != 'pending':
                continue
            try:
                budget.check()
                old = _authorize(candidates, reauthorize)
                _authorize(seed, reauthorize)
                fresh = _authorize(arms[a].pop('fresh'), reauthorize)
                union = _merge(old, fresh)
                required = [state['source']] if state.get('source') else []
                focus = {contract.slots[state['proposal'].target_slot].slot_id: state['queries']['treatment']}
                chosen = (evidence_selector or select_slot_evidence)(
                    contract, union, required_ids=required, focus_queries=focus, **caps)
                trace['selection_focus'] = 'shared_model_query_scoped_lexical_only'
                trace['selection_focus_sha256'] = _hash(state['queries']['treatment'])
                _authorize(chosen.evidence, reauthorize)
                if not set(required) <= {r['chunk_id'] for r in chosen.evidence}:
                    raise _IntegrityError()
                arms[a].update(status='selected', evidence=chosen.evidence, candidates=union)
                trace['arms'][a] = {'context_tokens': chosen.context_tokens,
                    'candidate_count': len(union), 'context_count': len(chosen.evidence),
                    'selected_sha256': [_hash(r['chunk_id']) for r in chosen.evidence],
                    'presence': public_presence(assess_slot_presence(contract, chosen.evidence)),
                    'slot_floor': [{k: f[k] for k in ('slot_id', 'basis', 'status', 'reason')}
                                  for f in chosen.trace['slot_floor']]}
            except Exception as exc:
                fail('select', exc, a)
        trace['status'] = 'completed' if all(a['status'] == 'selected' for a in arms.values()) else 'partial_failure'
        return {}

    graph = StateGraph(RepairState)
    for name, node in [('gate', gate), ('proposal', planning), ('independent', independent),
                       ('bridge', bridge), ('validate', validation),
                       ('first_search', dispatch(order[0])), ('second_search', dispatch(order[1])), ('select', selection)]:
        graph.add_node(name, node)
    graph.add_edge(START, 'gate')
    graph.add_conditional_edges('gate', lambda s: 'end' if s.get('stopped') else 'next', {'end': END, 'next': 'proposal'})
    graph.add_conditional_edges('proposal', lambda s: 'end' if s.get('stopped') else s['proposal'].mode,
                                {'end': END, 'independent': 'independent', 'entity_bridge': 'bridge'})
    for node in ('independent', 'bridge'):
        graph.add_conditional_edges(node, lambda s: 'end' if s.get('stopped') else 'next', {'end': END, 'next': 'validate'})
    graph.add_conditional_edges('validate', lambda s: 'end' if s.get('stopped') else 'next', {'end': END, 'next': 'first_search'})
    graph.add_edge('first_search', 'second_search')
    graph.add_edge('second_search', 'select')
    graph.add_edge('select', END)
    try:
        with use_budget(budget):
            graph.compile().invoke({'stopped': False})
    except Exception as exc:
        fail('graph', exc)
    after = budget.snapshot()
    trace['calls'] = {role: {k: after['calls'].get(role, {}).get(k, 0) - before['calls'].get(role, {}).get(k, 0)
                          for k in ('attempted', 'succeeded', 'failed', 'wall_ms')} for role in ('policy', 'search')}
    for a, arm in arms.items():
        arm.pop('fresh', None)
        trace['arms'].setdefault(a, {})['status'] = arm['status']
    return SupplementResult(arms, trace)
