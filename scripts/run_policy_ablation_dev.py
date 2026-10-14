"""Experiment-only source components; no CLI, registration, labels or generation.

The caller supplies frozen authorized relevant_doc seed/candidates and existing
transport capabilities. Raw evidence and ranking traces remain private memory.
A fresh run owns its exact-query score cache; it is never global or persisted.
"""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
import time

from agent.evidence_supplement import _IntegrityError, _NoBridge, _authorize, _hash, _merge
from agent.planned import _terms, focus_document, ungrounded_terms
from agent.supplement_proposal import EntityBridgeProposal, FocusProposal, make_proposal
from agent.supplement_repair import run_repair
from agent.tools import SearchArgs
from app.chunking import token_count
from app.evidence_focus import _ranked, select_aspect_evidence
from app.evidence_selection import _rank
from app.execution_budget import BudgetExceeded, ExecutionBudget, TaskCancelled, current_budget, use_budget
from scripts.run_external_repair_dev import backbone_ids

ARMS = ('no_supp', 'original_supp', 'controlled_supp')
SOURCE_LIMITS = {'policy': 2, 'search': 1, 'embedding': 1, 'ranker': 8,
                 'generation': 0, 'judge': 0, 'judge_tokens': 0}
CAPS = {'limit': 8, 'token_budget': 5000, 'document_quota': None, 'source_quota': None}


@dataclass
class ComponentResult:
    arms: dict
    trace: dict
    private: dict


class BudgetedEmbedding:
    """One prepaid actual embed attempt; existing Models owns HTTP/timeouts.

    Retrieval sees only embed. Policy/generation decorators retain their existing
    owner, so this adapter cannot count those roles twice.
    """
    __slots__ = ('models', 'budget')

    def __init__(self, models, budget=None):
        self.models, self.budget = models, budget

    def embed(self, texts):
        allocation = self.budget or current_budget()
        if allocation is None:
            raise ValueError('Experiment embedding requires an execution budget')
        with use_budget(allocation), allocation.call('embedding'):
            return self.models.embed(texts)


def cached_rank(query, rows, cache, ranker, budget, events, *, arm=None):
    """Charge actual scoring before dispatch, disclose task-local pair reuse."""
    budget.check()
    keys = {r['chunk_id']: _hash([query, {k: r.get(k) for k in (
        'chunk_id', 'document_id', 'version_id', 'source_sha256', 'title', 'text',
        'metadata', 'locator', 'active', 'authorized')}]) for r in rows}
    fresh = [r for r in rows if keys[r['chunk_id']] not in cache]
    event = {'query_sha256': _hash(query), 'input_pairs': len(rows),
             'fresh_pairs': len(fresh), 'cached_pairs': len(rows)-len(fresh),
             'physical_ranker_calls': 0, 'cache_shared_across_arms': any(
                 keys[r['chunk_id']] in cache and cache[keys[r['chunk_id']]][1] != arm
                 for r in rows),
             'status': 'failed', 'provider_server_ms': None}
    events.append(event)
    began = time.monotonic()
    try:
        if fresh:
            with use_budget(budget), budget.call('ranker'):
                event['physical_ranker_calls'] = 1
                scores, _ = _ranked(query, fresh, ranker)
            for cid, score in scores:
                cache[keys[cid]] = (score, arm)
        result = [dict(r, rerank_score=cache[keys[r['chunk_id']]][0]) for r in rows]
        budget.check()
        event['status'] = 'ok'
        return sorted(result, key=lambda r: (-r['rerank_score'], _rank(r), r['chunk_id']))
    finally:
        event['wall_ms'] = round((time.monotonic()-began)*1000, 3)


def _allowed_rows(contract, rows):
    bindings = {(d, v): sha for d, v, sha in contract.authorized_bindings}
    allowed = {pair for req in contract.source_requests for pair in req.document_versions}
    for row in rows:
        pair = (row['document_id'], row['version_id'])
        if ((pair not in bindings or not bindings[pair]
                or bindings[pair] != row['source_sha256'])
                or (contract.source_requests and all(req.resolution == 'supported' for req in contract.source_requests)
                    and pair not in allowed)):
            raise _IntegrityError()
    return rows


def _grounded_proposal(models, contract, seed, needed):
    proposal = make_proposal(models, contract, seed, needed)
    if isinstance(proposal, (FocusProposal, EntityBridgeProposal)):
        # Admissibility is conservative vocabulary provenance, never coverage.
        # Written question entities remain usable even when absent from the seed.
        slot = contract.slots[proposal.target_slot]
        requests = {r.request_id: r for r in contract.source_requests}
        allowed = None
        if slot.source_request_ids:
            scopes = [set(requests[i].document_versions) for i in slot.source_request_ids]
            allowed = set.intersection(*scopes)
        scoped = [r for r in seed if allowed is None or
                  (r['document_id'], r['version_id']) in allowed]
        written = {'chunk_id': 'question', 'document_id': 'question', 'title': '',
                   'text': contract.question}
        sources = [written, *scoped]
        terms = set().union(*(_terms(r['title']+' '+r['text']) for r in sources))
        query = proposal.query if isinstance(proposal, FocusProposal) else proposal.first_query
        queries = [query] if isinstance(proposal, FocusProposal) else [
            query, proposal.followup_prefix + ' ' + proposal.followup_suffix]
        if any(_terms(q) - terms for q in queries):
            raise _NoBridge(('no_grounded_query',), stage='query_provenance')
        document, _ = focus_document(query, sources)
        if ungrounded_terms(query, sources, document):
            raise _NoBridge(('subject_focus',), stage='query_provenance')
    return proposal


def run_components(contract, seed, candidates, *, models, reauthorize, search,
                   validate_scope, ranker, cfg, order=ARMS):
    """Return the true three source arms with equal per-arm maxima.

    ``search`` is one authorized retrieval call (use authorized_search below for
    live transport). It executes inside the active arm's ExecutionBudget.
    The original arm is independent of the controlled proposal's outcome.
    No correctness, gold, task IDs, paths, retries or answers are accepted here.
    """
    if not isinstance(order, (tuple, list)) or len(order) != 3 or set(order) != set(ARMS):
        raise ValueError('Invalid component arm order')
    order = tuple(order)
    seed, candidates = deepcopy(seed), deepcopy(candidates)
    budgets = {}
    arms = {name: {'status': 'pending', 'evidence': None, 'candidates': None, 'query': None} for name in ARMS}
    trace = {'status': 'pending', 'order': list(order), 'selection_policy': 'rank_floor',
             'caps': dict(CAPS), 'source_limits': dict(SOURCE_LIMITS),
             'answer_correctness': None, 'errors': [], 'physical_events': {},
             'backend_events': {}, 'retrieval_events': {},
             'rank_cache': 'task_local_exact_query_complete_source_identity',
             'cache_reuse_scope': 'same_task_across_source_arms'}
    rankings = {name: [] for name in ARMS}
    rank_events = {name: [] for name in ARMS}
    search_events = {name: [] for name in ARMS}
    cache = {}

    def fail(name, stage, exc):
        arms[name].update(status='failed', evidence=None)
        trace['errors'].append({'arm': name, 'stage': stage, 'type': type(exc).__name__,
                                'reason': exc.reason if isinstance(exc, BudgetExceeded) else
                                'scope_changed' if isinstance(exc, _NoBridge) and 'scope_changed' in exc.codes else
                                'source_identity_or_acl_changed' if isinstance(exc, _IntegrityError) else 'operational_failure'})
        if isinstance(exc, _NoBridge):
            trace.setdefault('rejections', []).append({'arm': name, **exc.public()})

    def selector(name):
        allocation = budgets[name]
        def select(c, rows, *, required_ids=(), focus_queries=None, **kwargs):
            rows = _allowed_rows(c, _authorize(rows, reauthorize))
            # Four baseline doc/version witnesses plus an independent bridge ID.
            # Preserve all mandatory IDs; infeasibility is the allocator's error.
            required = list(dict.fromkeys([*required_ids, *backbone_ids(c, seed, rows)]))
            with use_budget(allocation):
                result = select_aspect_evidence(c, rows, required_ids=required,
                    focus_queries=focus_queries, selection_policy='rank_floor',
                    ranker=lambda q, rs: cached_rank(q, rs, cache, ranker, allocation, rank_events[name], arm=name),
                    **kwargs)
            result.trace['mandatory_witness_order'] = required
            rankings[name].append(deepcopy(result.trace))
            result.trace['slot_floor'] = result.trace['aspect_floor']
            _allowed_rows(c, _authorize(result.evidence, reauthorize))
            return result
        return select

    def dispatch(name, args):
        event = {'query_sha256': _hash(args.query), 'status': 'failed', 'physical_search_calls': 0}
        search_events[name].append(event)
        began = time.monotonic()
        try:
            if validate_scope(contract.question, args.query) is not True:
                raise _NoBridge(('scope_changed',), stage='scope')
            _authorize(seed, reauthorize)
            event['physical_search_calls'] = 1
            fresh = _allowed_rows(contract, _authorize(search(args), reauthorize))
            if validate_scope(contract.question, args.query) is not True:
                raise _NoBridge(('scope_changed',), stage='scope')
            event.update(status='ok', candidate_count=len(fresh))
            return fresh
        finally:
            event['wall_ms'] = round((time.monotonic()-began)*1000, 3)

    # These are route enums, never permission or correctness conclusions.
    stop = ('empty_seed' if not seed else
            'original_scope_unresolved' if any(r.resolution != 'supported' for r in contract.source_requests) else
            'history_not_applicable' if contract.reason == 'unsupported_history' else
            'query_too_long' if len(contract.question) > 500 else None)
    if stop == 'original_scope_unresolved':
        trace['retained_seed_scope'] = 'authorized_frozen_baseline_only'
    interruption = None
    for name in order:
        # Source phase time belongs to this arm, starting at its own turn.
        allocation = ExecutionBudget(cfg, saved={'limits': dict(SOURCE_LIMITS)})
        allocation.state['deadline'] = min(allocation.state['deadline'], allocation.clock()+180)
        budgets[name] = allocation
        physical = getattr(models, 'events', None)
        start_event = len(physical) if physical is not None else None
        backend_events = getattr(search, 'backend_events', None)
        retrieval_events = getattr(search, 'events', None)
        backend_start = len(backend_events) if backend_events is not None else None
        retrieval_start = len(retrieval_events) if retrieval_events is not None else None
        try:
            with use_budget(allocation):
                allocation.check()
                current = _authorize(seed, reauthorize)
                allocation.check()
                if validate_scope(contract.question, contract.question) is not True:
                    raise _NoBridge(('scope_changed',), stage='scope')
                allocation.check()
                _allowed_rows(contract, current)
                if len(seed) > 8 or len({r['chunk_id'] for r in seed}) != len(seed) or sum(
                        token_count(r['text'])+token_count(r['title'])+100 for r in seed) > 5000:
                    raise _IntegrityError()
                if stop:
                    arms[name].update(status='seed_retained' if name == 'no_supp' else 'skipped',
                                      evidence=deepcopy(seed), candidates=deepcopy(candidates))
                elif name == 'no_supp':
                    arms[name].update(status='seed_retained', evidence=deepcopy(seed), candidates=deepcopy(candidates))
                elif name == 'original_supp':
                    with allocation.call('search'):
                        fresh = dispatch(name, SearchArgs(query=contract.question, top_k=8))
                    old = _allowed_rows(contract, _authorize(candidates, reauthorize))
                    union = _merge(old, fresh)
                    chosen = selector(name)(contract, union, **CAPS)
                    arms[name].update(status='selected', evidence=chosen.evidence,
                                      candidates=union, query=contract.question)
                else:
                    result = run_repair(contract, seed, candidates, models=models, reauthorize=reauthorize,
                        search=lambda args: dispatch(name, args), validate_scope=validate_scope,
                        budget=allocation, caps=CAPS, active_arms=('treatment',),
                        proposal_provider=_grounded_proposal, evidence_selector=selector(name))
                    arms[name] = result.arms['treatment']
                    trace['controlled_supp'] = result.trace
                allocation.check()
                if validate_scope(contract.question, contract.question) is not True:
                    raise _NoBridge(('scope_changed',), stage='scope')
                if arms[name]['evidence'] is not None:
                    if validate_scope(contract.question, arms[name]['query'] or contract.question) is not True:
                        raise _NoBridge(('scope_changed',), stage='scope')
                    _allowed_rows(contract, _authorize(arms[name]['evidence'], reauthorize))
                if validate_scope(contract.question, contract.question) is not True:
                    raise _NoBridge(('scope_changed',), stage='scope')
                allocation.check()
        except (TaskCancelled, KeyboardInterrupt) as exc:
            interruption = exc
            arms[name].update(status='interrupted', evidence=None)
            trace.update(interrupted=True, interrupted_arm=name,
                interruption_reason='task_cancelled' if isinstance(exc, TaskCancelled) else 'keyboard_interrupt')
            for pending in order[order.index(name)+1:]:
                arms[pending].update(status='not_run', evidence=None)
            break
        except Exception as exc:
            fail(name, 'source', exc)
        finally:
            trace['physical_events'][name] = deepcopy(physical[start_event:]) if physical is not None else None
            trace['backend_events'][name] = deepcopy(backend_events[backend_start:]) if backend_events is not None else None
            trace['retrieval_events'][name] = deepcopy(retrieval_events[retrieval_start:]) if retrieval_events is not None else None
    failed = [a for a in arms.values() if a['status'] == 'failed']
    trace['status'] = ('interrupted' if interruption is not None else
        'scope_mismatch' if len(failed) == 3 and all(e['reason'] == 'scope_changed' for e in trace['errors']) else
        'seed_or_scope_invalid' if len(failed) == 3 and all(e['reason'] == 'source_identity_or_acl_changed' for e in trace['errors']) else
        'partial_failure' if failed else stop or 'completed')
    trace['budgets'] = {name: budgets[name].snapshot() if name in budgets else None for name in ARMS}
    trace['ranker_events'], trace['search_events'] = rank_events, search_events
    trace['cache_shared_across_arms'] = any(e['cache_shared_across_arms']
        for rows in rank_events.values() for e in rows)
    trace['provider_telemetry'] = deepcopy(getattr(models, 'events', None))
    # Raw ranking queries and complete evidence stay private, alongside arm queries.
    private = {'ranking_traces': rankings}
    result = ComponentResult(arms, trace, private)
    if isinstance(interruption, KeyboardInterrupt):
        interruption.component_result = result
        raise interruption
    return result


def authorized_search(db, user, contract, *, cfg, models, backend, corpus):
    """Compose existing scope/rebind/retrieval helpers with bounded transport.

    Construction and calls are explicitly caller-owned. This module never opens
    a database, reads a corpus, discovers a provider or registers an experiment.
    The returned search capability owns embedding accounting; its caller owns
    exactly one search reservation, as run_components/run_repair already do.
    """
    from app.retrieval import retrieve_authorized, routing_goal
    from app.security import readable_documents
    from scripts.run_supplement_contract_dev import contract_for, repair_scope_validator, scope_keys
    from scripts.run_evidence_selection_dev import rebind_rows
    from scripts.run_evidence_supplement_dev import live_candidates, supplement_configuration

    if contract_for(db, user, contract.question) != contract:
        raise _IntegrityError()
    validate, _ = repair_scope_validator(db, user, contract.question, cfg)
    frozen_docs = readable_documents(db, user)
    frozen_doc_versions = {(d.id, d.active_version_id) for d in frozen_docs}
    frozen_scope = scope_keys(contract.question, frozen_docs, cfg)
    # Explicit source scope freezes the original publication constraints. Without
    # a publisher clause, the original authorized document/version set is the scope.
    bound_pairs = {(d, v) for d, v, _ in contract.authorized_bindings}
    allowed = ({pair for request in contract.source_requests for pair in request.document_versions}
               if contract.source_requests else bound_pairs)
    configured = supplement_configuration(cfg).model_copy(update={
        'focused_generation_enabled': True, 'document_quota': None})
    events = []

    def check(original, query):
        current_docs = readable_documents(db, user)
        unchanged = (original == contract.question and contract_for(db, user, original) == contract
            and {(d.id, d.active_version_id) for d in current_docs} == frozen_doc_versions
            and scope_keys(original, current_docs, cfg) == frozen_scope)
        # An unresolved original request is a legitimate no-search stop. Its
        # frozen scope can be unchanged while a generated query is inadmissible.
        return unchanged and (query == contract.question or validate(original, query) is True)

    def scoped_docs(_db, _user):
        if contract_for(_db, _user, contract.question) != contract:
            raise _IntegrityError()
        return [d for d in readable_documents(_db, _user) if (d.id, d.active_version_id) in allowed]

    def search(args):
        allocation = current_budget()
        if allocation is None:
            raise ValueError('Authorized experiment search requires an execution budget')
        allocation.check()
        if not check(contract.question, args.query):
            raise _NoBridge(('scope_changed',), stage='scope')
        if (not contract.authorized_bindings or
                any(r.resolution != 'supported' for r in contract.source_requests)
                or contract.reason == 'unsupported_history'):
            raise _NoBridge(('original_scope_unresolved',), stage='scope')
        model_events = getattr(models, 'events', None)
        backend_events = getattr(backend, 'events', None)
        model_start = len(model_events) if model_events is not None else None
        backend_start = len(backend_events) if backend_events is not None else None
        event = {'query_sha256': _hash(args.query), 'status': 'failed',
                 'embedding_requests': None, 'backend_requests': None}
        events.append(event)
        began = time.monotonic()
        try:
            with routing_goal(contract.question, fuse=False):
                found = retrieve_authorized(db, user, args.query, top_k=args.top_k,
                    cfg=configured, models=BudgetedEmbedding(models, allocation), search=backend,
                    readable_documents_fn=scoped_docs)
            if found.blocked_reason:
                raise _NoBridge(('retrieval_scope_blocked',), stage='scope')
            if not check(contract.question, args.query):
                raise _NoBridge(('scope_changed',), stage='scope')
            rows = _allowed_rows(contract, live_candidates(db, user, found, corpus))
            event.update(status='ok', candidate_count=len(rows))
            return rows
        finally:
            event.update(wall_ms=round((time.monotonic()-began)*1000, 3),
                embedding_requests=sum(e['role'] == 'embedding' for e in model_events[model_start:])
                    if model_events is not None else None,
                backend_requests=len(backend_events)-backend_start if backend_events is not None else None)
    search.events = events
    search.backend_events = getattr(backend, 'events', None)

    return SimpleNamespace(search=search, validate_scope=check, events=events, cfg=configured,
                           reauthorize=lambda rows: rebind_rows(db, user, rows, corpus))
