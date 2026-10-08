import json
from types import SimpleNamespace

from app.query_planner import plan_search_queries
from app.retrieval import retrieve_authorized
from test_source_routing import article


def test_search_planner_accepts_only_distinct_bounded_queries():
    calls = []
    models = SimpleNamespace(_chat=lambda body: calls.append(body) or {
        'message': {'content': json.dumps({'queries': ['FTX trial testimony', 'FTX fraud allegations']})},
        'prompt_eval_count': 12, 'eval_count': 8})
    queries, usage = plan_search_queries(models, 'Compare two reports on FTX.',
                                         SimpleNamespace(agent_policy_model='local-7b'))
    assert queries == ['FTX trial testimony', 'FTX fraud allegations']
    assert usage == {'prompt_tokens': 12, 'completion_tokens': 8}
    assert calls[0]['model'] == 'local-7b'
    assert calls[0]['options']['num_predict'] <= 140


def test_planned_queries_keep_acl_and_account_for_model_cost():
    documents = {'a': article('a', 'TechCrunch'), 'b': article('b', 'TechCrunch'),
                 'z': article('z', 'Wired')}
    chunks = {key: (SimpleNamespace(id=key, text='fact', locator={}),
                    SimpleNamespace(id='v' + key), document)
              for key, document in documents.items()}
    scopes = []
    class Search:
        def retrieve_hybrid(self, text, _vector, _tenant, versions, _size):
            scopes.append((text, set(versions)))
            selected = 'a' if 'trial' in text else 'b' if 'fraud' in text else 'z'
            return [{'chunk_id': selected, 'score': 1.0}] if 'v' + selected in versions else []
    models = SimpleNamespace(
        embed=lambda texts: [[0.0] for _ in texts],
        _chat=lambda _body: {'message': {'content': json.dumps({
            'queries': ['FTX trial testimony', 'FTX fraud allegations']})},
            'prompt_eval_count': 12, 'eval_count': 8})
    question = ('Between the TechCrunch report about the FTX trial and the later '
                'TechCrunch report about fraud allegations, were its actions portrayed differently?')
    result = retrieve_authorized(
        None, SimpleNamespace(tenant_id='t'), question,
        cfg=SimpleNamespace(top_k=4, retrieval_mode='hybrid', min_similarity=0.0,
                            context_token_budget=5000, source_query_plan=True,
                            agent_policy_model='local-7b'),
        models=models, search=Search(), top_k=6,
        readable_documents_fn=lambda *_: list(documents.values()),
        require_chunk_fn=lambda _db, _user, key, **_kw: chunks[key],
    )
    assert result.planning_tokens == 20 and result.planned_queries_count == 2
    assert {row['document_id'] for row in result.evidence} >= {'a', 'b'}
    assert scopes[0][1] == scopes[1][1] == {'va', 'vb'}
    assert scopes[2][1] == {'va', 'vb', 'vz'}
