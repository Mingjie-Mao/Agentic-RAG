"""Bounded, source-blind search-query decomposition for a local model."""
from pydantic import BaseModel, ConfigDict, Field


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    queries: list[str] = Field(min_length=2, max_length=3)


def plan_search_queries(models, question, cfg):
    """Return search text only. The model sees no corpus, ACLs, answer or gold."""
    result = models._chat({
        'model': cfg.agent_policy_model,
        'stream': False, 'keep_alive': '30m',
        'format': SearchPlan.model_json_schema(),
        'messages': [
            {'role': 'system', 'content': (
                'Turn a multi-document question into 2 or 3 short, DISTINCT search queries, '
                'one per evidence-bearing topic, event or source-side fact. Preserve named '
                'entities and specific actions. Do not answer the question, invent facts, '
                'include expected conclusions, or repeat the whole question. JSON only.')},
            {'role': 'user', 'content': question[:1200]},
        ],
        'options': {'temperature': 0, 'seed': 42, 'num_ctx': 4096, 'num_predict': 140},
    })
    plan = SearchPlan.model_validate_json(result['message']['content'])
    queries = list(dict.fromkeys(query.strip()[:240] for query in plan.queries
                                 if 4 <= len(query.strip()) <= 240))
    if len(queries) < 2:
        raise ValueError('Search plan has fewer than two distinct usable queries')
    return queries, {'prompt_tokens': result.get('prompt_eval_count', 0),
                     'completion_tokens': result.get('eval_count', 0)}
