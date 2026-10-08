"""Read-only cost and failure summary; never rescore or rerun frozen tasks."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_source_facts_external import percentile


def summarize(run):
    if 'summary' not in run or len(run['records']) != 36:
        raise ValueError('Wait for all three arms on the same 12 tasks')
    result = {'scope': 'Frozen local paired experiment; literal scores and lexical passage proxies, independent human semantics pending.',
              'stages': {}, 'failures': {}, 'defaults_enabled': False,
              'independent_human_reviewed': 0, 'api_cost': 0}
    for arm in ['legacy', 'candidate', 'partial_facts']:
        rows = [row for row in run['records'] if row['arm'] == arm]
        keys = sorted({key for row in rows for key in row['timings']})
        result['stages'][arm] = {key: {
            'observations': sum(key in row['timings'] for row in rows),
            'p50_ms': percentile([row['timings'][key] for row in rows if key in row['timings']], .5),
            'p95_ms': percentile([row['timings'][key] for row in rows if key in row['timings']], .95),
        } for key in keys}
        result['failures'][arm] = [{key: row.get(key) for key in [
            'id', 'question_type', 'status', 'verdict', 'failure_stage_proxy',
            'first_context_gold_documents', 'gold_facts', 'gold_facts_delivered_proxy',
            'facts_accepted', 'facts_rejected', 'extraction_fallback', 'execution_error',
            'scope_repairs']} for row in rows if not row['answer_correct']]
    return result


if __name__ == '__main__':
    report = summarize(json.loads(Path('artifacts/p0-p2-external-paired-v1.json').read_text()))
    Path('artifacts/p0-p2-stage-failures-v1.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['stages'], ensure_ascii=False, indent=2))
