from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from freeze_source_stage_v6 import stage
from summarize_source_stage_v6 import summarize


def test_stage_separates_candidate_admission_and_fact_coverage():
    gold = {'a', 'b'}
    assert stage(gold, {'a'}, {'a'}, 1, 2) == 'candidate_missing'
    assert stage(gold, gold, {'a'}, 1, 2) == 'quota_blocked'
    assert stage(gold, gold, gold, 2, 2) == 'fact_proxy_reached'
    assert stage(gold, gold, gold, 1, 2) is None


def test_stage_diagnosis_requires_actual_first_generation_fact_coverage():
    stages = ['candidate_missing', 'quota_blocked', 'fact_proxy_reached']
    items, rows = [], []
    for target in stages:
        for number in range(4):
            case_id = f'{target}-{number}'
            items.append({'id': case_id, 'target_stage': target, 'fact_proxy_reached': 2})
            rows.append({'id': case_id, 'answer_correct': False, 'status': 'answered',
                         'first_context_gold_documents': True, 'gold_facts_delivered_proxy': 2,
                         'gold_facts': 2, 'execution_error': None, 'total_prompt_tokens': 1,
                         'total_completion_tokens': 1})
    rows[-1]['gold_facts_delivered_proxy'] = 1
    report = summarize({'items': items, 'stages': stages}, {'records': rows})
    assert len(report['fact_complete_but_wrong_ids']) == 11
    assert report['observed_failure_layers']['document_or_passage_missing'] == 1
    assert report['by_frozen_stage']['fact_proxy_reached']['tasks'] == 4
