"""Keep the frozen retrieval-mode comparison's arms and denominators aligned."""
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_p0_p2_external as runner
from summarize_bm25_gate import evaluate


def test_retrieval_mode_arms_select_only_the_intended_mode():
    cfg = SimpleNamespace(retrieval_mode='hybrid')
    runner.arm_settings(cfg, 'bm25')
    assert cfg.retrieval_mode == 'bm25'
    assert not cfg.source_focus_queries
    assert not cfg.passage_window_enabled
    runner.arm_settings(cfg, 'hybrid')
    assert cfg.retrieval_mode == 'hybrid'


def test_report_pairs_the_first_manifest_arm_with_each_candidate(monkeypatch):
    rows = [{'arm': arm, 'question_type': 'comparison_query',
             'first_context_gold_documents': True, 'gold_facts': 2,
             'gold_facts_delivered_proxy': 1} for arm in ('hybrid', 'bm25')]
    result = {'records': rows}
    monkeypatch.setattr(runner, 'summarize', lambda records: {'count': len(records)})
    monkeypatch.setattr(runner, 'gate', lambda summary, records, _criteria: {
        'baseline_count': summary['legacy']['count'],
        'candidate_count': summary['source_facts']['count'],
        'mapped_arms': [row['arm'] for row in records],
    })
    runner.report(result, {'arms': ['hybrid', 'bm25'], 'gate': {}})
    assert result['gates']['bm25'] == {
        'baseline_count': 1, 'candidate_count': 1,
        'mapped_arms': ['legacy', 'source_facts'],
    }


def test_frozen_bm25_gate_does_not_ignore_unchanged_gold_document_count():
    manifest = {'retrieval_gate': {
        'minimum_all_gold_documents_delta': 0, 'minimum_gold_documents_delta': 1,
        'minimum_gold_fact_proxy_delta': 0, 'maximum_p50_latency_ratio': 1.0}}
    retrieval = {'summary': {
        'baseline': {'all_gold_documents': 5, 'gold_documents_found': 20,
                     'gold_facts_delivered_proxy': 9, 'p50_retrieval_ms': 469},
        'bm25': {'all_gold_documents': 6, 'gold_documents_found': 20,
                 'gold_facts_delivered_proxy': 11, 'p50_retrieval_ms': 169}}}
    answers = {'gates': {'bm25': {'decision': 'eligible_for_larger_validation'}}}
    result = evaluate(manifest, retrieval, answers)
    assert result['retrieval_checks']['gold_document_count'] is False
    assert result['decision'] == 'keep_default_off'


def test_single_arm_stage_report_keeps_stage_denominators(monkeypatch):
    monkeypatch.setattr(runner, 'summarize', lambda records: {'count': len(records)})
    result = {'records': [
        {'arm': 'hybrid', 'question_type': 'inference_query', 'target_stage': 'candidate_missing',
         'first_context_gold_documents': False, 'gold_facts': 2,
         'gold_facts_delivered_proxy': 0},
        {'arm': 'hybrid', 'question_type': 'comparison_query', 'target_stage': 'fact_proxy_reached',
         'first_context_gold_documents': True, 'gold_facts': 2,
         'gold_facts_delivered_proxy': 2},
    ]}
    runner.report(result, {})
    assert result['summary']['hybrid']['count'] == 2
    assert result['by_stage']['candidate_missing']['hybrid']['count'] == 1
    assert result['by_stage']['fact_proxy_reached']['hybrid']['count'] == 1
    assert result['gates'] == {}
