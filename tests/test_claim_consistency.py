from app.answer_contract import source_audit
from app.claim_consistency import explicit_consistency, reconcile_verdict, source_polarity_issue
from app.temporal import effective_interval, select_effective_versions
from app.config import Settings
from app.qa import answer_verdict


def test_source_negation_requires_same_subject_and_positive_restatement():
    quote = 'Ellison did not adequately protect customer funds.'
    assert source_polarity_issue('Ellison adequately protected customer funds.', [quote])
    assert not source_polarity_issue('Ellison did not adequately protect customer funds.', [quote])
    assert not source_polarity_issue('Wang adequately protected customer funds.', [quote])
    assert not source_polarity_issue('The new policy requires approval.',
                                     ['The old policy does not require approval.'])
    assert not source_polarity_issue('Alex supported it.', ['Dana did not support it. Alex supported it.'])
    assert not source_polarity_issue('乙可以出场。', ['甲不可以出场，乙可以出场。'])
    assert not source_polarity_issue('新政策要求审批。', ['旧政策不要求审批。'])


def test_new_polarity_audit_is_opt_in():
    quote = 'The policy does not require approval.'
    claim = 'The policy requires approval.'
    assert source_audit(claim, [quote]) == []
    assert source_audit(claim, [quote], check_polarity=True) == ['explicit_source_negation_lost']
    assert source_audit('Only A confirmed the event; B did not confirm it.',
                        ['A confirmed the event. B may have seen it but did not confirm.']) == []
    assert source_audit('只有甲报告确认事件已发生，乙报告尚未确认。',
                        ['甲报告：事件已发生。乙报告：事件可能发生，尚未确认。']) == []


def test_verdict_correction_is_bound_to_one_requested_relation():
    question = 'Are the launch dates consistent despite different operating hours?'
    claim = {'text': 'The launch dates are not consistent.', 'evidence_ids': ['c1']}
    verdict = {'value': 'yes', 'claim_index': 1, 'evidence_ids': ['c1']}
    fixed = reconcile_verdict(question, [claim], verdict)
    assert fixed['value'] == 'no' and fixed['evidence_ids'] == ['c1']
    assert reconcile_verdict(question, [{'text': 'The operating hours are not consistent.'}], verdict) == verdict
    assert explicit_consistency(question, 'The launch dates are not entirely consistent.') == 'no'
    assert explicit_consistency(question, 'The launch dates are consistent, but the operating hours differ.') == 'yes'
    assert explicit_consistency(question, 'The launch dates are consistent while the operating hours are not consistent.') == 'yes'
    assert explicit_consistency(question, 'The launch dates are not inconsistent.') is None
    assert explicit_consistency('Are the reports consistent?',
                                'The dates are consistent. The hours are not consistent.') is None


def test_answer_verdict_path_only_reconciles_when_opted_in(monkeypatch):
    class Model:
        def decide_verdict(self, _question, _claims):
            return type('Verdict', (), {'verdict': 'yes', 'claim_index': 1})(), {}

    question = 'Are the launch dates consistent?'
    claims = [{'text': 'The launch dates are not consistent.', 'evidence_ids': ['c1']}]
    cfg = Settings(claim_consistency_enabled=False)
    monkeypatch.setattr('app.qa.settings', lambda: cfg)
    assert answer_verdict(Model(), question, claims, 'answered')[0]['value'] == 'yes'
    cfg.claim_consistency_enabled = True
    result = answer_verdict(Model(), question, claims, 'answered')[0]
    assert result['value'] == 'no' and result['evidence_ids'] == ['c1']


def test_shared_year_stays_within_one_explicit_effective_range():
    old = effective_interval('Published August 20, 2026. Policy v1: effective January 1 through August 31, 2026; limit 30 minutes.')
    new = effective_interval('Policy v2: effective September 1 through September 30, 2026; limit 15 minutes.')
    assert old['status'] == new['status'] == 'explicit'
    assert old['valid_from'] == '2026-01-01' and old['valid_to'] == '2026-08-31'
    assert new['valid_from'] == '2026-09-01' and new['valid_to'] == '2026-09-30'
    assert old['provenance'] == 'shared_year_within_effective_range'
    assert select_effective_versions('What applied on September 15, 2026?', [
        {'version_id': 'v1', 'effective_interval': old},
        {'version_id': 'v2', 'effective_interval': new},
    ])['selections'][0]['version_id'] == 'v2'
    assert effective_interval('Published August 20, 2026. Effective January 1 through August 31.')['status'] == 'unknown'
    assert effective_interval('Effective October 1 through September 30, 2026.')['status'] == 'unknown'
