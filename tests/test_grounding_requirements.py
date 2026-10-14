"""Citation completeness feedback never manufactures evidence or changes verdicts."""
import json

from app.answer_contract import grounding_requirements
from app.answer_quality import recover_answer
from app.clients import Claim, GeneratedAnswer, Models
from app.config import settings
from app.qa import validate_claims


def evidence():
    return [
        {'id': 'E1', 'chunk_id': 'metric', 'document_id': 'd1', 'title': 'Observation',
         'text': 'East warehouse return rate is 4.5%.', 'metadata': {}},
        {'id': 'E2', 'chunk_id': 'rule', 'document_id': 'd2', 'title': 'Rule',
         'text': 'Escalate when the rate exceeds 3.0%. Morgan leads the review.', 'metadata': {}},
    ]


def test_requirements_use_only_question_and_supplied_evidence():
    rows = evidence()
    contract = grounding_requirements('Is escalation required, and who leads it?', rows)
    assert contract['acceptance_items']
    assert contract['evidence_ids'] == ['E1', 'E2']
    assert contract['not_a_support_verdict'] is True
    assert 'answer' not in contract and 'gold' not in json.dumps(contract)
    # No generated quote, citation or unconditional inferred fact enters the contract.
    assert 'claims' not in contract and 'quotes' not in contract


def test_missing_source_is_not_created_or_marked_supported():
    rows = evidence()[:1]
    contract = grounding_requirements('Compare the reports from The Guardian and BBC News.', rows)
    assert contract['evidence_ids'] == ['E1']
    assert contract['available_source_groups'] == []
    assert contract['not_a_support_verdict'] is True


def test_generation_transport_is_opt_in_and_server_binds_original_spans():
    calls = []
    models = Models(chat_backend=lambda body: calls.append(body) or {
        'message': {'content': json.dumps({'status': 'answered', 'claims': [
            {'text': 'The rate is 4.5%, exceeding 3.0%.', 'source_ids': ['E1:S1', 'E2:S1']}]})}})
    rows = evidence()
    ordinary, _ = models.generate('Is escalation required?', rows, check_conflict=False)
    assert 'grounding_requirements' not in json.loads(calls[-1]['messages'][1]['content'])
    candidate, _ = models.generate('Is escalation required?', rows, check_conflict=False,
                                  grounding_requirements=grounding_requirements('Is escalation required?', rows))
    inputs = json.loads(calls[-1]['messages'][1]['content'])
    assert inputs['grounding_requirements']['evidence_ids'] == ['E1', 'E2']
    assert candidate == ordinary
    assert candidate.claims[0].quotes == [rows[0]['text'], rows[1]['text']]
    assert validate_claims(candidate, rows)[1] == 'answered'


def test_opt_in_repair_shows_actual_bindings_but_does_not_attach_them():
    rows = evidence()
    bad = GeneratedAnswer(answerable=True, claims=[
        Claim(text='The rate is 4.5%, exceeding 3.0%.', evidence_ids=['E1'], quotes=[rows[0]['text']]),
        Claim(text='Morgan leads the review.', evidence_ids=['E2'], quotes=[rows[1]['text']]),
    ])
    calls = []

    class Model:
        def generate(self, _question, _evidence, **options):
            calls.append(options)
            # Deliberately repeat the invalid draft: extra context must not bypass validation.
            return bad, {}

    result, usage = recover_answer(Model(), 'Is escalation required and who leads?', rows,
                                   grounding_feedback=True)
    assert len(calls) == 2
    assert calls[0]['grounding_requirements']['evidence_ids'] == ['E1', 'E2']
    repair = calls[1]['repair_context']
    assert repair['failed_claims'][0]['bound_quotes'] == [rows[0]['text']]
    assert repair['retained_claim_bindings'][0]['quotes'] == [rows[1]['text']]
    assert repair['retained_claims_are_only_mechanically_checked'] is True
    assert validate_claims(result, rows)[1] == 'verification_failed'
    assert usage['answer_recovery']['repair_attempts'] == 1


def test_absent_threshold_and_date_remain_invalid_with_feedback():
    rows = evidence()[:1]
    draft = GeneratedAnswer(answerable=True, claims=[Claim(
        text='On October 31, 2023 the rate was 4.5%, exceeding 3.0%.',
        evidence_ids=['E1'], quotes=[rows[0]['text']])])

    class Model:
        def generate(self, *_args, **_kwargs):
            return draft, {}

    answer, _ = recover_answer(Model(), 'Was the threshold exceeded on October 31, 2023?', rows,
                               grounding_feedback=True)
    assert validate_claims(answer, rows)[1] == 'verification_failed'


def test_default_recovery_input_and_configuration_unchanged():
    calls = []

    class Model:
        def generate(self, *_args, **kwargs):
            calls.append(kwargs)
            return GeneratedAnswer(answerable=True, claims=[Claim(
                text='The rate is 4.5%.', evidence_ids=['E1'], quotes=[evidence()[0]['text']])]), {}

    recover_answer(Model(), 'What is the rate?', evidence())
    assert 'grounding_requirements' not in calls[0]
    assert settings().verdict_protocol == 'legacy'
    assert settings().answer_quality_enabled is False


def test_numeric_validator_does_not_claim_branch_entailment():
    # This is a known semantic gap, not a false rejection. No generic numeric
    # presence test may pretend it verifies that the right branch was cited.
    rows = evidence()
    wrong_branch = GeneratedAnswer(answerable=True, claims=[Claim(
        text='Rates not exceeding 3.0% require no escalation.',
        evidence_ids=['E2'], quotes=[rows[1]['text']])])
    assert validate_claims(wrong_branch, rows)[1] == 'answered'
