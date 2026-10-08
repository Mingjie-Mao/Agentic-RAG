import json

import pytest

from app.answer_contract import claim_scope_issues, comparison_dimension, source_audit
from app.clients import Claim, GeneratedAnswer, Models
from app.config import Settings
from app.qa import validate_claims
from app.verdict import StructuredVerdict, constrained_schema, resolve_verdict


@pytest.mark.parametrize('question,dimension', [
    ('Are launch dates consistent while operating hours differ?', 'launch dates'),
    ('两份通知的生效日期是否一致，发布日期有什么不同？', '生效日期'),
    ('Was fantasy football strategy consistent despite availability changes?', 'fantasy football strategy'),
])
def test_dimension_is_exact_requested_attribute(question, dimension):
    assert comparison_dimension(question) == dimension
    schema = constrained_schema(question)
    assert schema['properties']['comparison_dimension']['enum'] == ['', dimension]


@pytest.mark.parametrize('text', [
    'The article did not claim that the merger was unlawful.',
    'The report focuses on the jury, without attributing a motive.',
    '整份通知从未说明审批人。',
    'The whole report never mentions fines, based on provided excerpts.',
    'The article did not mention fines. Current excerpts mention costs.',
])
def test_missing_chunk_does_not_prove_document_absence(text):
    assert claim_scope_issues(text) == ['unbounded_document_absence']


@pytest.mark.parametrize('text', [
    'The provided excerpts of the report do not mention fines.',
    '当前片段未见审批人信息，无法判断整份通知。',
    'The policy does not require approval.',
    '律师林表示交易并不违法。',
])
def test_bounded_absence_and_explicit_factual_denials_are_not_blocked(text):
    assert claim_scope_issues(text) == []


def test_publication_assertion_over_reported_opinion_is_a_diagnostic_only():
    source = ['The defense argued that the merger was lawful.']
    assert source_audit('The article claimed that the merger was lawful.', source) == ['possible_speaker_attribution_loss']
    assert source_audit('The article reports that the defense argued the merger was lawful.', source) == []


def test_scope_guard_and_generation_contract_are_applied_to_real_code_path(monkeypatch):
    cfg = Settings(answer_contract_enabled=True)
    monkeypatch.setattr('app.clients.settings', lambda: cfg)
    monkeypatch.setattr('app.qa.settings', lambda: cfg)
    calls = []
    models = Models(chat_backend=lambda body: calls.append(body) or {
        'message': {'content': json.dumps({'status':'answered', 'claims':[{
            'text':'The article did not mention fines.', 'source_ids':['E1:S1']
        }]})}
    })
    evidence = [{'id':'E1','chunk_id':'chunk1','document_id':'doc1','title':'Report',
                 'text':'The analyst said the merger would lower costs.'}]
    generated, usage = models.generate('Does the entire article avoid mentioning fines?', evidence, check_conflict=False)
    assert validate_claims(generated, evidence) == ([], 'verification_failed')
    body = json.loads(calls[0]['messages'][1]['content'])
    assert body['source_attribution_cues'] == {'E1:S1':['said']}
    assert body['question_contract']['absence_scope']
    assert usage['source_audit'][0]['issues'] == ['unbounded_document_absence']
    cfg.answer_contract_enabled = False
    assert validate_claims(generated, evidence)[1] == 'answered'


def test_explicit_negative_fact_still_returns_cited_answer(monkeypatch):
    monkeypatch.setattr('app.qa.settings', lambda: Settings(answer_contract_enabled=True))
    evidence = [{'id':'E1','chunk_id':'chunk1','text':'The policy does not require approval.'}]
    answer = GeneratedAnswer(answerable=True, claims=[Claim(text=evidence[0]['text'], evidence_ids=['E1'], quotes=[evidence[0]['text']])])
    claims, status = validate_claims(answer, evidence)
    assert status == 'answered' and claims[0]['evidence_ids'] == ['chunk1']


def test_comparison_alias_cannot_override_bound_dimension():
    question = 'Are launch dates consistent while operating hours differ?'
    decision = StructuredVerdict(operator='comparison', question_complete=True,
        comparison_dimension='operating hours', propositions=[{
            'question_span':question.rstrip('?'), 'truth':'contradicted','claim_indices':[1]
        }])
    output = resolve_verdict(decision, question, [{'evidence_ids':['c1']}], constrained=True)
    assert output['value'] == 'unclear'
    assert 'comparison_dimension_not_bound' in output['validation_issues']


def test_explicit_source_absence_is_different_from_inferred_absence(monkeypatch):
    monkeypatch.setattr('app.qa.settings', lambda: Settings(answer_contract_enabled=True))
    text = 'The notice does not specify an approval deadline.'
    evidence = [{'id':'E1','chunk_id':'c1','text':text}]
    answer = GeneratedAnswer(answerable=True, claims=[Claim(text=text, evidence_ids=['E1'], quotes=[text])])
    assert validate_claims(answer, evidence)[1] == 'answered'
    # An invented quote cannot open the exception.
    answer.claims[0].quotes = ['The notice does not specify a retention deadline.']
    assert validate_claims(answer, evidence) == ([], 'verification_failed')


@pytest.mark.parametrize('question,expected', [
    ('Despite different publication dates, are effective dates consistent?', 'effective dates'),
    ('发布日期不同，生效日期是否一致？', '生效日期'),
    ('Are launch dates and operating hours consistent?', 'Are launch dates and operating hours consistent'),
])
def test_background_attributes_cannot_replace_requested_comparison(question, expected):
    assert comparison_dimension(question) == expected


@pytest.mark.parametrize('question,expected', [
    ('Does one report imply less control over privacy compared to the other article about experiences?', 'control over privacy'),
    ('Are publication dates consistent despite differences in personal privacy?', 'publication dates'),
    ('两份报道的个人隐私控制权是否一致，收入不同？', '个人隐私控制权'),
])
def test_privacy_facet_is_bound_to_question_without_inventing_creative_control(question, expected):
    assert comparison_dimension(question) == expected


@pytest.mark.parametrize('operator,value,required', [
    ('comparison', 'yes', True), ('comparison', 'no', True),
    ('atomic', 'yes', True), ('all', 'yes', True), ('any', 'no', True),
    ('all', 'no', False), ('any', 'yes', False),
])
def test_source_group_proof_requires_both_sides_except_logical_short_circuit(operator, value, required):
    from app.answer_contract import bind_verdict_source_groups
    question = 'Are launch dates consistent in The Guardian and BBC News - Technology?'
    rows = [dict(id='E1', chunk_id='c1', document_id='a', version_id='v1', metadata={'source':'The Guardian'}),
            dict(id='E2', chunk_id='c2', document_id='b', version_id='v2', metadata={'source':'BBC News - Technology'})]
    verdict = dict(value=value, operator=operator, evidence_ids=['c1'], claim_index=1,
                   claim_indices=[1], question_complete=True)
    result = bind_verdict_source_groups(question, verdict, rows)
    assert (result['value'] == 'unclear') is required
    if required:
        assert result['evidence_ids'] == [] and result['question_complete'] is False
        assert result['validation_issues'] == ['missing_source_group_proof']
    verdict['evidence_ids'] = ['c1','c2']
    assert bind_verdict_source_groups(question, verdict, rows)['value'] == value


def test_named_publication_cannot_borrow_another_desks_quote(monkeypatch):
    monkeypatch.setattr('app.qa.settings', lambda: Settings(answer_quality_enabled=True))
    rows = [dict(id='E1', chunk_id='c1', document_id='a', version_id='v1',
                 metadata={'source':'BBC News - Technology'}, text='Dana approved the proposal.'),
            dict(id='E2', chunk_id='c2', document_id='b', version_id='v2',
                 metadata={'source':'BBC News - Sport'}, text='Dana approved the proposal.')]
    claim = Claim(text='BBC News - Technology reports Dana approved the proposal.',
                  evidence_ids=['E2'], quotes=[rows[1]['text']])
    assert validate_claims(GeneratedAnswer(answerable=True, claims=[claim]), rows)[1] == 'verification_failed'
    claim.evidence_ids = ['E1']
    assert validate_claims(GeneratedAnswer(answerable=True, claims=[claim]), rows)[1] == 'answered'
    claim.text = 'Dana approved the proposal.'
    claim.evidence_ids = ['E2']
    assert validate_claims(GeneratedAnswer(answerable=True, claims=[claim]), rows)[1] == 'answered'


@pytest.mark.parametrize('question,allowed', [
    ('Does the report discuss game accessibility?', False),
    ('Can I access the report?', True), ('Do I have permissions?', True),
    ('Was unauthorized access possible?', True), ('撤权后还可以读吗？', True),
])
def test_accessibility_is_not_an_access_permission_task(monkeypatch, question, allowed):
    from app.clients import agent_decision_schema
    monkeypatch.setattr('app.clients.settings', lambda: Settings(answer_quality_enabled=True))
    choices = {branch['properties']['action']['enum'][0] for branch in agent_decision_schema([{'handles':['c1']}], question)['anyOf']}
    assert ('verify_chunk_access' in choices) is allowed


def test_explicit_third_party_attribution_is_a_legal_alternative_proof_path(monkeypatch):
    from app.answer_contract import bind_verdict_source_groups
    monkeypatch.setattr('app.qa.settings', lambda: Settings(answer_quality_enabled=True))
    rows = [dict(id='E1', chunk_id='c1', document_id='a', version_id='v1', title='A',
                 metadata={'source':'The Guardian'}, text='Launch date is Monday.'),
            dict(id='E2', chunk_id='c2', document_id='b', version_id='v2', title='B',
                 metadata={'source':'BBC News - Technology'}, text='Launch date is Monday.'),
            dict(id='E3', chunk_id='c3', document_id='c', version_id='v3', title='C',
                 metadata={'source':'Reuters'}, text='BBC News - Technology reports the launch date is Monday.')]
    claim = Claim(text='BBC News - Technology reports the launch date is Monday.',
                  evidence_ids=['E3'], quotes=[rows[2]['text']])
    checked, status = validate_claims(GeneratedAnswer(answerable=True, claims=[claim]), rows)
    assert status == 'answered' and checked[0]['evidence_ids'] == ['c3']
    verdict = dict(value='yes', operator='comparison', evidence_ids=['c1','c3'],
                   claim_index=1, claim_indices=[1,2], question_complete=True)
    question = 'Are launch dates consistent in The Guardian and BBC News - Technology?'
    assert bind_verdict_source_groups(question, verdict, rows, checked)['value'] == 'yes'
    # An uncited third-party attribution cannot rescue one-sided proof.
    verdict['evidence_ids'] = ['c1']
    assert bind_verdict_source_groups(question, verdict, rows, checked)['value'] == 'unclear'


@pytest.mark.parametrize("case", ["absent_named", "relay_quote", "no_snapshot", "snapshot_not_covering", "own_source"])
def test_claim_naming_a_readable_but_unretrieved_publication_needs_that_provenance(case):
    from app.answer_contract import publication_binding_issues
    from app.retrieval import _READABLE_SOURCES

    evidence = [
        {"id": "E1", "chunk_id": "c1", "document_id": "ind", "version_id": "v1", "title": "Swift profile",
         "text": "x", "metadata": {"source": "The Independent - Life and Style"}},
        {"id": "E2", "chunk_id": "c2", "document_id": "gua", "version_id": "v2", "title": "Britney",
         "text": "y", "metadata": {"source": "The Guardian"}},
    ]
    names = frozenset({"The Independent - Life and Style", "The Guardian", "BBC News - Entertainment & Arts"})
    snapshot = {"no_snapshot": None, "snapshot_not_covering": (frozenset({"other"}), names)}.get(
        case, (frozenset({"ind", "gua", "bbc"}), names))
    claim = "The BBC News - Entertainment & Arts article implies Taylor Swift keeps control of her privacy."
    quote = "She told BBC News - Entertainment & Arts that some things stay private." if case == "relay_quote" \
        else "There's going to be a whole chaotic situation outside the restaurant."
    refs = ["E1"]
    if case == "own_source":
        claim, refs, quote = "The Guardian suggests Britney had less control.", ["E2"], "y"
    token = _READABLE_SOURCES.set(snapshot)
    try:
        issues = publication_binding_issues(claim, refs, evidence, [quote])
    finally:
        _READABLE_SOURCES.reset(token)
    assert issues == (["publication_reference_mismatch"] if case == "absent_named" else [])
