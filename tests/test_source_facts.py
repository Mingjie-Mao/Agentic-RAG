import json

from app.clients import Claim, GeneratedAnswer, Models, evidence_spans
from app.config import settings
from app.qa import answer_verdict, validate_claims
from app.source_facts import FactAnswer, conclusion, dates, public_facts, validate_facts


def evidence(text, number=1, title='Policy'):
    return {'id':f'E{number}','chunk_id':f'c{number}','document_id':f'd{number}',
            'version_id':f'v{number}','title':title,'text':text,'locator':{}}


def wire(source_id='E1:S1', **changes):
    values={'source_id':source_id,'subject':'East store','speaker':'','modality':'asserted',
            'attribute':'opens','value':'June 4, 2026','kind':'opening_date',
            'time_source_id':'','time_value':'','time_end':'','time_role':'none'}
    return {**values,**changes}


def checked(rows, facts):
    sources,_=evidence_spans(rows)
    return validate_facts(FactAnswer(answerable=True,facts=facts),sources,rows)


def test_fields_cannot_be_present_somewhere_else_in_the_same_chunk():
    facts,issues=checked([evidence('East store opens June 4, 2026。West store opens July 8, 2026。')],
                         [wire(subject='West store',value='July 8, 2026')])
    assert not facts and 'field_not_in_source' in issues[0]['issues']


def test_reported_claim_retains_full_speaker_and_modality_span():
    rows=[evidence('According to the audit team, Orion may have understated costs.')]
    facts,issues=checked(rows,[wire(subject='Orion',speaker='audit team',modality='uncertain',
                                  attribute='understated',value='costs',kind='text')])
    assert not issues
    assert facts[0]['quote'] in facts[0]['claim']['text']
    assert 'may have' in facts[0]['claim']['text']
    _,issues=checked(rows,[wire(subject='Orion',speaker='audit team',attribute='understated',
                               value='costs',kind='text')])
    assert 'reported_or_uncertain_span_marked_asserted' in issues[0]['issues']


def test_speaker_literal_does_not_establish_speaker_role():
    facts,issues=checked([evidence('Analyst Dana said Orion may save costs.')],
                         [wire(subject='Orion',speaker='Orion',modality='uncertain',
                               attribute='save',value='costs',kind='text')])
    assert not facts and 'speaker_not_bound_to_reporting_cue' in issues[0]['issues']


def test_comparison_only_uses_requested_attribute_not_distracting_hours():
    rows=[evidence('East store opens June 4, 2026 at 09:00.',1),
          evidence('East store opens June 4, 2026 at 10:00.',2)]
    facts,issues=checked(rows,[wire(),wire('E2:S1')])
    assert not issues
    assert conclusion('Are the opening dates consistent despite different operating hours?',facts)['value']=='yes'
    assert conclusion('Are the effective dates consistent?',facts)['value']=='unclear'
    rows[1]['text']='East store opens June 5, 2026 at 10:00.'
    facts,_=checked(rows,[wire(),wire('E2:S1',value='June 5, 2026')])
    assert conclusion('Are the opening dates consistent?',facts)['value']=='no'


def test_different_subject_and_missing_side_do_not_mean_date_conflict():
    rows=[evidence('East store opens June 4, 2026.',1),evidence('West store opens June 5, 2026.',2)]
    facts,_=checked(rows,[wire(),wire('E2:S1',subject='West store',value='June 5, 2026')])
    assert conclusion('Are the opening dates consistent?',facts)['value']=='unclear'
    assert conclusion('Are the opening dates consistent?',facts[:1])['value']=='unclear'
    assert conclusion('Are the opening dates not consistent?',facts[:1])['value']=='unclear'


def test_publication_date_cannot_become_an_effective_date():
    rows=[evidence('Policy published August 20, 2026; effective September 1, 2026。Policy limit is 15 minutes。')]
    extracted=wire('E1:S2',subject='Policy',attribute='limit',value='15 minutes',kind='number',
                   time_source_id='E1:S1',time_role='effective',time_value='August 20, 2026')
    facts,issues=checked(rows,[extracted])
    assert not facts and 'effective_date_not_locally_bound' in issues[0]['issues']


def test_explicit_intervals_select_historical_values_and_boundaries():
    rows=[evidence('Policy effective January 1, 2026 through August 31, 2026。Policy limit is 30 minutes。',1),
          evidence('Policy effective September 1, 2026 through September 30, 2026。Policy limit is 15 minutes。',2)]
    base={'subject':'Policy','attribute':'limit','kind':'number','time_role':'effective'}
    facts,issues=checked(rows,[
        wire('E1:S2',**base,value='30 minutes',time_source_id='E1:S1',time_value='January 1, 2026',time_end='August 31, 2026'),
        wire('E2:S2',**base,value='15 minutes',time_source_id='E2:S1',time_value='September 1, 2026',time_end='September 30, 2026'),
    ])
    assert not issues
    result=conclusion('What was the Policy limit on August 31, 2026 and September 1, 2026?',facts)
    assert result['value']=='not_applicable'
    assert [row['value'] for row in result['selections']]==['30 minutes','15 minutes']
    assert conclusion('What was the Policy limit on October 2, 2026?',facts)['value']=='unclear'
    facts[1]['time']['end_iso']=None
    assert conclusion('What was the Policy limit on September 15, 2026?',facts)['value']=='unclear'


def test_effective_time_from_another_chunk_cannot_be_borrowed():
    rows=[evidence('Policy limit is 15 minutes。',1),evidence('Policy effective September 1, 2026。',2)]
    facts,issues=checked(rows,[wire(subject='Policy',attribute='limit',value='15 minutes',kind='number',
                                  time_source_id='E2:S1',time_value='September 1, 2026',time_role='effective')])
    assert not facts and 'time_not_from_same_evidence' in issues[0]['issues']


def test_date_parser_requires_a_year_and_handles_dates_touching_chinese():
    assert dates('在2026-09-01和2026年10月1日。')[0][1].isoformat()=='2026-09-01'
    assert dates('4 June 2026')[0][1].isoformat()=='2026-06-04'
    assert not dates('June 4') and not dates('2026-02-30')


def test_public_bundle_and_verdict_only_use_final_validated_citations(monkeypatch):
    rows=[evidence('East store opens June 4, 2026.',1),evidence('East store opens June 4, 2026.',2)]
    facts,_=checked(rows,[wire(),wire('E2:S1')])
    generated=GeneratedAnswer(answerable=True,claims=[Claim(**fact['claim']) for fact in facts],facts=facts)
    claims,status=validate_claims(generated,rows)
    bundle=public_facts(facts,claims)
    assert all('claim' not in fact for fact in bundle)
    assert len(public_facts(facts,claims[:1]))==1
    class NoJudge:
        def decide_verdict(self,*args):
            raise AssertionError('Program comparison does not add a judge call')
    verdict,usage=answer_verdict(NoJudge(),'Are the opening dates consistent?',claims,status,facts=bundle)
    assert verdict['value']=='yes' and set(verdict['evidence_ids'])=={'c1','c2'} and not usage


def test_client_extraction_is_one_call_and_unknown_relations_do_not_get_legacy_verdict(monkeypatch):
    monkeypatch.setattr(settings(),'source_facts_enabled',True)
    rows=[evidence('East store opens June 4, 2026.')]
    class Stub(Models):
        def _chat(self,body):
            self.body=body
            return {'message':{'content':json.dumps({'answerable':True,'facts':[wire()]})},
                    'eval_count':100,'prompt_eval_count':200}
    model=Stub()
    generated,usage=model.generate('When does East store open?',rows)
    claims,status=validate_claims(generated,rows)
    assert status=='answered' and usage['generation_protocol']=='source_bound_facts_v1'
    assert generated.facts and model.body['options']['num_predict']==700
    verdict,_=answer_verdict(object(),'Are the opening dates consistent?',claims,status,
                            facts=public_facts(generated.facts,claims))
    assert verdict['value']=='unclear'


def test_revocation_hides_persisted_fact_fields_and_quotes(monkeypatch):
    from agent.controller import create_task, run_task, task_payload
    from agent.tools import ToolResult
    from app.models import Document
    from test_agent import agent_db
    monkeypatch.setattr(settings(),'source_facts_enabled',True)
    db,user=agent_db()
    class Tools:
        def __init__(self):
            self.user=user
        def call(self,name,args):
            return ToolResult('ok',{'matches':[{'title':'恢复政策','snippet':'RPO为15分钟。'}]},['c2'],{})
    class Stub(Models):
        def _chat(self,body):
            source_id=body['format']['$defs']['ExtractedFact']['properties']['source_id']['enum'][0]
            return {'message':{'content':json.dumps({'answerable':True,'facts':[
                wire(source_id,subject='RPO',attribute='RPO',value='15 分钟',kind='number')]})}}
    task=create_task(db,user,'RPO是多少？','workflow',4)
    run_task(db,user,task,models=Stub(),tools=Tools())
    payload=task_payload(db,user,task)
    assert payload['result']['facts'][0]['value']=='15 分钟'
    db.get(Document,'doc-a').read_groups=['finance']
    db.get(Document,'doc-a').owner_id='another-owner'
    db.commit()
    revoked=task_payload(db,user,task)
    assert revoked['result']['status']=='access_changed'
    assert 'facts' not in revoked['result']
    assert '15 分钟' not in json.dumps(revoked,ensure_ascii=False)


def test_fact_validator_rejects_partial_numbers_and_lost_units_end_to_end():
    for supplied,original in [('50','250%'),('1','16'),('2%','12%'),('15','15 minutes')]:
        rows=[evidence(f'Policy limit is {original}.')]
        facts,issues=checked(rows,[wire(subject='Policy',attribute='limit',value=supplied,kind='number')])
        assert not facts and issues


def test_effective_binding_cannot_use_a_later_publication_label():
    rows=[evidence('Policy effective September 1, 2026; published August 20, 2026。Policy limit is 15 minutes。')]
    facts,issues=checked(rows,[wire('E1:S2',subject='Policy',attribute='limit',value='15 minutes',kind='number',
                                  time_source_id='E1:S1',time_role='effective',time_value='August 20, 2026')])
    assert not facts and 'effective_date_not_locally_bound' in issues[0]['issues']


def test_date_attribute_cannot_bind_across_english_sentences_or_another_date():
    rows=[evidence('East store opens June 4, 2026. West store opens June 5, 2026.')]
    facts,issues=checked(rows,[wire(value='June 5, 2026')])
    assert not facts and 'attribute_value_not_locally_bound' in issues[0]['issues']
