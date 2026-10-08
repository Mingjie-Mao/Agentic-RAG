"""Fresh paired fixed-workflow evaluation; compare protocols, never train on outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sqlalchemy import select
from app.config import settings
from app.db import SessionLocal
from app.models import AgentTask, User
from build_multihop_subset import FILES, fetch, digest
from run_multihop_eval import documents_for_chunks, run_workflow, score

MANIFEST=Path('fixtures/source_contract/external-paired-v1.json')
OUT=Path('artifacts/source-facts-external-paired-v1.json')
CODE=['app/source_facts.py','app/source_fact_guards.py','app/clients.py','app/qa.py','app/config.py','app/answer_contract.py',
      'app/verdict.py','agent/controller.py','agent/planner.py','agent/workflow_state.py',
      'app/task_analysis.py','scripts/run_source_facts_external.py','scripts/run_multihop_eval.py']


def write(path, result):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def percentile(values,q, decimals=1):
    ordered=sorted(values)
    if not ordered:
        return None
    index=(len(ordered)-1)*q
    lo=int(index)
    hi=min(lo+1,len(ordered)-1)
    return round(ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo),decimals)


def summarize(rows):
    answerable=[row for row in rows if row['question_type']!='null_query']
    null=[row for row in rows if row['question_type']=='null_query']
    return {'questions':len(rows),'correct':sum(row['answer_correct'] for row in rows),
            'false_refusals':sum(row['status']=='insufficient_evidence' for row in answerable),
            'verification_refusals':sum(row['status']=='verification_failed' for row in answerable),
            'false_answers_on_null':sum(row['status'] in {'answered','conflict'} for row in null),
            'null_correct_refusals':sum(row['status']=='insufficient_evidence' for row in null),
            'unclear_verdicts':sum(row.get('verdict')=='unclear' for row in rows),
            'execution_errors':sum(bool(row.get('execution_error')) for row in rows),
            'p50_ms':percentile([row['latency_ms'] for row in rows],.5),
            'p95_ms':percentile([row['latency_ms'] for row in rows],.95),
            'total_prompt_tokens':sum(row['total_prompt_tokens'] for row in rows),
            'total_completion_tokens':sum(row['total_completion_tokens'] for row in rows),
            'total_tokens':sum(row['total_prompt_tokens']+row['total_completion_tokens'] for row in rows)}


def gate(summary,records,criteria):
    old,new=summary['legacy'],summary['source_facts']
    before={row['id']:row['answer_correct'] for row in records if row['arm']=='legacy'}
    after={row['id']:row['answer_correct'] for row in records if row['arm']=='source_facts'}
    if set(before)!=set(after):
        return {'decision':'incomplete','checks':{}}
    differences=[int(after[key])-int(before[key]) for key in sorted(before)]
    rng=random.Random(42)
    samples=[sum(rng.choices(differences,k=len(differences)))/len(differences) for _ in range(5000)]
    ci=[percentile(samples,.025,4),percentile(samples,.975,4)]
    checks={
        'correct_gain':new['correct']-old['correct']>=criteria['minimum_answer_gain'],
        'no_additional_false_refusals':new['false_refusals']+new['verification_refusals']<=old['false_refusals']+old['verification_refusals'],
        'no_additional_false_answers_on_null':new['false_answers_on_null']<=old['false_answers_on_null'],
        'p50_budget':new['p50_ms']<=old['p50_ms']*criteria['maximum_latency_ratio'],
        'p95_budget':new['p95_ms']<=old['p95_ms']*criteria['maximum_latency_ratio'],
        'token_budget':new['total_tokens']<=old['total_tokens']*criteria['maximum_total_token_ratio'],
        'no_execution_errors':old['execution_errors']==new['execution_errors']==0,
    }
    return {'decision':'eligible_for_larger_validation' if all(checks.values()) else 'keep_default_off',
            'checks':checks,'paired_accuracy_delta':round(sum(differences)/len(differences),4),
            'paired_bootstrap_ci95':ci,'improved':differences.count(1),'regressed':differences.count(-1),
            'default_enabled':False,
            'note':'Small screened sample, literal v3 accuracy, independent human semantic review=0. A pass permits larger validation; it never flips the production default.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--report-only',action='store_true',help='Read frozen results without rerunning, rescoring or hash relaxation')
    args=parser.parse_args()
    if args.report_only:
        stored=json.loads(OUT.read_text())
        if 'summary' not in stored:
            raise SystemExit('Experiment incomplete')
        print(json.dumps({'summary':stored['summary'],'gate':stored['gate']},ensure_ascii=False,indent=2))
        return
    raw=MANIFEST.read_bytes()
    manifest=json.loads(raw)
    data={digest(row['query']):row for row in fetch('MultiHopRAG.json',FILES['MultiHopRAG.json'],False)}
    cfg=settings()
    metadata={'manifest_sha256':hashlib.sha256(raw).hexdigest(),'model':cfg.chat_model,
              'code_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in CODE},
              'settings':{key:getattr(cfg,key) for key in ['retrieval_mode','source_routing','passage_rerank',
                           'passage_expand_documents','context_token_budget','rerank_mode','verdict_protocol']},
              'scoring':'literal_v3; semantic support reported separately','human_reviewed':0,'api_cost':0,
              'scope':'Same fixed-workflow budget and authorized corpus; only generation/conclusion protocol differs. No reference facts given to system.'}
    result={'metadata':metadata,'records':[]}
    if OUT.exists():
        if not args.resume:
            raise SystemExit('Use --resume; frozen results cannot be overwritten')
        result=json.loads(OUT.read_text())
        if result['metadata']!=metadata:
            raise SystemExit('Code/configuration/input changed; use new unused questions')
    done={(row['id'],row['arm']) for row in result['records']}
    original=(cfg.source_facts_enabled,cfg.answer_contract_enabled)
    cfg.answer_contract_enabled=False
    try:
        for index,item in enumerate(manifest['items']):
            original_question=data[item['query_sha256']]
            if digest(original_question['answer'])!=item['answer_sha256']:
                raise ValueError('Pinned gold changed')
            for arm in (['legacy','source_facts'] if index%2==0 else ['source_facts','legacy']):
                if (item['id'],arm) in done:
                    continue
                cfg.source_facts_enabled=arm=='source_facts'
                started=time.monotonic()
                with SessionLocal() as db:
                    user=db.get(User,'mh-eval')
                    if not user:
                        raise ValueError('External corpus not ingested')
                    error=None
                    try:
                        payload,fired,chunk_ids,usage,wall=run_workflow(db,user,original_question['query'])
                        docs=documents_for_chunks(db,chunk_ids)
                    except Exception as exc:
                        error=type(exc).__name__
                        payload={'status':'execution_failed','claims':[],'citations':[]}
                        fired,docs,usage,wall=[],[],{},(time.monotonic()-started)*1000
                    task=db.scalar(select(AgentTask).where(AgentTask.user_id==user.id,
                                  AgentTask.goal==original_question['query']).order_by(AgentTask.created_at.desc()))
                    record=score(item,original_question,payload,docs,wall,usage,fired,rule='v3')
                    record.update(arm=arm,execution_error=error,task_id=task.id if task else None,
                                  total_prompt_tokens=(usage.get('prompt_tokens') or 0)+(usage.get('verdict_prompt_tokens') or 0),
                                  total_completion_tokens=(usage.get('completion_tokens') or 0)+(usage.get('verdict_completion_tokens') or 0),
                                  facts=len(payload.get('facts',[])),fact_rejections=usage.get('facts_rejected',[]),
                                  relation_reason=usage.get('relation_reason'),
                                  claims_sha256=digest(json.dumps(payload.get('claims',[]),ensure_ascii=False,sort_keys=True)))
                    result['records'].append(record)
                write(OUT,result)
                print(f"{item['id']} {arm} {record['status']} correct={record['answer_correct']} {wall/1000:.1f}s",flush=True)
        result['summary']={arm:summarize([row for row in result['records'] if row['arm']==arm])
                           for arm in ['legacy','source_facts']}
        result['by_type']={kind:{arm:summarize([row for row in result['records'] if row['arm']==arm and row['question_type']==kind])
                               for arm in ['legacy','source_facts']}
                           for kind in sorted({row['question_type'] for row in result['records']})}
        result['gate']=gate(result['summary'],result['records'],manifest['gate'])
        write(OUT,result)
        print(json.dumps({'summary':result['summary'],'gate':result['gate']},ensure_ascii=False,indent=2))
    finally:
        cfg.source_facts_enabled,cfg.answer_contract_enabled=original


if __name__=='__main__':
    main()
