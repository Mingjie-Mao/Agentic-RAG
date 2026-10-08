"""Paired, frozen authored contrast cases; semantic rubric calibration, not human gold."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.clients import Models
from app.config import settings
from app.semantic_review import RUBRIC_VERSION, review
from semantic_scorers import llm_judge


def write(path, result):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def metrics(rows):
    supported = [row for row in rows if row['expected']=='supported']
    unsafe = [row for row in rows if row['expected']!='supported']
    return {'cases':len(rows),'exact_labels_correct':sum(row['label']['entailment']==row['expected'] for row in rows),
            'false_accepts':sum(row['label']['entailment']=='supported' for row in unsafe),
            'unsafe_cases':len(unsafe),
            'false_rejects':sum(row['label']['entailment']!='supported' for row in supported),
            'supported_cases':len(supported),
            'unclear':sum(row['label']['entailment']=='unclear' for row in rows),
            'p50_ms':round(statistics.median(row['wall_ms'] for row in rows),1) if rows else None,
            'prompt_tokens':sum(row['usage']['prompt_tokens'] for row in rows),
            'completion_tokens':sum(row['usage']['completion_tokens'] for row in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    fixture=Path('fixtures/source_contract/semantic-calibration-v4.json')
    output=Path('artifacts/source-semantic-calibration-v4.json')
    raw=fixture.read_bytes()
    metadata={'fixture_sha256':hashlib.sha256(raw).hexdigest(),'model':settings().chat_model,
              'rubric':RUBRIC_VERSION,'human_reviewed':0,'api_cost':0,
              'scope':'Authored, assistant-labelled contrast cases. No external or independent human accuracy.',
              'code_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
                             for name in ['app/semantic_review.py','scripts/semantic_scorers.py']}}
    result={'metadata':metadata,'records':[]}
    if output.exists():
        if not args.resume:
            raise SystemExit('Use --resume; never overwrite frozen outputs')
        result=json.loads(output.read_text())
        if result['metadata']!=metadata:
            raise SystemExit('Inputs changed; use a new experiment')
    done={(row['id'],row['arm']) for row in result['records']}
    old=llm_judge()
    models=Models()
    for index,item in enumerate(json.loads(raw)['items']):
        for arm in (['legacy','source_v4'] if index%2==0 else ['source_v4','legacy']):
            if (item['id'],arm) in done:
                continue
            started=time.monotonic()
            if arm=='legacy':
                label,usage=old(item['question'],item['claim'],item['evidence'])
            else:
                label,usage=review(models,item['question'],item['claim'],item['evidence'])
            result['records'].append({'id':item['id'],'category':item['category'],'arm':arm,
                                      'expected':item['expected'],'label':label,'usage':usage,
                                      'wall_ms':round((time.monotonic()-started)*1000,1)})
            write(output,result)
            print(f"{item['id']} {arm} {label['entailment']} expected={item['expected']}",flush=True)
    result['summary']={arm:metrics([row for row in result['records'] if row['arm']==arm])
                       for arm in ['legacy','source_v4']}
    result['by_category']={category:{arm:metrics([row for row in result['records']
                                                if row['arm']==arm and row['category']==category])
                                      for arm in ['legacy','source_v4']}
                           for category in sorted({row['category'] for row in result['records']})}
    result['decision']='shadow_only; assistant authored calibration does not qualify a production semantic gate'
    write(output,result)
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
