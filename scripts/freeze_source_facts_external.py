"""Freeze 12 fresh external tasks BEFORE running either protocol.

Exclude all B1-B3 manifests, all actually asked external queries in the database,
and near-duplicates. No source body or reference answer is passed to the generator.
"""
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sqlalchemy import select
from app.db import SessionLocal
from app.models import AgentTask, Answer
from build_multihop_subset import FILES, fetch, digest

MANIFEST=Path('fixtures/source_contract/external-paired-v1.json')


def normalized_words(query):
    return set(re.findall(r'[a-z0-9]+',query.casefold()))


def near(left,right):
    a,b=normalized_words(left),normalized_words(right)
    return max(SequenceMatcher(None,left.casefold(),right.casefold()).ratio(),
               len(a&b)/len(a|b) if a|b else 0)


def freeze(data, used, pool):
    quota=[('comparison_query','yes',2),('comparison_query','no',2),
           ('temporal_query','yes',2),('temporal_query','no',2),
           ('inference_query','other',2),('null_query','other',2)]
    chosen,screened=[],[]
    previous=[data[key] for key in sorted(used) if key in data]
    signatures={tuple(sorted(e['title'] for e in row.get('evidence_list',[]))) for row in previous}
    for kind,gold,count in quota:
        found=0
        for key in sorted(pool):
            row=data[key]
            label={'true':'yes','false':'no'}.get(row['answer'].strip().casefold(),row['answer'].strip().casefold())
            if row['question_type']!=kind or (label if label in {'yes','no'} else 'other')!=gold:
                continue
            if key in used or any(old['query_sha256']==key for old in chosen):
                continue
            signature=tuple(sorted(e['title'] for e in row.get('evidence_list',[])))
            if signature and signature in signatures:
                screened.append({'query_sha256':key,'reason':'previous_gold_document_combination'})
                continue
            distances=[(near(row['query'],old['query']),digest(old['query'])) for old in previous]
            distances.extend((near(row['query'],data[item['query_sha256']]['query']),item['query_sha256']) for item in chosen)
            similarity,nearest=max(distances,default=(0,None))
            if similarity>=0.80:
                screened.append({'query_sha256':key,'reason':'near_duplicate','similarity':round(similarity,4)})
                continue
            chosen.append({'id':f"SF-{key[:12]}",'question_type':kind,'query_sha256':key,
                           'answer_sha256':digest(row['answer']),'nearest_previous_hash':nearest,
                           'max_similarity':round(similarity,4),'gold_document_count':len(set(signature))})
            found+=1
            if found==count:
                break
        if found!=count:
            raise ValueError(f'Insufficient unused, screened candidates for {kind}/{gold}')
    return chosen,screened


def main():
    if MANIFEST.exists():
        raise SystemExit('Selection is already frozen; refusing to overwrite')
    data={digest(row['query']):row for row in fetch('MultiHopRAG.json',FILES['MultiHopRAG.json'],False)}
    corpus=fetch('corpus.json',FILES['corpus.json'],False)
    titles={row['title'] for row in corpus}
    used=set()
    for filename in ['subset.json','subset-r2.json','subset-r3.json']:
        used.update(row['query_sha256'] for row in json.loads((Path('fixtures/multihop')/filename).read_text())['items'])
    oracle=Path('fixtures/verdict/external-oracle-r4-8.json')
    used.update(row['query_sha256'] for row in json.loads(oracle.read_text())['items'])
    with SessionLocal() as db:
        used.update(digest(question) for question in db.scalars(select(Answer.question).where(Answer.user_id=='mh-eval')))
        used.update(digest(goal) for goal in db.scalars(select(AgentTask.goal).where(AgentTask.user_id=='mh-eval')))
    pool={key for key,row in data.items() if key not in used
          and all(e['title'] in titles for e in row.get('evidence_list',[]))}
    chosen,screened=freeze(data,used,pool)
    payload={'name':'source-facts-external-paired-v1','source_files':FILES,
             'scope':'Fresh small end-to-end fixed-workflow comparison. Source overlap is possible; not independent human semantic gold.',
             'selection':'sha256 order per stratum; comparison/temporal 2 yes and 2 no each; inference/null 2 each; exclude B1-B3 and all actually asked external queries; reject old gold-document combinations and max(character similarity, token Jaccard)>=0.80.',
             'used_question_hashes':sorted(used),'items':chosen,'screened_out':screened,
             'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'gate':{'minimum_answer_gain':2,'maximum_additional_false_refusals':0,
                     'maximum_additional_false_answers_on_null':0,'maximum_latency_ratio':1.15,
                     'maximum_total_token_ratio':1.15,'required_execution_errors':0,
                     'independent_human_review_required_for_global_semantic_gate':True}}
    MANIFEST.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'selected':len(chosen),'used_excluded':len(used),'screened_out':len(screened),
                      'max_similarity':max(row['max_similarity'] for row in chosen)},indent=2))


if __name__=='__main__':
    main()
