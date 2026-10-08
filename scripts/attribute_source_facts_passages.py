"""Trace actual generation evidence, not just documents encountered by a tool."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.db import SessionLocal
from app.models import AgentTask, Chunk, DocumentVersion, User
from app.security import require_chunk
from attribute_multihop_passages import classify, document_id
from build_multihop_subset import FILES, fetch, digest


def main():
    run=json.loads(Path('artifacts/source-facts-external-paired-v1.json').read_text())
    if 'summary' not in run:
        raise SystemExit('Wait for complete same-denominator experiment')
    manifest=json.loads(Path('fixtures/source_contract/external-paired-v1.json').read_text())
    items={row['id']:row for row in manifest['items']}
    data={digest(row['query']):row for row in fetch('MultiHopRAG.json',FILES['MultiHopRAG.json'],False)}
    rows=[]
    with SessionLocal() as db:
        user=db.get(User,'mh-eval')
        for result in run['records']:
            question=data[items[result['id']]['query_sha256']]
            if result['question_type']=='null_query':
                continue
            task=db.get(AgentTask,result['task_id'])
            if not task or task.user_id!=user.id:
                raise ValueError('Missing or mismatched task')
            by_document={}
            for ref in task.evidence_chunk_ids:
                require_chunk(db,user,ref,active_only=True)
                chunk=db.get(Chunk,ref)
                version=db.get(DocumentVersion,chunk.version_id)
                by_document.setdefault(version.document_id,[]).append(chunk.text)
            gold={document_id(e['title']) for e in question['evidence_list']}
            missing=not gold<=set(by_document)
            stage,delivered=classify(question['evidence_list'],by_document,missing)
            rows.append({'id':result['id'],'arm':result['arm'],'task_id':task.id,
                         'answer_correct':result['answer_correct'], 'stage_proxy':stage,
                         'all_gold_documents_in_generation':not missing,
                         'gold_facts':len(delivered),'gold_facts_delivered_proxy':sum(delivered),
                         'evidence_chunks':len(task.evidence_chunk_ids)})
    output={'scope':'Final generation evidence handles; gold word-4gram >=50% is a passage proxy, not semantic entailment.',
            'human_reviewed':0,'rows':rows,'summary':{}}
    for arm in ['legacy','source_facts']:
        arm_rows=[row for row in rows if row['arm']==arm]
        failures=[row for row in arm_rows if not row['answer_correct']]
        output['summary'][arm]={
            'answerable_tasks':len(arm_rows),
            'all_gold_documents_in_generation':sum(row['all_gold_documents_in_generation'] for row in arm_rows),
            'gold_facts':sum(row['gold_facts'] for row in arm_rows),
            'gold_facts_delivered_proxy':sum(row['gold_facts_delivered_proxy'] for row in arm_rows),
            'failure_stages':{stage:sum(row['stage_proxy']==stage for row in failures)
                              for stage in sorted({row['stage_proxy'] for row in failures})}}
    Path('artifacts/source-facts-passage-attribution-v1.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
