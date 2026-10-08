"""Separate mechanically observable failure layers; never invent semantic gold."""
import json
from pathlib import Path
from collections import Counter


def failure_layer(row):
    if row.get('execution_error'):
        return 'execution_error'
    if row.get('answer_correct'):
        return 'literal_pass_semantics_unreviewed'
    if row['question_type']!='null_query' and not row.get('all_gold_retrieved'):
        return 'gold_document_missing'
    if row['arm']=='source_facts' and row.get('fact_rejections'):
        return 'source_field_or_schema_rejected'
    if row['arm']=='source_facts' and row.get('facts',0)==0:
        return 'no_relevant_structured_facts'
    if row.get('verdict')=='unclear':
        return 'relation_unresolved'
    if row.get('verdict') in {'yes','no'}:
        return 'conclusion_or_source_interpretation_needs_review'
    return 'answer_or_literal_scorer_needs_review'


def main():
    path=Path('artifacts/source-facts-external-paired-v1.json')
    result=json.loads(path.read_text())
    if 'summary' not in result:
        raise SystemExit('Wait for frozen same-denominator experiment to complete')
    rows=[{'id':row['id'],'arm':row['arm'],'task_id':row['task_id'],
           'classification':failure_layer(row),'answer_correct':row['answer_correct'],
           'independent_semantic_label':None} for row in result['records']]
    output={'scope':'Mechanical failure-layer attribution only; document coverage and literal scores are not semantic proof.',
            'human_reviewed':0,'counts':{arm:dict(Counter(row['classification'] for row in rows if row['arm']==arm))
                                      for arm in ['legacy','source_facts']},
            'rows':rows,'training_use':'none; source/format/retrieval errors are not policy failures'}
    Path('artifacts/source-facts-failure-attribution-v1.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output['counts'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
