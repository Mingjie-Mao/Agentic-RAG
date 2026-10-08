"""Local-only three-arm frozen experiment. Quality gates never flip defaults."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import select
from app.config import settings
from app.db import SessionLocal
from app.models import AgentEvent, AgentTask, User
from app.security import require_chunk
from build_multihop_subset import FILES, fetch, digest
from run_multihop_eval import documents_for_chunks, run_workflow, score
from run_source_facts_external import write, summarize, gate
from attribute_multihop_passages import classify, document_id

MANIFEST = Path('fixtures/source_contract/external-p0-p2-v1.json')
OUT = Path('artifacts/p0-p2-external-paired-v1.json')
CODE = ['app/clients.py', 'app/config.py', 'app/source_facts.py', 'app/source_fact_guards.py',
        'app/answer_contract.py', 'app/date_comparison.py', 'app/passages.py', 'app/temporal.py',
        'app/routing.py', 'app/qa.py', 'app/task_analysis.py', 'app/retrieval.py',
        'app/retrieval_queries.py', 'app/query_planner.py', 'app/verdict.py',
        'agent/controller.py', 'agent/planner.py', 'agent/tools.py', 'agent/workflow_state.py',
        'scripts/run_p0_p2_external.py', 'scripts/run_multihop_eval.py']
FLAGS = ['retrieval_mode', 'passage_window_enabled', 'source_focus_queries', 'source_facts_enabled', 'source_facts_protocol',
         'source_facts_fallback_enabled', 'focused_generation_enabled', 'adaptive_routing_enabled',
         'answer_contract_enabled', 'source_facet_queries', 'source_facet_document_queries',
         'article_first_lanes', 'source_query_plan', 'claim_consistency_enabled']


def arm_settings(cfg, arm):
    if arm in {'hybrid', 'bm25'}:
        cfg.retrieval_mode = arm
    cfg.source_focus_queries = arm == 'retrieval_focus'
    cfg.passage_window_enabled = arm in {'candidate', 'partial_facts'}
    cfg.focused_generation_enabled = arm in {'candidate', 'partial_facts'}
    cfg.adaptive_routing_enabled = arm in {'candidate', 'partial_facts'}
    cfg.source_facts_enabled = arm == 'partial_facts'
    cfg.source_facts_protocol = 'partial' if arm == 'partial_facts' else 'strict'
    cfg.source_facts_fallback_enabled = arm == 'partial_facts'
    cfg.answer_contract_enabled = False
    cfg.source_facet_queries = False
    cfg.source_facet_document_queries = False
    cfg.article_first_lanes = False
    cfg.source_query_plan = False
    cfg.claim_consistency_enabled = False


def report(result, manifest):
    arms = manifest.get('arms', ['hybrid'])
    baseline_arm = arms[0]
    result['summary'] = {arm: summarize([row for row in result['records'] if row['arm'] == arm]) for arm in arms}
    result['by_type'] = {kind: {arm: summarize([row for row in result['records']
        if row['arm'] == arm and row['question_type'] == kind]) for arm in arms}
        for kind in sorted({row['question_type'] for row in result['records']})}
    result['by_stage'] = {stage: {arm: summarize([row for row in result['records']
        if row['arm'] == arm and row.get('target_stage') == stage]) for arm in arms}
        for stage in sorted({row['target_stage'] for row in result['records'] if row.get('target_stage')})}
    result['gates'] = {}
    for arm in arms[1:]:
        rows = [dict(row, arm='source_facts' if row['arm'] == arm else 'legacy')
                for row in result['records'] if row['arm'] in {baseline_arm, arm}]
        result['gates'][arm] = gate({'legacy': result['summary'][baseline_arm],
                                     'source_facts': result['summary'][arm]}, rows, manifest['gate'])
    result['passages'] = {}
    for arm in arms:
        rows = [row for row in result['records'] if row['arm'] == arm and row['question_type'] != 'null_query']
        result['passages'][arm] = {
            'answerable_tasks': len(rows),
            'all_gold_documents_in_first_generation': sum(row['first_context_gold_documents'] for row in rows),
            'gold_facts': sum(row['gold_facts'] for row in rows),
            'gold_facts_delivered_proxy': sum(row['gold_facts_delivered_proxy'] for row in rows),
            'note': 'First actual generation context, word-4gram >=50% proxy; not semantic gold.'}


def main():
    global MANIFEST, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    MANIFEST, OUT = args.manifest, args.output
    manifest = json.loads(MANIFEST.read_text())
    if args.report_only:
        print(json.dumps(json.loads(OUT.read_text()).get('summary', {}), indent=2))
        return
    cfg = settings()
    metadata = {'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                'code_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in CODE},
                'model': cfg.chat_model, 'api_cost': 0, 'human_reviewed': 0,
                'settings': {key: getattr(cfg, key) for key in ['retrieval_mode', 'context_token_budget',
                    'passage_window_extra', 'passage_scan_limit', 'source_routing',
                    'source_focus_queries', 'passage_rerank', 'verdict_protocol']},
                'scope': 'Feature-off baseline vs candidate arms in manifest. Identical frozen queries and model. Rotated arm order. Literal v3, no human semantic accuracy claim.'}
    result = {'metadata': metadata, 'records': []}
    if OUT.exists():
        if not args.resume:
            raise SystemExit('Use --resume; refusing overwrite')
        result = json.loads(OUT.read_text())
        if metadata != result['metadata']:
            raise SystemExit('Code/config changed; freeze a new unused experiment')
    data = {digest(row['query']): row for row in fetch('MultiHopRAG.json', FILES['MultiHopRAG.json'], False)}
    done = {(row['id'], row['arm']) for row in result['records']}
    original = {key: getattr(cfg, key) for key in FLAGS}
    try:
        for index, item in enumerate(manifest['items']):
            question = data[item['query_sha256']]
            if digest(question['answer']) != item['answer_sha256']:
                raise ValueError('Gold fingerprint changed')
            arms = manifest.get('arms', ['hybrid'])
            offset = index % len(arms)
            order = arms[offset:] + arms[:offset]
            for arm in order:
                if (item['id'], arm) in done:
                    continue
                arm_settings(cfg, arm)
                with SessionLocal() as db:
                    user = db.get(User, 'mh-eval')
                    started = time.monotonic()
                    error = None
                    try:
                        payload, fired, refs, usage, wall = run_workflow(db, user, question['query'])
                        docs = documents_for_chunks(db, refs)
                    except Exception as exc:
                        error = type(exc).__name__
                        payload, fired, docs, usage = {'status': 'execution_failed', 'claims': []}, [], [], {}
                        wall = (time.monotonic() - started) * 1000
                    task = db.scalar(select(AgentTask).where(AgentTask.user_id == user.id,
                        AgentTask.goal == question['query']).order_by(AgentTask.created_at.desc()))
                    record = score(item, question, payload, docs, wall, usage, fired, rule='v3')
                    context = {}
                    event = db.scalar(select(AgentEvent).where(AgentEvent.task_id == task.id,
                        AgentEvent.event_type == 'generation_context').order_by(AgentEvent.sequence)) if task else None
                    for ref in event.evidence_chunk_ids if event else []:
                        chunk, version, document = require_chunk(db, user, ref, active_only=True)
                        context.setdefault(document.id, []).append(chunk.text)
                    gold = {document_id(row['title']) for row in question.get('evidence_list', [])}
                    missing = not gold <= set(context)
                    stage, delivered = classify(question.get('evidence_list', []), context, missing)
                    record.update(arm=arm, target_stage=item.get('target_stage'),
                        execution_error=error, task_id=task.id if task else None,
                        total_prompt_tokens=(usage.get('prompt_tokens') or 0) + (usage.get('verdict_prompt_tokens') or 0),
                        total_completion_tokens=(usage.get('completion_tokens') or 0) + (usage.get('verdict_completion_tokens') or 0),
                        first_context_gold_documents=not missing, gold_facts=len(delivered),
                        gold_facts_delivered_proxy=sum(delivered), failure_stage_proxy=stage,
                        facts_accepted=usage.get('facts_accepted'), facts_rejected=usage.get('facts_rejected'),
                        extraction_fallback=usage.get('extraction_fallback', False),
                        timings={key: value for key, value in usage.items() if key.endswith('_ms')},
                        passage_selection=usage.get('passage_selection'), scope_repairs=usage.get('scope_repairs'),
                        claims_sha256=digest(json.dumps(payload.get('claims', []), ensure_ascii=False, sort_keys=True)))
                    result['records'].append(record)
                write(OUT, result)
                print(f"{index+1}/{len(manifest['items'])} {arm} {record['status']} correct={record['answer_correct']} {wall/1000:.1f}s", flush=True)
        current = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in CODE}
        if current != metadata['code_sha256']:
            raise ValueError('Code changed during experiment')
        report(result, manifest)
        write(OUT, result)
        print(json.dumps({'summary': result['summary'], 'gates': result['gates'], 'passages': result['passages']}, indent=2))
    finally:
        for key, value in original.items():
            setattr(cfg, key, value)


if __name__ == '__main__':
    main()
