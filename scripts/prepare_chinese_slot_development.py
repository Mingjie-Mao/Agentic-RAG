"""Author isolated synthetic Chinese dev sources and export actual runtime inputs.

No human labels or correctness assertions are generated. The entire new source is
designated development before model execution; old test families stay unchanged.
"""
import argparse
from collections import Counter
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_runtime import atomic_json, checkpoint_lock  # noqa: E402
from research_review import digest, group_splits, packet  # noqa: E402

TENANT = "slot-calibration-zh-dev"
USER = "slot-calibration-zh-reader"
FIXTURES = ROOT / "fixtures/semantic_slot_zh_development"
DISCLAIMER = "本资料由助手编写，仅用于合成开发数据，不代表任何真实组织制度；所有标签待独立人工审核。"
SOURCES = [
    ("certificate", "晨汐设备证书轮换规程",
     "晨汐设备证书的轮换宽限期为12小时。宽限期内新旧证书均可完成握手。"
     "宽限期结束后，旧证书只能查询轮换进度，不能上传遥测。运维人员应在轮换结束前2小时通知设备负责人。",
     ["晨汐设备证书有多长的轮换宽限期？", "晨汐旧证书在宽限期结束后可以做什么，不能做什么？",
      "晨汐运维人员应在证书轮换结束前多久通知谁？"]),
    ("export", "河湾离线导出任务运行说明",
     "河湾离线导出失败后最多自动重试4次，每次间隔45秒。排队任务保留36小时。"
     "结果文件保留7天，保留期限从导出完成时计算，不能从提交任务时计算。超过保留期的文件不提供恢复。",
     ["河湾导出失败后最多自动重试几次，每次间隔多久？", "河湾排队任务与结果文件分别保留多久？",
      "河湾结果文件的保留期限从何时开始计算，超过期限能恢复吗？"]),
    ("delegate", "杉谷临时代理授权规定",
     "杉谷临时代理授权有效期为4工作日，从审批通过的次一工作日起算。"
     "代理人可以查看工单处理进度，但不能变更工单负责人，也不能再次转授权限。"
     "申请人在授权期内可以随时撤回代理，撤回后30秒内失效。",
     ["杉谷临时代理授权有效期多长，从什么时候起算？", "杉谷代理人能查看哪些信息，不能进行哪些操作？",
      "杉谷临时代理能否提前撤回，撤回后多久失效？"]),
    ("support", "松原紧急工单应答约定",
     "松原P2级工单的首次响应时限为6小时，P1级为40分钟。首次响应不等于故障恢复。"
     "若P2级工单等待超过6小时仍未响应，值班负责人必须电话通知协调员。"
     "等待恰好6小时不会触发上述电话升级。用户留言不能替代值班负责人的电话通知。",
     ["松原P1与P2级工单的首次响应时限分别是多少？", "松原P2级工单在哪种情况下由谁通知谁？",
      "松原P2级工单等待恰好6小时是否必须电话升级，用户留言能否代替电话？"]),
    ("archive", "潮生档案上传容量说明",
     "潮生档案单批上传最多800份文件，批次总容量不能超过2GB。单份文件最大为60MB。"
     "加密压缩包不被接受，即使容量合规也不能上传。未完成上传的临时分片保留18小时后清除。",
     ["潮生档案单批上传最多多少份文件，总容量上限是多少？", "潮生档案单份文件最大多大，加密压缩包能否上传？",
      "潮生档案未完成上传的临时分片如何处理？"]),
    ("payment", "澄江供应商付款复核流程",
     "澄江单笔供应商付款金额超过13000元时，需要财务主管与采购负责人共同复核。"
     "金额恰好13000元只需要采购负责人复核。财务主管不在岗时不能用系统管理员代替。"
     "复核记录保留5年，从付款完成日期起算，不从发票开具日期起算。",
     ["澄江供应商付款金额超过13000元时需要谁复核？", "澄江付款恰好13000元时谁复核，管理员能替代财务主管吗？",
      "澄江付款复核记录保留多久，从什么日期起算？"]),
]


def recipe():
    docs = [{"id": "slotdev-zh-"+key, "title": title,
             "text": f"# {title}\n\n{DISCLAIMER}\n\n{text}\n", "questions": questions}
            for key, title, text, questions in SOURCES]
    return {"version": "isolated-chinese-slot-development-v1", "origin": "assistant_authored_synthetic",
            "partition": "development_only", "independent_human_labels": 0, "documents": docs}


def ingest(spec):
    from sqlalchemy import select
    from app.db import SessionLocal
    from app.ingestion import queue_document, process_job
    from app.models import Tenant, User, Document, Job, uid, now
    from app.security import hasher
    with SessionLocal() as db:
        if not db.get(Tenant, TENANT):
            db.add(Tenant(id=TENANT, name="中文槽位校准合成开发资料"))
            db.flush()
        if not db.get(User, USER):
            db.add(User(id=USER, tenant_id=TENANT, username="reviewer@slot-calibration-zh.invalid",
                        display_name="开发数据自动化账号", role="member", groups=["engineering"],
                        active=True, password_hash=hasher.hash(secrets.token_urlsafe(48))))
            db.commit()
        user = db.get(User, USER)
        for row in spec["documents"]:
            data = row["text"].encode()
            doc = db.get(Document, row["id"])
            if doc and doc.tenant_id != TENANT:
                raise ValueError("development document ID collision")
            doc, version = queue_document(db, user, row["id"]+".md", data, row["title"],
                ["engineering"], False, metadata={"origin": spec["origin"], "partition": "development_only"},
                document_id=row["id"])
            if version.content_hash != hashlib.sha256(data).hexdigest():
                raise ValueError("development source changed after import")
            if version.status != "ready":
                # Claim only this import's job. Do not drain other users' uploads.
                job = db.scalar(select(Job).where(Job.version_id == version.id).with_for_update())
                if not job or job.state != "queued":
                    raise ValueError("development import already processing or failed")
                job.state, job.lease_token, job.lease_until = "processing", uid(), now()+timedelta(minutes=10)
                job.attempts += 1
                version.status = "processing"
                job_id, token = job.id, job.lease_token
                db.commit()
                process_job(job_id, token)
                db.expire_all()
            if db.get(Document, doc.id).active_version_id != version.id:
                raise ValueError("development document was not published")
            print(f"published {row['id']}", flush=True)


def export(spec, out, *, resume=False):
    from sqlalchemy import select
    from agent.planner import subgoals
    from app.config import settings
    from app.db import SessionLocal
    from app.models import Answer, User
    from app.qa import answer_question
    from app.security import require_chunk
    from paired_semantic_replay import code_inputs
    from run_unseen_benchmark import configuration

    cfg = settings()
    for flag in ("semantic_slot_shadow_enabled", "semantic_coverage_shadow", "semantic_coverage_control",
                 "semantic_slot_control_enabled", "semantic_shadow_enabled", "task_contract_enabled"):
        setattr(cfg, flag, False)
    identity = {"recipe_sha256": digest(spec), "code_inputs": code_inputs(), "configuration": configuration()}
    state_path = out / "runs.json"
    if state_path.exists():
        if not resume:
            raise ValueError("development runs exist; use --resume for the same frozen inputs")
        state = json.loads(state_path.read_text())
        if state["identity"] != identity:
            raise ValueError("development code/models/source changed; cannot resume")
    else:
        if resume:
            raise ValueError("no development checkpoint to resume")
        state = {"identity": identity, "run_ids": {}}
        atomic_json(state_path, state)
    rows = []
    with SessionLocal() as db:
        user = db.get(User, USER)
        for doc in spec["documents"]:
            for question in doc["questions"]:
                if code_inputs() != identity["code_inputs"]:
                    raise ValueError("code changed during development export")
                qid = "zh-dev-"+digest(question)[:20]
                token = "zh-dev-"+digest([identity, qid])
                record = db.get(Answer, state["run_ids"][qid]) if qid in state["run_ids"] else None
                if record is None:
                    # Recover a committed answer if interruption preceded checkpoint write.
                    record = db.scalar(select(Answer).where(Answer.tenant_id == TENANT,
                        Answer.payload["trace"]["benchmark_run_token"].as_string() == token))
                if record is None:
                    result = answer_question(db, user, question, benchmark_run_token=token)
                    record = db.get(Answer, result["id"])
                state["run_ids"][qid] = record.id
                atomic_json(state_path, state)
                payload = record.payload
                def passage(cid):
                    c, v, d = require_chunk(db, user, cid, active_only=True)
                    if d.tenant_id != TENANT:
                        raise ValueError("development evidence outside isolated tenant")
                    return dict(chunk_id=c.id, version_id=v.id, document_id=d.id, text=c.text, title=d.title)
                context = [passage(cid) for cid in payload["trace"]["generation_context_chunk_ids"]]
                if not context:
                    raise ValueError("development generation had no evidence")
                base = {"corpus": "s3", "qid": qid, "question": question, "generated_run_id": record.id,
                        "origin": spec["origin"], "partition": "development_only"}
                def add(kind, slot, evidence, variant):
                    row = {**base, "kind": kind, "slot": slot, "evidence": evidence, "variant": variant,
                           "families": sorted({p["document_id"] for p in evidence} | {doc["id"]})}
                    rows.append({**row, "id": "zh-dev-"+digest(row)[:24]})
                for n, text in enumerate(subgoals(question), 1):
                    slot = {"id": f"s{n}", "text": text, "required": True, "origin": "agent.planner.subgoals"}
                    add("coverage", slot, context, "actual_generation_context")
                    for p in context:
                        add("relevance", slot, [p], "actual_candidate")
                    for removed in range(len(context)):
                        add("coverage", slot, context[:removed]+context[removed+1:], "leave_one_out")
                for n, claim in enumerate(payload.get("claims", []), 1):
                    refs = list(dict.fromkeys(claim["evidence_ids"]))
                    add("claim", {"id": f"claim{n}", "text": claim["text"], "purpose": "claim", "allowed_refs": refs},
                        [passage(cid) for cid in refs], "actual_generated_claim")
                print(f"exported {qid}: {len(context)} actual passages, {len(payload.get('claims', []))} claims", flush=True)
    group_splits(rows)
    for row in rows:
        row["split"] = "dev"  # Whole source predesignated dev; no test labels used.
    atomic_json(out / "component-inputs.json", packet(rows, "isolated synthetic Chinese actual-runtime development inputs"))
    return {"version": spec["version"], "origin": spec["origin"], "independent_human_labels": 0,
            "questions": len(state["run_ids"]), "counts": dict(Counter(r["kind"] for r in rows)),
            "partition": "development_only", "source_sha256": digest(rows)}


def merge_development(source, development):
    for value in (source, development):
        if value["source_sha256"] != digest(value["items"]) or value.get("reviews"):
            raise ValueError("merge only immutable unreviewed component inputs")
    old_docs = {doc for row in source["items"] for doc in row["families"]}
    old_questions = {row["question"] for row in source["items"]}
    if any(set(row["families"]) & old_docs or row["question"] in old_questions for row in development["items"]):
        raise ValueError("new development source overlaps existing test/development inputs")
    if any(r["split"] != "dev" or r.get("partition") != "development_only" for r in development["items"]):
        raise ValueError("new source must be wholly predesignated development")
    return packet(source["items"]+development["items"], "actual-input component review with isolated synthetic Chinese dev")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--merge-source", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if (args.output_dir / "expanded.json").exists():
        raise ValueError("review packet already finalized; do not overwrite")
    FIXTURES.mkdir(parents=True, exist_ok=True)
    spec = recipe()
    path = FIXTURES / "source.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("development recipe changed")
    if not path.exists():
        atomic_json(path, spec)
    with checkpoint_lock(args.output_dir / "runs.json"):
        ingest(spec)
        summary = export(spec, args.output_dir, resume=args.resume)
        merged = merge_development(json.loads(args.merge_source.read_text()),
                                   json.loads((args.output_dir / "component-inputs.json").read_text()))
        atomic_json(args.output_dir / "expanded.json", merged)
        summary.update(expanded_counts=dict(Counter(r["kind"] for r in merged["items"])),
                       strata=dict(Counter(f'{r["kind"]}:{r["corpus"]}:{r["split"]}' for r in merged["items"])))
        atomic_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
