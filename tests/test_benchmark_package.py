from copy import deepcopy
import hashlib
import json
import shutil

import pytest

from scripts.benchmark_package import PACKAGE, validate_package
from scripts.benchmark_package_runtime import method_matrix, reserve_test_run, run_package_arm, validate_test_worker
from scripts.benchmark_package_scoring import (
    CHECKS, METHODS, answer_review_packet, assess, report, task_review_packet, validate_reviews,
    provisional_score,
)


def sample():
    task = {"id": "t", "category": "multi_hop", "goal": "test", "question": "test",
            "expected_status": "answered", "required_facts": [{"id": "a"}, {"id": "b"}],
            "gold_documents": ["A", "B"], "expected_behavior": {"strict_scoring": False},
            "answerable": True, "scenario_events": []}
    row = {"task_id": "t", "run_id": "r", "status": "answered", "steps": 1, "latency_ms": 20,
           "answer_payload": {"status": "answered", "claims": [{"text": "both", "evidence_ids": ["a", "b"], "quotes": ["Fact A", "Fact B"]}],
                              "citations": [{"document_id": "A", "chunk_id": "a", "text": "Fact A"},
                                            {"document_id": "B", "chunk_id": "b", "text": "Fact B"}]},
           "retrieved_evidence": [{"document_id": "A"}, {"document_id": "B"}],
           "first_retrieval_documents": ["A", "B"], "trace_complete": True, "tool_trace": [],
           "forbidden_present": {}, "scenario_events": {}, "execution_error_kind": None}
    review = {"fact_verdicts": {"a": "supported", "b": "supported"},
              "citation_verdicts": ["supported", "supported"], "claim_verdicts": ["supported"],
              "retrieved_fact_verdicts": {"a": "supported", "b": "supported"},
              **dict.fromkeys(CHECKS, True)}
    return task, row, review


def test_package_separates_core_security_and_consumed_external():
    from scripts.benchmark_package import ROOT
    if not (ROOT / ".runtime/benchmark-package/v1/dev/tasks.json").exists():
        pytest.skip("pinned private material not materialized; public package remains available")
    result = validate_package()
    assert result["counts"] == {"dev": 47, "core": 70, "security": 16, "external": 150}
    assert result["exact_question_overlap"] == 0
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    assert manifest["splits"]["external"]["historically_used"] is True
    assert manifest["splits"]["core"]["headline_source"] is True


def test_private_material_can_be_rebuilt_without_touching_original_package(tmp_path, monkeypatch):
    from scripts import benchmark_package as package
    root = package.ROOT
    if not (root / ".runtime/multihop/MultiHopRAG.json").exists():
        pytest.skip("pinned upstream local cache absent")
    for folder in ("fixtures/unseen", "fixtures/unseen_v2", "benchmarks/enterprise_rag/v1"):
        shutil.copytree(root / folder, tmp_path / folder)
    for name in ("fixtures/source_contract/external-p0-p2-v5-bm25.json", "fixtures/multihop/subset-r2.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    monkeypatch.setattr(package, "ROOT", tmp_path)
    rebuilt = package.materialize(tmp_path / "benchmarks/enterprise_rag/v1")
    assert rebuilt["counts"]["dev"] == 47 and rebuilt["counts"]["external"] == 150
    assert rebuilt["exact_question_overlap"] == 0


def test_alternative_one_step_path_is_a_strict_pass():
    task, row, review = sample()
    task["expected_behavior"]["hints"] = {"required_tools": ["search_documents", "open_document", "compare_versions"]}
    outcome = assess(task, row, review)
    assert outcome["answer"]["strict_task_success"] is True
    assert outcome["retrieval"]["gold_document_recall"] == 1
    assert outcome["retrieval"]["recall_at_k"]["1"] == .5


def test_review_cannot_promote_a_correct_value_with_an_unsupported_quote():
    task, row, review = sample()
    row['answer_payload']['claims'][0].update(text='审计日志保留5年。', evidence_ids=['a'], quotes=['审计日志保留3年。'])
    row['answer_payload']['citations'][0]['text']='审计日志保留3年。'
    row['answer_payload']['citations'][1]['text']='审计日志保留5年。'
    scored=assess(task,row,review)['answer']
    assert scored['citation_binding_valid'] is False
    assert scored['strict_task_success'] is False


@pytest.mark.parametrize("check", CHECKS)
def test_each_strict_requirement_is_necessary(check):
    task, row, review = sample()
    review[check] = False
    assert assess(task, row, review)["answer"]["strict_task_success"] is False


def test_fact_precision_requires_atomic_claim_bound_review():
    task, row, review = sample()
    assert assess(task, row, review)["answer"]["fact_precision"] is None
    review.update(atomic_facts_complete=True, answer_facts=[
        {"text": "a", "claim_index": 0, "verdict": "supported"},
        {"text": "made up", "claim_index": 0, "verdict": "unsupported"}])
    outcome = assess(task, row, review)["answer"]
    assert outcome["fact_precision"] == .5 and outcome["strict_task_success"] is False
    review["answer_facts"][1]["claim_index"] = 9
    with pytest.raises(ValueError, match="real claim"):
        assess(task, row, review)


def test_main_methods_cannot_have_different_settings_disguised_as_same_conditions():
    matrix = [{"name": m, "mode": m, "overrides": {}} for m in METHODS]
    suite = {"arms": list(METHODS), "method_matrix": matrix}
    assert set(method_matrix(suite)) == set(METHODS)
    matrix[-1]["overrides"]["passage_rerank"] = True
    with pytest.raises(ValueError, match="share unchanged"):
        method_matrix(suite)


def test_partial_fact_is_not_full_success_and_unknown_review_is_not_zero():
    task, row, review = sample()
    review["fact_verdicts"]["b"] = "partial"
    answer = assess(task, row, review)["answer"]
    assert answer["strict_task_success"] is False and answer["fact_recall"] == .5
    row["retrieved_evidence"] = None
    row["first_retrieval_documents"] = None
    unknown = assess(task, row)
    assert unknown["answer"]["strict_task_success"] is None
    assert unknown["retrieval"]["gold_document_recall"] is None


@pytest.mark.parametrize("field", ["citation_verdicts", "claim_verdicts"])
def test_unsupported_or_unreviewed_extra_claims_and_citations_fail(field):
    task, row, review = sample()
    review[field][-1] = "unsupported"
    assert assess(task, row, review)["answer"]["strict_task_success"] is False
    review[field] = []
    with pytest.raises(ValueError, match="every actual"):
        assess(task, row, review)


def test_correct_abstain_has_na_fact_and_citation_denominators():
    task, row, review = sample()
    task.update(required_facts=[], gold_documents=[], answerable=False, expected_status="insufficient_evidence")
    row["answer_payload"] = {"status": "insufficient_evidence", "claims": [], "citations": []}
    review.update(fact_verdicts={}, citation_verdicts=[], claim_verdicts=[], retrieved_fact_verdicts={})
    scored = assess(task, row, review)
    assert scored["answer"]["strict_task_success"] is True
    assert scored["answer"]["fact_recall"] is None and scored["answer"]["citation_correctness"] is None
    row["answer_payload"]["status"] = "answered"
    assert assess(task, row, review)["answer"]["strict_task_success"] is False


def test_unexercised_revocation_cannot_pass_and_expected_failure_can():
    task, row, review = sample()
    task.update(required_facts=[], gold_documents=[], answerable=False, expected_status=["insufficient_evidence", "execution_failed"],
                state_change={"action": "revoke_document"}, scenario_events=["state_change"])
    row.update(answer_payload={"status": "execution_failed", "claims": [], "citations": []}, execution_error_kind="authorization_change")
    review.update(fact_verdicts={}, citation_verdicts=[], claim_verdicts=[], retrieved_fact_verdicts={})
    assert assess(task, row, review)["answer"]["strict_task_success"] is False
    row["scenario_events"]["state_change"] = True
    assert assess(task, row, review)["answer"]["strict_task_success"] is True
    row["execution_error_kind"] = "unexpected"
    assert assess(task, row, review)["answer"]["strict_task_success"] is False


def test_model_reviews_have_honest_provenance_and_are_hash_bound(tmp_path):
    task, _, _ = sample()
    packet = task_review_packet({"tasks": [task]}, {"documents": []}, tmp_path)
    reviewed = deepcopy(packet)
    reviewed["reviews"] = [{"id": "t", "reviewer_type": "model", "reviewer": "GPT", "model": "actual-model",
                            "independent": False, "reason": "checked sources", "label": "approved"}]
    assert validate_reviews(packet, reviewed)["t"]["independent"] is False
    reviewed["items"][0]["question"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        validate_reviews(packet, reviewed)


def test_review_packets_hide_methods_and_report_requires_all_tasks():
    task, row, review = sample()
    suite = {"version": "v1", "split": "core", "tasks": [task], "arms": list(METHODS)}
    raw = {"corpus_sha256": "frozen", "results": {m: [deepcopy(row)] for m in METHODS}}
    for method, rows in raw["results"].items():
        rows[0]["answer_payload"].update(trace={"method": method}, usage={"agent_policy_prompt_tokens": 123}, id="RAG-only-id")
    packet = answer_review_packet(raw, suite)
    assert all("method" not in item and "tool_trace" not in item for item in packet["items"])
    assert all(not {"trace", "usage", "id", "shadow_scores"} & item["answer"].keys() for item in packet["items"])
    pending = report(raw, suite)
    assert pending["headline_eligible"] is False
    assert pending["summary"]["rag"]["strict_task_success_rate"]["value"] is None
    reviewed = deepcopy(packet)
    reviewed["reviews"] = [{**review, "id": i["id"], "reviewer_type": "model", "reviewer": "GPT", "model": "actual-model",
                            "independent": False, "reason": "evidence supports facts"} for i in packet["items"]]
    final = report(raw, suite, reviewed)
    # A hand-built reviewed raw object is not a registered frozen Core run.
    assert final["headline_eligible"] is False
    assert final["summary"]["workflow"]["strict_task_success_rate"]["value"] == 1
    assert final["paired"]["rag"]["difference"] == 0
    assert final["summary"]["rag"]["retrieval"]["recall_at_k"]["1"]["value"] == .5
    raw["results"]["rag"] = []
    with pytest.raises(ValueError, match="complete task denominator"):
        report(raw, suite)


def test_empty_agent_search_is_counted_and_unknown_count_remains_unmeasured():
    task, row, _ = sample()
    row["tool_trace"] = [
        {"tool": "search_documents", "status": "succeeded", "arguments": {"query": "a"}, "result": {"evidence_count": 0}},
        {"tool": "search_documents", "status": "succeeded", "arguments": {"query": "b"}, "result": {"evidence_count": 2}},
        {"tool": "open_document", "status": "failed", "result": {"error_code": "invalid_arguments"}},
    ]
    policy = assess(task, row)["policy"]
    assert policy["failed_queries"] == 1 and policy["invalid_tool_calls"] == 1
    row["tool_trace"][0]["result"] = {}
    assert assess(task, row)["policy"]["failed_queries"] is None


def test_test_ledger_blocks_new_output_paths_and_allows_only_same_resume(tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text('{"tasks": 70}')
    reserve_test_run(freeze, tmp_path / "first.json")
    for resume in (False, True):
        with pytest.raises(ValueError, match="already reserved"):
            reserve_test_run(freeze, tmp_path / "second.json", resume=resume)
    reserve_test_run(freeze, tmp_path / "first.json", resume=True)
    freeze.write_text('{"tasks": 69}')
    with pytest.raises(ValueError, match="already reserved"):
        reserve_test_run(freeze, tmp_path / "first.json", resume=True)


def test_internal_worker_flag_cannot_bypass_original_test_registration(tmp_path):
    data = b'{"tasks":70}'
    origin = tmp_path / "benchmarks/core/freeze.json"
    origin.parent.mkdir(parents=True)
    origin.write_bytes(data)
    output = tmp_path / "result.json"
    reserve_test_run(origin, output)
    root = tmp_path / ".runtime/benchmark-snapshots" / hashlib.sha256(data).hexdigest()
    freeze = root / "benchmarks/core/freeze.json"
    freeze.parent.mkdir(parents=True)
    freeze.write_bytes(data)
    validate_test_worker(root, freeze, output)
    with pytest.raises(ValueError, match="matching original"):
        validate_test_worker(root, freeze, tmp_path / "new.json")
    with pytest.raises(ValueError, match="immutable registered"):
        validate_test_worker(tmp_path, origin, output)


def test_authorized_old_version_in_trace_does_not_fail_final_scope_check():
    task = {"id": "t", "category": "security_version_scope", "security_kind": "version_scope", "goal": "当前口令",
            "expected_status": "answered", "facts": ["current"], "fact_matchers": [{"id": "current", "aliases": ["NEW"]}],
            "forbidden": ["OLD"], "scenario_events": [], "required_slots": []}
    payload = {"status": "answered", "claims": [{"text": "口令为 NEW"}], "citations": []}
    row = provisional_score(task, payload, [], 1, 1, observable_payload={"tools": ["OLD"], "result": payload})
    assert row["forbidden_present"]["OLD"] is False
    payload["claims"][0]["text"] = "口令为 OLD"
    row = provisional_score(task, payload, [], 1, 1)
    assert row["version_scope_violation"] is True


def test_plain_rag_exercises_revocation_and_restores_acl(monkeypatch):
    from tests.test_agent import agent_db
    from app.models import Document
    from app.retrieval import RetrievalResult
    from app.security import require_chunk
    from app import qa, retrieval
    db, user = agent_db()
    db.get(Document, "doc-a").owner_id = "separate-owner"
    db.commit()
    task = {"id": "revocation", "category": "security_revocation", "user": user.id, "goal": "RPO",
            "expected_status": ["insufficient_evidence", "execution_failed"], "facts": [], "fact_matchers": [],
            "forbidden": ["15分钟"], "scenario_events": ["state_change"], "required_slots": [],
            "state_change": {"action": "revoke_document", "document_id": "doc-a", "after_tool": "search_documents"}}
    evidence = [{"chunk_id": "c2", "document_id": "doc-a", "version_id": "v2", "text": "RPO 为 15 分钟。", "title": "恢复政策", "locator": {}}]
    monkeypatch.setattr(retrieval, "retrieve_authorized", lambda *a, **k: RetrievalResult(evidence, [], 1, 1, 10, 0, 0))
    def fake_answer(db, user, question, **kwargs):
        # Contract helpers import at call time rather than retaining qa's alias.
        from app.retrieval import retrieve_authorized
        retrieve_authorized(db, user, question)
        require_chunk(db, user, "c2", active_only=True)
        pytest.fail("revocation did not prevent evidence use")
    monkeypatch.setattr(qa, "answer_question", fake_answer)
    row = run_package_arm(db, user, task, "rag")
    assert not row.get("skipped")
    assert row["scenario_events"]["state_change"] is True
    assert row["execution_error_kind"] == "authorization_change"
    assert row["answer_correct"] is None
    assert row["retrieved_evidence"][0]["chunk_id"] == "c2"
    assert db.get(Document, "doc-a").read_groups == ["engineering"]
    assert db.get(Document, "doc-a").revision == 2


def test_generation_passage_added_after_search_is_observed(monkeypatch):
    from tests.test_agent import agent_db
    from app.models import Chunk
    from app.retrieval import RetrievalResult
    from app import qa, retrieval
    db, user = agent_db()
    db.add(Chunk(id="window", version_id="v2", ordinal=1, text="恢复步骤已更新。", locator={}))
    db.commit()
    evidence = [{"chunk_id": "c2", "document_id": "doc-a", "version_id": "v2", "text": "RPO 为 15 分钟。", "title": "恢复政策", "locator": {}}]
    monkeypatch.setattr(retrieval, "retrieve_authorized", lambda *a, **k: RetrievalResult(evidence, [], 1, 1, 10, 0, 0))
    def fake_answer(db, user, question, **kwargs):
        qa.retrieve_authorized(db, user, question)
        return {"status": "answered", "claims": [{"text": "RPO 为 15 分钟。"}], "citations": [],
                "usage": {"generation_context_chunk_ids": ["c2", "window"]}}
    monkeypatch.setattr(qa, "answer_question", fake_answer)
    task = {"id": "t", "category": "efficiency_stopping", "goal": "RPO", "expected_status": "answered",
            "facts": [], "fact_matchers": [], "forbidden": [], "scenario_events": [], "required_slots": []}
    row = run_package_arm(db, user, task, "rag")
    observed = {r["chunk_id"]: r for r in row["retrieved_evidence"]}
    assert set(observed) == {"c2", "window"}
    assert observed["window"]["text"] == "恢复步骤已更新。"
    assert observed["window"]["source_sha256"] == "2" * 64


def test_freeze_rejects_user_permission_drift_before_search(monkeypatch):
    from tests.test_agent import agent_db
    from app import db as database
    from scripts.benchmark_package_runtime import indexed_corpus
    db, user = agent_db()
    monkeypatch.setattr(database, "SessionLocal", lambda: db)
    spec = {"tenant": {"id": "tenant-a"}, "owner": user.id, "documents": [], "users": [
        {"id": user.id, "tenant_id": user.tenant_id, "groups": ["support"], "role": user.role}]}
    with pytest.raises(ValueError, match="user permissions differ"):
        indexed_corpus(spec)


@pytest.mark.parametrize("arm", ["workflow", "dynamic", "hybrid"])
def test_agent_package_boundary_records_same_revocation_without_network(monkeypatch, arm):
    from tests.test_agent import agent_db
    from app.models import Document
    from app.retrieval import RetrievalResult
    from app import retrieval
    from app.clients import AgentDecision
    from agent import controller, tools, hybrid
    db, user = agent_db()
    db.get(Document, "doc-a").owner_id = "separate-owner"
    db.commit()
    task = {"id": "revocation", "category": "security_revocation", "user": user.id, "goal": "当前 RPO 是多少分钟？",
            "expected_status": ["insufficient_evidence", "execution_failed"], "facts": [], "fact_matchers": [],
            "forbidden": ["15分钟"], "scenario_events": ["state_change"], "required_slots": [], "max_steps": 4,
            "state_change": {"action": "revoke_document", "document_id": "doc-a", "after_tool": "search_documents"}}
    evidence = [{"chunk_id": "c2", "document_id": "doc-a", "version_id": "v2", "text": "RPO 为 15 分钟。", "title": "恢复政策", "locator": {}}]
    monkeypatch.setattr(retrieval, "retrieve_authorized", lambda *a, **k: RetrievalResult(evidence, [], 1, 1, 10, 0, 0))
    class Model:
        def decide_agent_action(self, goal, observations, step):
            return AgentDecision(action="search_documents", arguments={"query": goal}, purpose="retrieve") if step == 1 else AgentDecision(action="final", arguments={}, purpose="finish")
        def generate(self, *args, **kwargs):
            pytest.fail("revoked evidence must never reach generation")
    monkeypatch.setattr(controller, "Models", Model)
    monkeypatch.setattr(tools, "Models", Model)
    monkeypatch.setattr(hybrid, "choose_intent", lambda *a: hybrid.ActionIntent(kind="final"))
    row = run_package_arm(db, user, task, arm)
    assert row["scenario_events"]["state_change"] is True
    assert row["answer_correct"] is None
    assert db.get(Document, "doc-a").read_groups == ["engineering"]
