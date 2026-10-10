from uuid import uuid4

import pytest
from fastapi import HTTPException, Response

from app.chat_progress import chat_progress, complete_chat, report_stage, track_chat
from app.config import settings
from app.main import demo_status, enter_visitor
from app.models import LoginSession, User
from app.security import COOKIE, hash_token
from app.trial import is_trial_user, require_trial_read_only, reserve_trial_request, visitor_user
from test_trial import enable_trial, trial_db


def test_visitor_entry_preserves_read_only_budget_and_is_not_the_source_account(monkeypatch):
    db, source = trial_db()
    enable_trial(monkeypatch)
    monkeypatch.setattr(settings(), "demo_mode", True)
    response = Response()
    payload = enter_visitor(response, db)
    assert payload["id"] != source.id
    assert payload["role"] == "member"
    visitor = db.get(User, payload["id"])
    assert is_trial_user(visitor)
    with pytest.raises(HTTPException, match="只读"):
        require_trial_read_only(visitor)
    reserve_trial_request(db, visitor, "chat")
    assert visitor_user(db).id == visitor.id
    again = enter_visitor(Response(), db)
    assert again["trial"]["remaining"] == 1
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie
    token = cookie.split(COOKIE + "=", 1)[1].split(";", 1)[0]
    assert db.get(LoginSession, hash_token(token)).user_id == visitor.id


def test_visitor_entry_is_disabled_outside_public_trial(monkeypatch):
    db, _ = trial_db()
    enable_trial(monkeypatch)
    monkeypatch.setattr(settings(), "demo_mode", False)
    assert not demo_status()["visitor_enabled"]
    with pytest.raises(HTTPException) as error:
        enter_visitor(Response(), db)
    assert error.value.status_code == 404


def test_visitor_never_inherits_admin_access(monkeypatch):
    db, source = trial_db()
    enable_trial(monkeypatch)
    monkeypatch.setattr(settings(), "demo_mode", True)
    source.role = "admin"
    db.commit()
    with pytest.raises(HTTPException) as error:
        visitor_user(db)
    assert error.value.status_code == 503


def test_progress_cannot_be_read_across_users_and_does_not_repeat_execution():
    _, user = trial_db()
    identifier = str(uuid4())
    with track_chat(user, identifier) as row:
        report_stage("retrieval")
        assert chat_progress(user, identifier)["stage"] == "retrieval"
        other = User(id="other", tenant_id=user.tenant_id)
        with pytest.raises(HTTPException) as error:
            chat_progress(other, identifier)
        assert error.value.status_code == 404
        complete_chat(row, "answer-id")
    assert chat_progress(user, identifier)["answer_id"] == "answer-id"
    with pytest.raises(HTTPException) as error:
        with track_chat(user, identifier):
            pytest.fail("duplicate execution")
    assert error.value.status_code == 409


def test_progress_records_failure_and_restores_callback_context():
    _, user = trial_db()
    identifier = str(uuid4())
    with pytest.raises(RuntimeError):
        with track_chat(user, identifier):
            report_stage("generation")
            raise RuntimeError("dependency unavailable")
    report_stage("completed")
    result = chat_progress(user, identifier)
    assert result["status"] == "failed"
    assert result["stage"] == "generation"
    assert "answer_id" not in result


def test_progress_is_hidden_across_tenants_even_when_user_ids_match():
    _, user = trial_db()
    identifier = str(uuid4())
    with track_chat(user, identifier):
        report_stage("generation")
        other = User(id=user.id, tenant_id="another-tenant")
        with pytest.raises(HTTPException) as error:
            chat_progress(other, identifier)
        assert error.value.status_code == 404


def test_expired_progress_reports_unknown_instead_of_false_success(monkeypatch):
    import importlib

    module = importlib.import_module("app.chat_progress")
    clock = [module.monotonic()]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    _, user = trial_db()
    identifier = str(uuid4())
    with track_chat(user, identifier):
        report_stage("generation")
    clock[0] += module.TTL_SECONDS + 1
    with pytest.raises(HTTPException) as error:
        chat_progress(user, identifier)
    assert error.value.status_code == 404


def test_exhausted_request_does_not_start_model_or_refund_previous_quota(monkeypatch):
    from fastapi import Request
    import app.main as main
    from app.trial import trial_status

    db, user = trial_db()
    enable_trial(monkeypatch)
    reserve_trial_request(db, user, "chat")
    reserve_trial_request(db, user, "chat")
    monkeypatch.setattr(main, "answer_question", lambda *args: pytest.fail("model must not start"))
    identifier = str(uuid4())
    request = Request({"type": "http", "headers": [(b"x-chat-request", identifier.encode())]})
    with pytest.raises(HTTPException) as error:
        main.chat(main.QuestionBody(question="API 配额是多少？"), user, db, request)
    assert error.value.status_code == 429
    assert trial_status(db, user)["remaining"] == 0
    assert chat_progress(user, identifier)["status"] == "failed"


def test_agent_cancel_reports_finished_on_repeat_and_does_not_refund_quota(monkeypatch):
    from sqlalchemy import func, select
    from agent.controller import cancel_task, create_task
    from app.models import AgentEvent
    from app.trial import trial_status

    db, user = trial_db()
    enable_trial(monkeypatch)
    reserve_trial_request(db, user, "agent")
    task = create_task(db, user, "比较政策变化", "workflow", 4)
    cancel_task(db, user, task.id)
    with pytest.raises(HTTPException) as error:
        cancel_task(db, user, task.id)
    assert error.value.status_code == 409
    assert task.status == "cancelled"
    assert db.scalar(select(func.count()).select_from(AgentEvent).where(
        AgentEvent.task_id == task.id, AgentEvent.event_type == "task_cancelled")) == 1
    assert trial_status(db, user)["remaining"] == 1


def test_agent_cancel_cannot_change_another_users_task():
    from agent.controller import cancel_task, create_task

    db, user = trial_db()
    task = create_task(db, user, "比较政策变化", "workflow", 4)
    other = User(id="another-user", tenant_id=user.tenant_id)
    with pytest.raises(HTTPException) as error:
        cancel_task(db, other, task.id)
    assert error.value.status_code == 404
    assert task.status == "queued"
