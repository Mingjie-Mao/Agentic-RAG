from types import SimpleNamespace
import time

import pytest

from app.execution_budget import BudgetExceeded, ExecutionBudget, TaskCancelled, bounded_model, current_budget, use_budget


def config(**changes):
    return SimpleNamespace(**({"agent_task_timeout_seconds": 60, "agent_policy_max_calls": 2,
        "agent_judge_max_calls": 2, "agent_judge_token_budget": 100,
        "agent_generation_max_calls": 2} | changes))


def test_policy_attempts_failures_and_resume_never_reset_budget():
    records = []
    budget = ExecutionBudget(config(), persist=records.append)
    with pytest.raises(ValueError), budget.call("policy"):
        raise ValueError("malformed policy")
    saved = records[-1]
    resumed = ExecutionBudget(config(agent_policy_max_calls=100), saved)
    with resumed.call("policy") as usage:
        usage.update(prompt_tokens=12, completion_tokens=3)
    with pytest.raises(BudgetExceeded, match="policy_call_budget_exhausted"):
        with resumed.call("policy"):
            pass
    row = resumed.snapshot()["calls"]["policy"]
    assert (row["attempted"], row["succeeded"], row["failed"], row["prompt_tokens"]) == (2, 1, 1, 12)
    assert resumed.state["deadline"] == saved["deadline"]


def test_crash_counts_inflight_as_failed_and_retains_reservation():
    budget = ExecutionBudget(config())
    saved = budget.snapshot()
    saved["calls"] = {"judge": dict(attempted=1, succeeded=0, failed=0, prompt_tokens=0, completion_tokens=0, wall_ms=0)}
    saved["judge_tokens_reserved"] = 70
    resumed = ExecutionBudget(config(), saved)
    assert resumed.state["calls"]["judge"]["failed"] == 1
    with pytest.raises(BudgetExceeded, match="judge_token_budget_exhausted"):
        with resumed.call("judge", reserve_tokens=31):
            pass


def test_token_reservation_settles_actual_usage_but_failed_request_keeps_it():
    budget = ExecutionBudget(config())
    with budget.call("judge", reserve_tokens=70) as usage:
        usage.update(prompt_tokens=10, completion_tokens=5)
    assert budget.state["judge_tokens_reserved"] == 15
    with pytest.raises(RuntimeError), budget.call("judge", reserve_tokens=50):
        raise RuntimeError("transport unavailable")
    assert budget.state["judge_tokens_reserved"] == 65
    assert budget.state["calls"]["judge"]["attempted"] == 2


def test_deadline_cancellation_timeout_and_context_restoration():
    budget = ExecutionBudget(config(), clock=lambda: 100)
    assert budget.timeout(180) == 60
    budget.state["deadline"] = 99
    with pytest.raises(BudgetExceeded):
        with use_budget(budget):
            pass
    assert current_budget() is None
    cancelled = ExecutionBudget(config(), cancelled=lambda: True)
    with pytest.raises(TaskCancelled):
        with cancelled.call("generation"):
            pass
    assert cancelled.state["calls"] == {}


def test_worker_deadline_interrupts_blocking_dependency():
    budget = ExecutionBudget(config(agent_task_timeout_seconds=.03))
    with pytest.raises(BudgetExceeded, match="task_deadline_exceeded"):
        with use_budget(budget), budget.call("policy"):
            time.sleep(.2)
    assert budget.state["calls"]["policy"]["failed"] == 1
    assert current_budget() is None


def test_policy_schema_failure_tokens_are_counted_and_each_generation_separate():
    class Model:
        @bounded_model("policy")
        def policy(self):
            self.agent_policy_usage = dict(prompt_tokens=10, completion_tokens=2)
            raise ValueError("invalid JSON")

        @bounded_model("generation")
        def chat(self):
            return dict(prompt_eval_count=5, eval_count=3)
    budget = ExecutionBudget(config())
    with use_budget(budget):
        with pytest.raises(ValueError):
            Model().policy()
        Model().chat()
        Model().chat()
    calls = budget.snapshot()["calls"]
    assert calls["policy"]["failed"] == 1 and calls["policy"]["prompt_tokens"] == 10
    assert calls["generation"]["attempted"] == 2 and calls["generation"]["completion_tokens"] == 6


def test_controller_budget_exhaustion_cancel_and_resume_have_persisted_terminal_states(monkeypatch):
    from test_agent import agent_db
    from agent.controller import create_task, run_task
    from agent.tools import ToolResult
    from app.config import settings
    monkeypatch.setattr(settings(), "agent_policy_max_calls", 1)
    db, user = agent_db()
    class ExpensiveTools:
        def call(self, *args):
            with current_budget().call("policy"):
                pass
            with current_budget().call("policy"):
                pass
    task = create_task(db, user, "当前RPO是多少？", "workflow", 4)
    with pytest.raises(BudgetExceeded, match="policy_call_budget_exhausted"):
        run_task(db, user, task, models=object(), tools=ExpensiveTools())
    assert task.status == "failed" and task.error == "policy_call_budget_exhausted"
    assert task.input["_execution_budget"]["calls"]["policy"]["attempted"] == 1
    saved = task.input["_execution_budget"]
    saved["deadline"] = time.time() - 1
    task.input = dict(task.input, _execution_budget=saved)
    db.commit()
    with pytest.raises(BudgetExceeded, match="task_deadline_exceeded"):
        run_task(db, user, task, models=object(), tools=ExpensiveTools())
    assert task.status == "failed"
    class CancellingTools:
        def call(self, *args):
            cancelled.status = "cancelled"
            db.commit()
            return ToolResult("ok", {}, ["c2"], {})
    cancelled = create_task(db, user, "当前RPO是多少？", "workflow", 4)
    run_task(db, user, cancelled, models=object(), tools=CancellingTools())
    assert cancelled.status == "cancelled" and not cancelled.result


def test_http_budget_exhaustion_has_clear_service_response():
    import asyncio
    import json
    from app.main import execution_budget_error
    response = asyncio.run(execution_budget_error(None, BudgetExceeded("generation_call_budget_exhausted")))
    assert response.status_code == 503
    payload = json.loads(response.body)
    assert payload["stage"] == "execution_budget" and not payload.get("claims")


def test_model_stage_measurements_survive_resume_and_do_not_infer_queue_time():
    class Model:
        @bounded_model('generation')
        def chat(self):
            return dict(prompt_eval_count=7,eval_count=2,total_duration=12_000_000,
                        load_duration=1_000_000,prompt_eval_duration=4_000_000,eval_duration=6_000_000)
    budget=ExecutionBudget(config())
    with use_budget(budget):
        Model().chat()
    record=budget.snapshot()['model_call_records'][0]
    assert record['role']=='generation' and record['status']=='ok'
    assert record['server_ms']==12 and record['prompt_eval_ms']==4 and record['completion_eval_ms']==6
    assert 'queue_ms' not in record
    resumed=ExecutionBudget(config(),budget.snapshot())
    assert resumed.snapshot()['model_call_records']==[record]


def test_policy_call_record_keeps_server_timings_and_leaves_unknown_absent():
    from app.clients import accumulate_policy_usage

    models = SimpleNamespace()

    @bounded_model("policy")
    def decide(models, result):
        accumulate_policy_usage(models, result, 0.0)

    budget = ExecutionBudget(config())
    with use_budget(budget):
        decide(models, {"prompt_eval_count": 900, "eval_count": 40, "total_duration": 9_000_000_000,
                        "load_duration": 50_000_000, "prompt_eval_duration": 7_000_000_000,
                        "eval_duration": 1_900_000_000})
        decide(models, {"prompt_eval_count": 10, "eval_count": 2})
    first, second = budget.snapshot()["model_call_records"]
    assert first["prompt_tokens"] == 900 and first["server_ms"] == 9000.0
    assert first["prompt_eval_ms"] == 7000.0 and first["completion_eval_ms"] == 1900.0 and first["load_ms"] == 50.0
    assert second["prompt_tokens"] == 10 and "server_ms" not in second and "prompt_eval_ms" not in second
    assert models.agent_policy_usage["prompt_eval_ms"] == 7000.0
