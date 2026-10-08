"""Server-owned cumulative budgets; resumed tasks cannot reset model-call limits."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import signal
import threading
import time


class BudgetExceeded(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class TaskCancelled(RuntimeError):
    pass


_CURRENT = ContextVar("execution_budget", default=None)


def current_budget():
    return _CURRENT.get()


def bounded_model(role):
    """Count policy schema failures too; generation counts each transport call."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            budget = current_budget()
            if budget is None:
                return function(*args, **kwargs)
            before = dict(getattr(args[0], "agent_policy_usage", None) or {})
            with budget.call(role) as usage:
                try:
                    result = function(*args, **kwargs)
                    if isinstance(result, dict):
                        usage.update(prompt_tokens=(result.get("prompt_eval_count") or 0),
                                     completion_tokens=(result.get("eval_count") or 0))
                        for field,target in [('total_duration','server_ms'),('load_duration','load_ms'),
                                             ('prompt_eval_duration','prompt_eval_ms'),('eval_duration','completion_eval_ms')]:
                            if field in result:
                                usage[target]=round(result[field]/1e6,1)
                    return result
                finally:
                    if role == "policy":
                        after = getattr(args[0], "agent_policy_usage", None) or {}
                        for key in ("prompt_tokens", "completion_tokens"):
                            usage[key] = after.get(key, 0) - before.get(key, 0)
                        # Server timings only when this call reported them; never 0 for unknown.
                        for key, target in (("model_duration_ms", "server_ms"), ("model_load_ms", "load_ms"),
                                            ("prompt_eval_ms", "prompt_eval_ms"),
                                            ("completion_eval_ms", "completion_eval_ms")):
                            if after.get(key, 0) != before.get(key, 0):
                                usage[target] = round(after[key] - before.get(key, 0), 1)
        return wrapped
    return decorate


class ExecutionBudget:
    def __init__(self, cfg, saved=None, *, persist=None, cancelled=None, clock=time.time):
        self.clock, self.persist, self.cancelled = clock, persist, cancelled
        self.state = dict(saved or {})
        self.state.setdefault("deadline", clock() + cfg.agent_task_timeout_seconds)
        self.state.setdefault("limits", {
            "policy": cfg.agent_policy_max_calls, "judge": cfg.agent_judge_max_calls,
            "generation": cfg.agent_generation_max_calls, "judge_tokens": cfg.agent_judge_token_budget,
        })
        self.state["calls"] = {key: dict(value) for key, value in self.state.get("calls", {}).items()}
        for row in self.state["calls"].values():
            # A reserved attempt without a completion was interrupted. Never give
            # its budget back on resume, and expose it as a failed attempt.
            outstanding = row["attempted"] - row["succeeded"] - row["failed"]
            row["failed"] += max(0, outstanding)
        self.state.setdefault("judge_tokens_reserved", 0)
        self.state.setdefault('model_call_records',[])
        self._save()

    def _save(self):
        if self.persist:
            self.persist(self.snapshot())

    def snapshot(self):
        return {**self.state, "calls": {k: dict(v) for k, v in self.state["calls"].items()},
                'model_call_records':[dict(r) for r in self.state['model_call_records']]}

    def check(self):
        if self.cancelled and self.cancelled():
            raise TaskCancelled("task_cancelled")
        if self.clock() >= self.state["deadline"]:
            raise BudgetExceeded("task_deadline_exceeded")

    def timeout(self, configured):
        self.check()
        return max(0.001, min(configured, self.state["deadline"] - self.clock()))

    @contextmanager
    def call(self, role, *, reserve_tokens=0):
        self.check()
        calls = self.state["calls"].setdefault(role, {
            "attempted": 0, "succeeded": 0, "failed": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "wall_ms": 0.0,
        })
        if calls["attempted"] >= self.state["limits"].get(role, 100):
            raise BudgetExceeded(f"{role}_call_budget_exhausted")
        if role == "judge":
            if self.state["judge_tokens_reserved"] + reserve_tokens > self.state["limits"]["judge_tokens"]:
                raise BudgetExceeded("judge_token_budget_exhausted")
            # Reserve the upper bound before dispatch. Failed/in-flight calls retain
            # reservations across recovery; successful calls settle to actual usage.
            self.state["judge_tokens_reserved"] += reserve_tokens
        calls["attempted"] += 1
        self._save()
        started = time.monotonic()
        usage = {}
        succeeded=False
        try:
            yield usage
            self.check()
        except BaseException:
            calls["failed"] += 1
            raise
        else:
            calls["succeeded"] += 1
            succeeded=True
            if role == "judge":
                actual = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                self.state["judge_tokens_reserved"] += actual - reserve_tokens
        finally:
            for key in ("prompt_tokens", "completion_tokens"):
                calls[key] += usage.get(key, 0)
            elapsed=round((time.monotonic() - started) * 1000, 1)
            calls["wall_ms"] += elapsed
            self.state['model_call_records'].append({'role':role,'status':'ok' if succeeded else 'failed',
                'wall_ms':elapsed,**usage})
            self._save()


@contextmanager
def use_budget(budget):
    token = _CURRENT.set(budget)
    previous_handler, previous_timer, started = None, None, time.monotonic()
    # The worker runs tasks on its main thread: interrupt even a blocking dependency.
    # API thread pools use deadline checks and bounded transport timeouts instead.
    if threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer"):
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        remaining = max(0.001, budget.state["deadline"] - budget.clock())
        if previous_timer[0] > 0:
            remaining = min(remaining, previous_timer[0])

        def alarm(_signum, _frame):
            raise BudgetExceeded("task_deadline_exceeded")

        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        budget.check()
        yield budget
    finally:
        if previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL,
                                max(0.001, previous_timer[0] - (time.monotonic() - started)),
                                previous_timer[1])
        _CURRENT.reset(token)
