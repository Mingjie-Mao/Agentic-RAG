import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "training_gate", Path(__file__).resolve().parents[1] / "scripts/training_gate.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
assess = MODULE.assess


def test_only_adjudicated_replayable_unresolved_policy_failures_count():
    rows = [
        {"task_id": "a", "failure_type": "loop", "failure_layer": "policy", "adjudicated": True,
         "resolved": False, "replay": "fixture-a"},
        {"task_id": "a", "failure_type": "loop", "failure_layer": "policy", "adjudicated": True,
         "resolved": False, "replay": "fixture-a"},
        {"task_id": "b", "failure_type": "wrong_answer", "failure_layer": "generation", "adjudicated": True,
         "resolved": False, "replay": "fixture-b"},
        {"task_id": "c", "failure_type": "stop", "failure_layer": "policy", "adjudicated": False,
         "resolved": False, "replay": "fixture-c"},
    ]
    result = assess(rows)
    assert result["qualified_unresolved_unique_policy_failures"] == 1
    assert result["decision"] == "do_not_train"


def test_training_gate_needs_quantity_and_failure_diversity():
    rows = [
        {"task_id": str(i), "failure_type": ("loop", "stop", "wrong_tool")[i % 3],
         "failure_layer": "policy", "adjudicated": True, "resolved": False, "replay": f"case-{i}"}
        for i in range(50)
    ]
    assert assess(rows)["decision"] == "consider_training_experiment"
