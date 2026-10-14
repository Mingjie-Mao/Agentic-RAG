import json
from copy import deepcopy

from scripts.validate_benchmark_v2 import MANIFEST, ROOT, validate_manifest, validate_repository


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def write_suite(root, path, tasks):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"tasks": tasks}, ensure_ascii=False), encoding="utf-8")


def tiny_manifest(known):
    data = deepcopy(manifest())
    data["suites"] = [
        {"id": "dev", "path": "dev.json", "expected_tasks": 2, "status": "existing"},
        {"id": "test", "path": "test.json", "expected_tasks": 2, "status": "existing"},
    ]
    data["known_exact_question_overlap"] = known
    return data


def test_repository_manifest_is_consistent_with_registered_overlap():
    data = manifest()
    assert validate_manifest(data) == []
    issues, info = validate_repository(data, ROOT)
    assert issues == []
    assert any("8 registered exact-overlap tasks" in line for line in info)


def test_proposed_manifest_cannot_claim_frozen_status():
    data = manifest()
    data["status"] = "frozen"
    assert "The proposed manifest must not claim tested/frozen status" in validate_manifest(data)


def test_unregistered_and_stale_overlap_both_fail(tmp_path):
    write_suite(tmp_path, "dev.json", [{"id": "D1", "question": "同一题"}, {"id": "D2", "question": "甲"}])
    write_suite(tmp_path, "test.json", [{"id": "T1", "question": "同 一题"}, {"id": "T2", "question": "乙"}])
    issues, _ = validate_repository(tiny_manifest({}), tmp_path)
    assert any("unregistered exact question overlap ['T1']" in i for i in issues)
    issues, _ = validate_repository(tiny_manifest({"test": ["T1", "T2"]}), tmp_path)
    assert any("registered overlap no longer present ['T2']" in i for i in issues)
    issues, _ = validate_repository(tiny_manifest({"test": ["T1"]}), tmp_path)
    assert issues == []
