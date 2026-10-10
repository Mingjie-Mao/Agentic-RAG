from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import json

from scripts.prerender_site import render

ROOT = Path(__file__).resolve().parents[1]


def copy_site_inputs(directory):
    for name in ("index.html", "data.json", "replay-20261009T072349Z.json"):
        shutil.copy(ROOT / "site" / name, directory / name)


class Structure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag in {"section", "details", "main", "nav"}:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in {"section", "details", "main", "nav"}:
            assert self.stack and self.stack.pop() == tag, f"unbalanced {tag}"


def test_home_keeps_architecture_evaluation_and_all_referenced_sections():
    text = (ROOT / "site/index.html").read_text()
    parser = Structure()
    parser.feed(text)
    assert not parser.stack
    assert len(parser.ids) == len(set(parser.ids))
    assert {"demo", "features", "architecture", "evaluation", "decisions", "why-build"} <= set(parser.ids)
    for anchor in re.findall(r'href="#([^" ]+)"', text):
        assert anchor in parser.ids
    for image in re.findall(r'src="(shots/[^" ]+)"', text):
        assert (ROOT / "site" / image).is_file()


def test_historical_metrics_are_present_without_javascript_and_core_stays_pending(tmp_path):
    copy_site_inputs(tmp_path)
    render(tmp_path)
    text = (tmp_path / "index.html").read_text()
    for identifier in ("t-holdout", "t-dev", "t-public"):
        assert re.search(f'<tbody id="{identifier}">\\s*<tr>', text)
    assert "90.0%" in text and "待正式运行" in text
    assert 'fetch("data.json")' not in text


def test_offline_replay_references_existing_recordings_and_does_not_call_model():
    text = (ROOT / "site/replay.html").read_text()
    names = re.findall(r'fetch\("([^" ]+\.json)"', text)
    assert len(names) == 2
    assert all((ROOT / "site" / name).is_file() for name in names)
    assert "/api/" not in text
    assert "历史回放" in text


def test_missing_measurements_remain_pending_instead_of_zero_or_failure(tmp_path):
    copy_site_inputs(tmp_path)
    data = json.loads((ROOT / "site/data.json").read_text())
    for key in ("status_correct", "literal_fact_coverage", "permission_violations", "hidden_source_leaks", "citation_identity_all_valid"):
        data["holdout"]["measured"][key] = None
    data["retrieval"]["bm25"]["recall_at_5"] = None
    data["retrieval"]["bm25"]["mrr"] = None
    data["public"]["retrieval"]["bm25"]["recall_at_5"] = None
    (tmp_path / "data.json").write_text(json.dumps(data))
    render(tmp_path)
    text = (tmp_path / "index.html").read_text()
    holdout = re.search(r'<tbody id="t-holdout">(.*?)</tbody>', text, re.S).group(1)
    assert holdout.count("pending") == 5
    assert "failed" not in holdout and ">None<" not in holdout
    assert "待核验" in holdout


def test_historical_prerender_is_stable_and_does_not_modify_source_data(tmp_path):
    copy_site_inputs(tmp_path)
    original_data = (tmp_path / "data.json").read_bytes()
    render(tmp_path)
    first = (tmp_path / "index.html").read_bytes()
    render(tmp_path)
    assert (tmp_path / "index.html").read_bytes() == first
    assert (tmp_path / "data.json").read_bytes() == original_data


def test_execution_summary_uses_only_recorded_tools_and_preserves_escaped_sources(tmp_path):
    copy_site_inputs(tmp_path)
    capture_path = tmp_path / "replay-20261009T072349Z.json"
    capture = json.loads(capture_path.read_text())
    run = capture["scenarios"][0]
    run["evidence"][0]["title"] = '<script>untrusted source</script>'
    capture_path.write_text(json.dumps(capture))
    original_capture = capture_path.read_bytes()
    render(tmp_path)
    text = (tmp_path / "index.html").read_text()
    summary = re.search(r'<!-- recorded-run:start -->(.*?)<!-- recorded-run:end -->', text, re.S).group(1)
    tools = [e["tool_name"] for e in run["events"] if e["event_type"] == "tool_completed"]
    for tool in tools:
        assert f'<strong>{tool}</strong>' in summary
    assert len(tools) == 2
    assert 'compare_versions' not in summary and 'verify_chunk_access' not in summary
    assert run["task_id"] not in summary and capture["recorded_at"] not in summary
    assert capture["model"] not in summary and '34.4s' not in summary
    assert '真实执行回放' in summary and 'Workflow' in summary
    assert '&lt;script&gt;untrusted source&lt;/script&gt;' in summary
    assert capture_path.read_bytes() == original_capture
