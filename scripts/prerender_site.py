"""Render historical tables from their existing data without client-side fetch."""

import argparse
import html
import json
from pathlib import Path
import re


def both(zh, en):
    return f'<span class="zh">{html.escape(zh)}</span><span class="en">{html.escape(en)}</span>'


def recorded_run(site):
    name = "replay-20261009T072349Z.json"
    capture = json.loads((site / name).read_text())
    run = capture["scenarios"][0]
    labels = {"generation_context": ("准备生成所需证据", "Prepare generation evidence"),
              "coverage_check": ("检查问题覆盖情况", "Check answer coverage"),
              "task_completed": ("返回核验结果", "Return the verified result")}
    items = []
    for event in run["events"]:
        if event["event_type"] == "tool_completed":
            title = html.escape(event["tool_name"])
            detail = '<small>' + both(f'{event["evidence_count"]} 条证据', f'{event["evidence_count"]} evidence spans') + '</small>'
        elif event["event_type"] in labels:
            title = both(*labels[event["event_type"]])
            detail = ""
        else:
            continue
        items.append(f'<li><strong>{title}</strong>{detail}</li>')
    claims = "".join(f'<p>{html.escape(claim["text"])}</p>' for claim in run["claims"])
    sources = "".join('<p>' + both("来源", "Source") + ': '
                      + html.escape(source["title"]) + ' · ' + html.escape(source["locator"]["label"]) + '</p>'
                      for source in run["evidence"])
    return (f'<p class="run-meta">{both("真实执行回放", "Recorded execution")} · {html.escape(run["mode"].capitalize())}</p>'
            f'<p>{both("任务", "Task")}: {html.escape(run["goal"])}</p>'
            '<div class="run-layout"><ol class="run-steps" aria-label="Recorded task events">'
            + "".join(items) + '</ol><div class="run-result"><h3>'
            + both("最终回答", "Final answer")
            + '</h3>' + claims + sources
            + f'<a href="/replay.html">{both("逐步回放与原文", "Replay steps and source")}</a>'
            '</div></div>')


def render(site):
    path = site / "index.html"
    text = path.read_text()
    data = json.loads((site / "data.json").read_text())
    def pct(v):
        return f"{v * 100:.1f}%" if v is not None else "pending"
    def number(v, format_spec=""):
        return format(v, format_spec) if v is not None else "pending"
    measured, targets = data["holdout"]["measured"], data["holdout"]["targets"]
    holdout = [
        [
            both("回答状态正确率", "Answer-state accuracy"),
            "≥ " + pct(targets["status_correct_min"]),
            pct(measured["status_correct"]),
        ],
        [
            both("字面事实覆盖", "Literal fact coverage"),
            "≥ " + pct(targets["literal_fact_coverage_min"]),
            pct(measured["literal_fact_coverage"]),
        ],
        [
            both("越权命中", "Unauthorized hits"),
            str(targets["permission_violations_max"]),
            number(measured["permission_violations"]),
        ],
        [
            both("隐藏文档泄漏", "Hidden-source leaks"),
            str(targets["hidden_source_leaks_max"]),
            number(measured["hidden_source_leaks"]),
        ],
        [
            both("引用身份校验", "Citation identity check"),
            both("全部通过", "all valid"),
            both("全部通过", "all valid")
            if measured["citation_identity_all_valid"] is True
            else both("待核验", "pending") if measured["citation_identity_all_valid"] is None
            else both("未通过", "failed"),
        ],
    ]
    names = {
        "bm25": both("BM25", "BM25"),
        "dense": both("向量检索", "Dense vectors"),
        "hybrid": both("混合检索（默认）", "Hybrid (default)"),
    }
    tables = {
        "t-holdout": holdout,
        "t-dev": [
            [
                names[k],
                number(data['retrieval'][k]['recall_at_5'], ".3f"),
                number(data['retrieval'][k]['mrr'], ".3f"),
                pct(data["dev"][k]["status_correct"]),
            ]
            for k in names
        ],
        "t-public": [
            [
                names[k],
                number(data['public']['retrieval'][k]['recall_at_5'], ".3f"),
                pct(data["public"]["end_to_end"][k]["status_correct"]),
            ]
            for k in names
        ],
    }
    for identifier, rows in tables.items():
        body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
        text, count = re.subn(
            f'<tbody id="{identifier}">.*?</tbody>',
            f'<tbody id="{identifier}">{body}</tbody>',
            text,
            flags=re.S,
        )
        if count != 1:
            raise ValueError(f"Missing historical table: {identifier}")
    text, count = re.subn(r'<!-- recorded-run:start -->.*?<!-- recorded-run:end -->',
                         lambda _: '<!-- recorded-run:start -->' + recorded_run(site) + '<!-- recorded-run:end -->',
                         text, flags=re.S)
    if count != 1:
        raise ValueError("Missing recorded execution summary")
    path.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, default=Path("site"))
    render(parser.parse_args().site)
