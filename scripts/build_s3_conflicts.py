"""Extend the frozen v1 set into s3-v2: real conflicts plus look-alikes that must not be conflicts.

v1 files are copied byte-identical so shared questions stay comparable. Only new
families are added, and all of them are development; the held-out split is untouched.
"""

import hashlib
import json
from pathlib import Path
import re
import shutil

from docx import Document
from openpyxl import Workbook
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate

SOURCE = Path("fixtures/s3")
ROOT = Path("fixtures/s3-v2")
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
STYLE = ParagraphStyle("cn", fontName="STSong-Light", fontSize=11, leading=17)

# Every subject below is disjoint from the v1 corpus; assert_disjoint enforces that.
# Reusing a v1 subject would silently contradict v1 gold labels that say "answered".
CONFLICTS = [
    ("cert-renewal", "证书续期提前量", "提前量", "30 天", "60 天"),
    ("drill-report", "演练报告提交时限", "提交时限", "3 个工作日", "10 个工作日"),
    ("badge-reissue", "门禁卡补办周期", "补办周期", "2 个工作日", "5 个工作日"),
    ("room-booking", "会议室预订上限", "单次预订上限", "4 小时", "8 小时"),
    ("desk-move", "工位搬迁通知期", "通知期", "5 个工作日", "2 个工作日"),
    ("training-hours", "培训学时要求", "年度学时", "16 学时", "40 学时"),
    ("book-loan", "图书借阅期限", "借阅期限", "14 天", "30 天"),
    ("medical-booking", "体检预约提前期", "提前期", "7 天", "21 天"),
    ("parcel-signoff", "快递签收时限", "签收时限", "24 小时", "72 小时"),
    ("invoice-issue", "发票开具时限", "开具时限", "5 个工作日", "15 个工作日"),
    ("shuttle-headway", "班车发车间隔", "发车间隔", "20 分钟", "45 分钟"),
    ("team-budget", "团建经费上限", "人均上限", "300 澳元", "800 澳元"),
]

# Each look-alike is a pattern the checker used to mis-fire on.
LOOKALIKES = [
    ("metric", "plant-care", "绿植养护频次", [("浇水频次", "每周 2 次"), ("修剪频次", "每季度 1 次")]),
    (
        "metric",
        "first-aid",
        "急救箱检查周期",
        [("日常检查周期", "每月 1 次"), ("药品更换周期", "每 6 个月")],
    ),
    ("metric", "print-quota", "打印配额", [("每月黑白页数", "500 页"), ("每月彩色页数", "80 页")]),
    ("object", "parking-slot", "停车位分配数量", [("普通员工", "1 个"), ("部门负责人", "2 个")]),
    ("object", "uniform-issue", "工服发放数量", [("行政岗", "2 套"), ("现场岗", "4 套")]),
    ("object", "locker-term", "储物柜使用期限", [("短期访客", "7 天"), ("长期员工", "365 天")]),
    (
        "version",
        "lost-found",
        "失物招领保管期",
        [("2026-08-31 之前", "30 天"), ("2026-09-01 起", "90 天")],
    ),
    (
        "version",
        "minutes-due",
        "会议纪要提交时限",
        [("2026-08-31 之前", "3 个工作日"), ("2026-09-01 起", "1 个工作日")],
    ),
    (
        "version",
        "archive-request",
        "纸质档案调阅时限",
        [("2026-08-31 之前", "5 个工作日"), ("2026-09-01 起", "2 个工作日")],
    ),
]

# The hard negatives: two documents, one subject, different numbers, no contradiction.
CROSS_LOOKALIKES = [
    ("object", "namecard", "名片印制批量", "新入职员工", "100 张", "客户经理", "500 张"),
    ("object", "medical-items", "年度体检项目数", "普通岗位", "12 项", "特种作业岗位", "20 项"),
    ("object", "shuttle-stops", "班车站点数量", "市区线路", "8 个", "郊区线路", "3 个"),
    (
        "period",
        "umbrella-loan",
        "雨伞借用期限",
        "2026-08-31 之前借出的雨伞",
        "3 天",
        "2026-09-01 起借出的雨伞",
        "7 天",
    ),
    (
        "period",
        "fire-drill",
        "消防演练频次",
        "2026-08-31 之前的年度计划",
        "每年 1 次",
        "2026-09-01 起的年度计划",
        "每半年 1 次",
    ),
    (
        "period",
        "aircon-setpoint",
        "空调温度设定",
        "2026-08-31 之前",
        "24 摄氏度",
        "2026-09-01 起",
        "26 摄氏度",
    ),
]


def assert_disjoint():
    """New subjects must not exist in v1, or v1 gold labelled "answered" becomes a lie."""
    from app.evaluation import prepare_snapshot

    _, _, blocks = prepare_snapshot(str(SOURCE))
    corpus = "\n".join(item["text"] for value in blocks.values() for item in value)
    subjects = (
        [row[1] for row in CONFLICTS]
        + [row[2] for row in LOOKALIKES]
        + [row[2] for row in CROSS_LOOKALIKES]
    )
    stem = re.compile(
        r"(时限|频次|周期|期限|数量|批量|上限|要求|时长|提前量|提前期|配额|间隔|设定|项目数|学时)$"
    )
    collisions = [
        subject
        for subject in subjects
        if subject in corpus or (stem.sub("", subject) and stem.sub("", subject) in corpus)
    ]
    assert not collisions, (
        f"New subjects already discussed in v1: {collisions}. "
        "Adding a second opinion on an existing subject turns v1 'answered' questions into real conflicts."
    )
    assert len(set(subjects)) == len(subjects), "Duplicate subject across the new families"


def cross_paragraphs(title, subject, scope, value, other_scope):
    return [
        f"{title}（2026-09 版）",
        f"本页只规定{scope}的{subject}，不涉及其他对象。",
        f"{scope}的{subject}为 {value}。",
        f"{other_scope}的{subject}另有规定，数值不同是因为适用对象不同，两页并不矛盾，不需要裁决。",
        f"引用本页数值时必须同时写明适用对象是{scope}，不得脱离对象单独引用。",
        "来源：本项目自建虚构业务资料，不代表真实企业。",
    ]


def paragraphs(title, subject, field, value, stance, other):
    return [
        f"{title}（2026-09 版）",
        f"本页规定星桥软件的{subject}，自 2026 年 9 月 1 日起执行，适用于全部客户与全部环境，没有地区或套餐差别。",
        f"{subject}的{field}为 {value}。",
        f"本页由{stance}维护。除本页外，{other}也在对外说明{subject}；两处口径不一致时不得自行选择其中一方，"
        "应保留两份来源并交由负责人裁决。",
        f"本页未设置失效日期，也不声明优先于其他文件。记录{field}时保留原始单位，不折算成其他计时口径。",
        "来源：本项目自建虚构业务资料，不代表真实企业。",
    ]


def write_document(path, title, lines, table=None):
    if path.suffix == ".md":
        path.write_text("# " + lines[0] + "\n\n" + "\n\n".join(lines[1:]) + "\n")
    elif path.suffix == ".pdf":
        SimpleDocTemplate(str(path)).build([Paragraph(line, STYLE) for line in lines])
    elif path.suffix == ".docx":
        document = Document()
        document.add_heading(lines[0], 0)
        for line in lines[1:]:
            document.add_paragraph(line)
        document.save(path)
    else:
        book = Workbook()
        sheet = book.active
        sheet.title = "规定"
        sheet.append(["事项", "约束", "规定"])
        for row in table or []:
            sheet.append(row)
        notes = book.create_sheet("说明")
        notes.append(["段落"])
        for line in lines:
            notes.append([line])
        book.save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(key, path, title, family, genre):
    return {
        "id": key,
        "source_key": key,
        "tenant_id": "xingqiao",
        "owner_id": "xq-admin",
        "path": str(path),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "title": title,
        "family": family,
        "split": "development",
        "tenant_public": True,
        "groups": [],
        "effective_from": "2026-09-01",
        "effective_to": None,
        "version": "s3-conflicts-v1",
        "provenance": "Synthetic conflict material authored for S5; not derived from any real policy.",
        "genre": genre,
    }


def build():
    assert SOURCE.joinpath("freeze.json").exists(), "v1 must be frozen before extending it"
    assert_disjoint()
    if ROOT.exists():
        shutil.rmtree(ROOT)
    shutil.copytree(SOURCE, ROOT)
    (ROOT / "freeze.json").unlink()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    questions = json.loads((ROOT / "questions.json").read_text())
    for row in manifest:
        row["path"] = row["path"].replace("fixtures/s3/", "fixtures/s3-v2/", 1)
    seen = {q["family"] for q in questions}
    number = 201

    for index, (key, subject, field, left, right) in enumerate(CONFLICTS):
        family = f"conflict-{key}"
        assert family not in seen, family
        suffixes = [".md", ".pdf", ".docx", ".xlsx"]
        titles = [f"{subject}规定（支持团队版）", f"{subject}规定（运维团队版）"]
        ids = []
        for side, (value, title) in enumerate(zip([left, right], titles, strict=True)):
            path = ROOT / "documents" / f"conflict-{key}-{'ab'[side]}{suffixes[(index + side) % 4]}"
            lines = paragraphs(
                title, subject, field, value, ["支持团队", "运维团队"][side], titles[1 - side]
            )
            write_document(path, title, lines, table=[[subject, field, value]])
            manifest.append(entry(path.stem, path, title, family, "冲突对照"))
            ids.append(path.stem)
        questions.append(
            {
                "id": f"Q{number:03}",
                "user": "xq-support",
                "question": f"星桥{subject}到底是多少？两处说法一致吗？",
                "expected": "conflict",
                "facts": [left, right],
                "source_ids": ids,
                "split": "development",
                "family": family,
                "kind": "conflict",
                "review": {
                    "status": "requires_human_review",
                    "author": "Codex",
                    "source_check": "authored pair: same subject, same window, no precedence",
                },
            }
        )
        number += 1

    for index, (pattern, key, subject, pairs) in enumerate(LOOKALIKES):
        family = f"lookalike-{key}"
        assert family not in seen, family
        suffix = [".md", ".docx", ".pdf", ".xlsx"][index % 4]
        path = ROOT / "documents" / f"lookalike-{key}{suffix}"
        title = f"{subject}说明"
        if pattern == "metric":
            lines = [
                f"{title}（2026-09 版）",
                f"本页列出星桥软件{subject}的各项参数，自 2026 年 9 月 1 日执行。",
                f"{subject}的{pairs[0][0]}为 {pairs[0][1]}。",
                f"{subject}的{pairs[1][0]}为 {pairs[1][1]}。",
                "以上两项是不同指标，各自独立约束，不互相替代，也不存在矛盾。",
                "来源：本项目自建虚构业务资料，不代表真实企业。",
            ]
            question = f"星桥{subject}的{pairs[0][0]}和{pairs[1][0]}分别是多少？"
        elif pattern == "object":
            lines = [
                f"{title}（2026-09 版）",
                f"本页按套餐分别规定{subject}，自 2026 年 9 月 1 日执行。",
                f"{pairs[0][0]}的{subject}为 {pairs[0][1]}。",
                f"{pairs[1][0]}的{subject}为 {pairs[1][1]}。",
                "两个数值针对不同套餐，适用对象不同，不构成冲突。",
                "来源：本项目自建虚构业务资料，不代表真实企业。",
            ]
            question = f"星桥{pairs[1][0]}的{subject}是多少？"
        else:
            lines = [
                f"{title}（含版本沿革）",
                f"本页记录星桥软件{subject}的历次规定，并标明各自生效区间。",
                f"{pairs[0][0]}，{subject}为 {pairs[0][1]}。该口径已于 2026 年 8 月 31 日失效。",
                f"{pairs[1][0]}，{subject}为 {pairs[1][1]}，为当前有效口径。",
                "新旧规定有明确的生效日期先后，按提问所属日期选择适用版本，不属于无法裁决的冲突。",
                "来源：本项目自建虚构业务资料，不代表真实企业。",
            ]
            question = f"星桥现在的{subject}是多久？"
        write_document(path, title, lines, table=[[subject, a, b] for a, b in pairs])
        manifest.append(entry(path.stem, path, title, family, "易误判反例"))
        questions.append(
            {
                "id": f"Q{number:03}",
                "user": "xq-support",
                "question": question,
                "expected": "answered",
                "facts": [pairs[1][1]] if pattern != "metric" else [pairs[0][1], pairs[1][1]],
                "source_ids": [path.stem],
                "split": "development",
                "family": family,
                "kind": "conflict_lookalike",
                "review": {
                    "status": "requires_human_review",
                    "author": "Codex",
                    "source_check": f"authored non-conflict pattern: {pattern}",
                },
            }
        )
        number += 1

    for index, (pattern, key, subject, left, lv, right, rv) in enumerate(CROSS_LOOKALIKES):
        family = f"cross-{key}"
        assert family not in seen, family
        suffixes = [".md", ".docx", ".pdf", ".xlsx"]
        ids = []
        for side, (scope, value) in enumerate([(left, lv), (right, rv)]):
            path = ROOT / "documents" / f"cross-{key}-{'ab'[side]}{suffixes[(index + side) % 4]}"
            title = f"{scope}{subject}说明"
            write_document(
                path,
                title,
                cross_paragraphs(title, subject, scope, value, [right, left][side]),
                table=[[subject, scope, value]],
            )
            manifest.append(entry(path.stem, path, title, family, "跨文档反例"))
            ids.append(path.stem)
        questions.append(
            {
                "id": f"Q{number:03}",
                "user": "xq-support",
                "question": f"星桥{right}的{subject}是多少？",
                "expected": "answered",
                "facts": [rv],
                "source_ids": [ids[1]],
                "split": "development",
                "family": family,
                "kind": "conflict_lookalike_cross",
                "review": {
                    "status": "requires_human_review",
                    "author": "Codex",
                    "source_check": f"authored cross-document non-conflict: {pattern}",
                },
            }
        )
        number += 1

    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n")
    split = json.loads((ROOT / "split.json").read_text())
    development = {q["family"] for q in questions if q["split"] == "development"}
    holdout = {q["family"] for q in questions if q["split"] == "holdout"}
    assert not development & holdout, "Family leak across splits"
    split.update(
        version="s3-v2-conflicts",
        documents=len(manifest),
        questions=len(questions),
        development=sum(q["split"] == "development" for q in questions),
        holdout=sum(q["split"] == "holdout" for q in questions),
        development_topics=sorted(development),
        holdout_topics=sorted(holdout),
        derived_from="fixtures/s3 (frozen v1), copied byte-identical; only development families added",
        added="12 conflicts, 9 single-document look-alikes, 6 cross-document look-alikes; conflict recall no longer rests on one question",
    )
    (ROOT / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n")
    print(
        f"s3-v2: {len(manifest)} documents, {len(questions)} questions "
        f"(+{len(CONFLICTS)} conflicts, +{len(LOOKALIKES)} single-doc, "
        f"+{len(CROSS_LOOKALIKES)} cross-doc look-alikes); human review still pending"
    )


if __name__ == "__main__":
    build()
