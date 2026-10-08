"""Generate the observation-dependent ("dynamic") benchmark from a structured fictional world.

Every answer is computed from the world data, never written by hand, so documents and
gold facts cannot drift apart. Dev and Test are separate tenants with disjoint entities:
tuning on Dev cannot reveal a Test fact. Whether a task really needs a dynamic step is
not decided here; `--measure` records one-shot retrieval gold recall for every task.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_package import annotate  # noqa: E402
from scripts.benchmark_package_scoring import task_review_packet  # noqa: E402
from scripts.benchmark_runtime import atomic_json  # noqa: E402

DISCLAIMER = "本项目自建的虚构资料，仅用于开发与演示，不代表任何真实公司的制度。"
CATEGORY = {"bridge": "latent_link", "bridge_deep": "latent_link", "alias": "query_recovery",
            "conditional": "conditional_planning", "enumeration": "enumeration", "event_version": "temporal_version",
            "conflict": "comparison", "control": "efficiency_stopping", "unanswerable": "null_insufficient"}
MAX_STEPS = {"bridge": 6, "bridge_deep": 10, "alias": 6, "conditional": 8, "enumeration": 10,
             "event_version": 8, "conflict": 6, "control": 4, "unanswerable": 8}
CREATURES = ["鹊桥", "青鸾", "玄武", "白泽", "毕方", "重明", "当康", "英招", "陆吾", "帝江", "夔牛", "精卫", "獬豸",
             "乘黄", "驺吾", "鸾鸟", "旋龟", "文鳐", "狻猊", "貔貅", "螭吻", "蒲牢", "朱厌", "鵸鵌"]
FUNCTIONS = [("网关", "统一接入与鉴权"), ("调度", "任务编排与定时调度"), ("结算", "订单结算与对账"),
             ("计费", "用量计费"), ("风控", "交易风险识别"), ("消息", "站内消息投递"), ("账务", "总账记账"),
             ("搜索", "商品检索"), ("报表", "经营报表生成"), ("库存", "库存同步"), ("质检", "来料质检记录"),
             ("排产", "生产排程"), ("门户", "供应商门户"), ("对象存储", "附件存储")]
SURNAMES = list("林沈顾陆苏韩邵谢程罗贺梁宋唐许姚卢钱秦尹孟")
GIVEN = ["舟", "砚", "朗", "岚", "澈", "棠", "珩", "芮", "越", "霁", "沅", "知秋", "以宁", "若川", "思远", "清和",
         "听澜", "景行", "望舒", "则安", "南星", "言蹊", "书禾", "子衿"]
TEAMS = ["平台基础组", "交易系统组", "数据平台组", "供应链系统组", "安全工程组"]
DAYS = ["周一", "周二", "周三", "周四", "周五"]
SYMPTOMS = ["大量请求超时", "消息积压", "间歇性 502 错误", "数据延迟入库", "连接池耗尽", "定时任务漏跑"]
CAUSES = ["上游批量任务未限流", "配置推送遗漏了一个可用区", "证书过期后未自动续期", "慢查询占满数据库连接",
          "扩容脚本误删了健康检查", "依赖库升级引入了重试风暴"]
FORMS = [("跨仓调拨申请单", "小蓝单"), ("紧急采购申请单", "红头单"), ("样品出库单", "白条"), ("外协加工单", "外发单"),
         ("报废处置单", "黑单"), ("临时用工申请单", "短工单")]
DIRECTORS = ["质量总监", "供应链总监", "财务总监", "运营副总裁"]
SITES = ["一厂区", "二厂区", "三厂区", "北仓", "南仓", "东仓"]
POLICY_ATTRS = [("外包人员门禁管理办法", "外包人员门禁卡有效期", "天"), ("差旅住宿管理办法", "一线城市住宿标准上限", "元/晚")]
CONFLICT_ATTRS = [("夜间叉车作业截止时间", ["22:00", "21:00"]), ("危化品库房单次进入人数上限", ["4 人", "3 人"]),
                  ("高温天气室外作业暂停时间", ["13:00", "12:30"])]


def world(seed, tenant, *, services, people, incidents, forms, sites, enum_rules, conflicts):
    rng = random.Random(seed)
    names = rng.sample([s + g for s in SURNAMES for g in GIVEN], people)
    exts = rng.sample(range(6100, 6999), people)
    persons = [{"name": n, "ext": str(e), "team": rng.choice(TEAMS), "day": rng.choice(DAYS),
                "restricted": False} for n, e in zip(names, exts)]
    for i, p in enumerate(persons):
        p["deputy"] = persons[(i + 1) % len(persons)]["name"]
    # The last two persons' profiles are HR-only: chains through them must abstain.
    for p in persons[-2:]:
        p["restricted"] = True
    creatures = rng.sample(CREATURES, services)
    functions = rng.sample(FUNCTIONS, services)
    svcs = []
    for i, (c, (kind, purpose)) in enumerate(zip(creatures, functions)):
        owner = persons[i % (people - 2)] if i < services - 2 else persons[-1 - (i - services + 2)]
        svcs.append({"name": c + kind, "code": f"{tenant.upper()}-SVC-{11 + i}", "purpose": purpose,
                     "team": owner["team"], "owner": owner["name"], "timeout": rng.choice([3, 5, 6, 8, 10, 12, 15]),
                     "resp": rng.choice([5, 10, 15, 20]), "hc": rng.choice([10, 15, 30])})
    # One documented service never registered an owner (the chain breaks honestly).
    svcs[services - 3]["owner"] = None
    incs = []
    for i, number in enumerate(rng.sample(range(100, 999), incidents)):
        svc = svcs[i % services]
        incs.append({"id": f"{tenant.upper()}-INC-{number}", "service": svc["name"],
                     "date": f"2026-0{rng.randint(3, 9)}-{rng.randint(10, 28)}", "symptom": rng.choice(SYMPTOMS),
                     "cause": rng.choice(CAUSES), "minutes": rng.randint(18, 95)})
    ids = Counter(x["id"] for x in incs)
    assert max(ids.values()) == 1, "incident ids must be unique"
    form_rows = []
    for (official, alias) in rng.sample(FORMS, forms):
        form_rows.append({"official": official, "alias": alias, "code": f"F-{rng.randint(10, 99)}",
                          "amount": rng.choice([3000, 5000, 8000, 20000, 50000]), "approver": rng.choice(DIRECTORS)})
    wh = []
    for site in rng.sample(SITES, sites):
        wh.append({"site": site, "rate": round(rng.choice([1.6, 2.2, 2.8, 3.4, 3.9, 4.6]), 1),
                   "supervisor": rng.choice(names[:people - 2])})
    if all(w["rate"] > 3.0 for w in wh) or all(w["rate"] <= 3.0 for w in wh):
        wh[0]["rate"], wh[1]["rate"] = 2.4, 3.7
    rules = []
    for i in range(enum_rules):
        covered = rng.sample([s for s in svcs if s["owner"] and not person(persons, s["owner"])["restricted"]],
                             rng.choice([2, 3]))
        rules.append({"id": f"R-{rng.randint(10, 99)}{i}", "services": [s["name"] for s in covered],
                      "window": rng.choice(["每月最后一个周五 18:00 起 48 小时", "季度结账前 3 个工作日"])})
    pols = []
    for title, attr, unit in POLICY_ATTRS:
        values = rng.sample([30, 45, 60, 90] if unit == "天" else [450, 500, 550, 600, 650], 3)
        pols.append({"title": title, "attr": attr, "unit": unit, "values": values,
                     "dates": ["2025-07-01", "2026-02-01", "2026-06-15"],
                     "notices": [f"{tenant.upper()}-AN-{rng.randint(10, 49)}", f"{tenant.upper()}-AN-{rng.randint(50, 99)}"]})
    confs = []
    for attr, (site_value, notice_value) in rng.sample(CONFLICT_ATTRS, conflicts):
        site = rng.choice(SITES)
        confs.append({"attr": attr, "site": site, "site_value": site_value, "notice_value": notice_value,
                      "notice": f"安全通告第 {rng.randint(3, 19)} 号"})
    return {"persons": persons, "services": svcs, "incidents": incs, "forms": form_rows, "warehouses": wh,
            "rules": rules, "policies": pols, "conflicts": confs, "director": rng.choice(names[:people - 2])}


def person(persons, name):
    return next(p for p in persons if p["name"] == name)


def documents(w, tenant, prefix):
    docs = []

    def add(key, title, body, *, groups=(), versions=None):
        docs.append({"document_id": f"{prefix}-{key}", "title": title, "groups": list(groups),
                     "tenant_public": not groups, "body": body, "versions": versions})

    for s in w["services"]:
        owner = f"服务负责人：{s['owner']}。" if s["owner"] else "服务负责人：交接中，暂未登记。"
        add(f"svc-{s['code'].lower()}", f"{s['name']}运维手册", (
            f"## 基本信息\n{s['name']}（服务编号 {s['code']}）由{s['team']}维护，主要负责{s['purpose']}。{owner}\n"
            f"日常变更需提前一个工作日在变更平台登记。\n\n## 运行参数\n单个请求超时阈值为 {s['timeout']} 秒，"
            f"健康检查间隔 {s['hc']} 秒。\n\n## 告警处理\n告警级别为 P1 时，值班人员应在 {s['resp']} 分钟内响应。"))
    for p in w["persons"]:
        add(f"staff-{hashlib.sha1(p['name'].encode()).hexdigest()[:6]}", f"员工名片：{p['name']}", (
            f"## 联系方式\n所属团队：{p['team']}。办公分机：{p['ext']}。\n\n## 值班与代班\n"
            f"每{p['day']}参与系统值班。休假或外出期间由{p['deputy']}代班。"),
            groups=("hr",) if p["restricted"] else ())
    for i in w["incidents"]:
        add(f"inc-{i['id'].lower()}", f"事故 {i['id']} 复盘", (
            f"## 概述\n{i['date']}，{i['service']}出现{i['symptom']}，持续约 {i['minutes']} 分钟。\n\n"
            f"## 根因\n{i['cause']}。\n\n## 改进\n补充监控告警，并在下次评审中复核处置流程。"))
    glossary = "\n".join(f"- “{f['alias']}”：即{f['official']}，因纸质版颜色或习惯得名。" for f in w["forms"])
    add("glossary", "内部常用俗称对照表", f"## 单据俗称\n{glossary}\n\n## 说明\n正式流程、审批与系统中均使用正式名称。")
    for f in w["forms"]:
        add(f"form-{f['code'].lower()}", f"{f['official']}管理规定", (
            f"## 适用范围\n本规定适用于{f['official']}（编号 {f['code']}）。\n\n## 审批\n"
            f"单笔金额不超过 {f['amount']} 元的，由部门负责人审批；超过 {f['amount']} 元的，须由{f['approver']}审批。"))
    rows = "\n".join(f"| {x['site']} | {x['rate']}% |" for x in w["warehouses"])
    add("quality-q3", "2026 年第三季度仓储质量报表", f"## 退货率\n| 仓库 | 季度退货率 |\n| --- | --- |\n{rows}")
    add("return-escalation", "退货率升级处理规则", (
        f"## 规则\n季度退货率超过 3.0% 的仓库须启动升级处理，由质量总监{w['director']}牵头复盘。\n"
        "季度退货率不超过 3.0% 的，不启动升级处理，由该仓仓库主管自行跟进。"))
    sup = "\n".join(f"| {x['site']} | {x['supervisor']} |" for x in w["warehouses"])
    add("warehouse-directory", "仓库主管名录", f"## 主管\n| 仓库 | 仓库主管 |\n| --- | --- |\n{sup}")
    for r in w["rules"]:
        add(f"freeze-{r['id'].lower()}", f"变更冻结规则 {r['id']}", (
            f"## 适用服务\n本规则适用于：{'、'.join(r['services'])}。\n\n## 冻结窗口\n{r['window']}内禁止生产变更，"
            "紧急修复须经值班经理批准。"))
    for k, p in enumerate(w["policies"]):
        versions = []
        for n, (value, date) in enumerate(zip(p["values"], p["dates"])):
            versions.append({"effective_from": f"{date}T00:00:00Z", "body": (
                f"## 第三条\n{p['attr']}为 {value} {p['unit']}。\n\n## 附则\n本版自 {date} 起施行。")})
        add(f"policy-{k}", p["title"], None, versions=versions)
        for date, notice in zip(p["dates"][1:], p["notices"]):
            add(f"notice-{notice.lower()}", f"行政公告 {notice}", (
                f"## 公告\n经研究，{p['title']}相关条款自 {date} 起修订，修订后的具体标准以制度正文为准。"))
    for k, c in enumerate(w["conflicts"]):
        add(f"site-rule-{k}", f"{c['site']}作业规范", (
            f"## 作业要求\n{c['site']}{c['attr']}为 {c['site_value']}。\n\n## 冲突处理\n"
            "本规范与公司安全通告不一致时，以安全通告为准。"))
        add(f"safety-{k}", c["notice"], (
            f"## 通告\n各厂区、仓库{c['attr']}统一调整为 {c['notice_value']}。本通告优先于各厂区作业规范。"))
    return docs


def tasks(w, prefix, user, counts, tid):
    out = []
    by_name = {s["name"]: s for s in w["services"]}
    staff = {p["name"]: p for p in w["persons"]}

    def doc(kind, key):
        return f"{prefix}-{kind}-{key}"

    def sdoc(name):
        return doc("svc", by_name[name]["code"].lower())

    def pdoc(name):
        return doc("staff", hashlib.sha1(name.encode()).hexdigest()[:6])

    def task(kind, goal, status, facts, docs, **extra):
        out.append({"id": f"{tid}{len(out) + 1:02d}", "dynamic_type": kind, "category": CATEGORY[kind], "user": user,
                    "goal": goal, "expected_status": status, "facts": [f["id"] for f in facts],
                    "fact_matchers": facts, "forbidden": extra.pop("forbidden", []), "scenario_events": [],
                    "expected_evidence_documents": docs, "max_steps": MAX_STEPS[kind],
                    "min_source_documents": len(docs), **extra})

    def fact(fid, *aliases):
        return {"id": fid, "aliases": list(dict.fromkeys(aliases))}

    ok = [i for i in w["incidents"] if by_name[i["service"]]["owner"]
          and not staff[by_name[i["service"]]["owner"]]["restricted"]]
    bridge_pool = ok[:counts["bridge"]]
    for n, i in enumerate(bridge_pool):
        s = by_name[i["service"]]
        if n % 2 == 0:
            task("bridge", f"事故 {i['id']} 中出故障的那个服务，单个请求的超时阈值是多少秒？", "answered",
                 [fact("timeout", f"{s['timeout']} 秒", f"{s['timeout']}秒")], [doc("inc", i["id"].lower()), sdoc(s["name"])],
                 conditional_transition={"observe_document": doc("inc", i["id"].lower()), "then_document": sdoc(s["name"])})
        else:
            o = staff[s["owner"]]
            task("bridge", f"事故 {i['id']} 中出故障的服务由谁负责？此人的办公分机是多少？", "answered",
                 [fact("owner", o["name"]), fact("ext", o["ext"])], [doc("inc", i["id"].lower()), sdoc(s["name"]), pdoc(o["name"])],
                 conditional_transition={"observe_document": doc("inc", i["id"].lower()), "then_document": sdoc(s["name"])})
    deep_pool = [i for i in ok if staff[staff[by_name[i["service"]]["owner"]]["deputy"]]["restricted"] is False]
    deep_pool = [i for i in deep_pool if i not in bridge_pool][:counts["bridge_deep"]] or deep_pool[:counts["bridge_deep"]]
    for i in deep_pool:
        s = by_name[i["service"]]
        o = staff[s["owner"]]
        d = staff[o["deputy"]]
        task("bridge_deep", f"事故 {i['id']} 中出故障的服务，其负责人休假时由谁代班？代班人的办公分机是多少？", "answered",
             [fact("deputy", d["name"]), fact("ext", d["ext"])],
             [doc("inc", i["id"].lower()), sdoc(s["name"]), pdoc(o["name"]), pdoc(d["name"])],
             conditional_transition={"observe_document": doc("inc", i["id"].lower()), "then_document": sdoc(s["name"])})
    for f in w["forms"][:counts["alias"]]:
        task("alias", f"{f['alias']}的金额超过多少元时需要更高级别审批？由谁审批？", "answered",
             [fact("amount", f"{f['amount']} 元", f"{f['amount']}元", f"{f['amount']:,} 元"), fact("approver", f["approver"])],
             [f"{prefix}-glossary", doc("form", f["code"].lower())],
             conditional_transition={"observe_document": f"{prefix}-glossary", "then_document": doc("form", f["code"].lower())})
    for x in w["warehouses"][:counts["conditional"]]:
        high = x["rate"] > 3.0
        lead = w["director"] if high else x["supervisor"]
        task("conditional", f"{x['site']}第三季度退货率是否需要启动升级处理？应由谁牵头或跟进？", "answered",
             [fact("rate", f"{x['rate']}%"), fact("decision", "需要" if high else "不需要", "需启动" if high else "不启动"),
              fact("lead", lead)],
             [doc("quality", "q3"), doc("return", "escalation")] + ([] if high else [doc("warehouse", "directory")]),
             conditional_transition={"observe_document": doc("quality", "q3"),
                                     "then_document": doc("return", "escalation") if high else doc("warehouse", "directory")})
    for r in w["rules"][:counts["enumeration"]]:
        owners = [by_name[n]["owner"] for n in r["services"]]
        task("enumeration", f"变更冻结规则 {r['id']} 覆盖的服务分别由谁负责？", "answered",
             [fact(f"owner{k + 1}", o) for k, o in enumerate(owners)] + [fact(f"svc{k + 1}", n) for k, n in enumerate(r["services"])],
             [doc("freeze", r["id"].lower())] + [sdoc(n) for n in r["services"]],
             conditional_transition={"observe_document": doc("freeze", r["id"].lower()), "then_document": sdoc(r["services"][0])})
    events = []
    for k, p in enumerate(w["policies"]):
        for n, notice in enumerate(p["notices"]):
            events.append((k, p, n, notice, "before"))
            events.append((k, p, n, notice, "after"))
    for k, p, n, notice, side in events[:counts["event_version"]]:
        value = p["values"][n] if side == "before" else p["values"][n + 1]
        phrase = "生效之前" if side == "before" else "生效之后、下一次修订之前" if n == 0 else "生效之后"
        task("event_version", f"行政公告 {notice} 宣布的那次修订{phrase}，{p['attr']}是多少？", "answered",
             [fact("value", f"{value} {p['unit']}", f"{value}{p['unit']}")],
             [doc("notice", notice.lower()), f"{prefix}-policy-{k}"],
             conditional_transition={"observe_document": doc("notice", notice.lower()), "then_document": f"{prefix}-policy-{k}"})
    for k, c in enumerate(w["conflicts"][:counts["conflict"]]):
        task("conflict", f"{c['site']}的{c['attr']}，作业规范与安全通告不一致时以哪份为准？具体是多少？", "answered",
             [fact("value", c["notice_value"], c["notice_value"].replace(" ", "")), fact("source", c["notice"])],
             [f"{prefix}-site-rule-{k}", f"{prefix}-safety-{k}"])
    controls = [s for s in w["services"] if s["owner"]][:counts["control"]]
    for n, s in enumerate(controls):
        if n % 2 == 0:
            task("control", f"{s['name']}的单个请求超时阈值是多少秒？", "answered",
                 [fact("timeout", f"{s['timeout']} 秒", f"{s['timeout']}秒")], [sdoc(s["name"])],
                 allowed_tools_after_sufficient=["retrieve_evidence", "verify_chunk_access"])
        else:
            task("control", f"{s['name']}的服务负责人是谁？", "answered", [fact("owner", s["owner"])], [sdoc(s["name"])],
                 allowed_tools_after_sufficient=["retrieve_evidence", "verify_chunk_access"])
    gaps = []
    no_owner = [i for i in w["incidents"] if not by_name[i["service"]]["owner"]]
    hidden = [i for i in w["incidents"] if by_name[i["service"]]["owner"] and staff[by_name[i["service"]]["owner"]]["restricted"]]
    for i in no_owner[:1]:
        gaps.append((i, None))
    for i in hidden:
        gaps.append((i, staff[by_name[i["service"]]["owner"]]))
    for i, hidden_person in gaps[:counts["unanswerable"]]:
        task("unanswerable", f"事故 {i['id']} 中出故障的服务，其负责人的办公分机是多少？", "insufficient_evidence", [],
             [], forbidden=[hidden_person["ext"]] if hidden_person else [])
    got = Counter(t["dynamic_type"] for t in out)
    short = {k: (got[k], v) for k, v in counts.items() if got[k] != v}
    if short:
        raise SystemExit(f"world too small for requested counts: {short}")
    return out


SPLITS = {
    "dev": {"seed": 20261007, "tenant": "qinghe", "name": "青禾智造", "tid": "DD",
            "world": dict(services=10, people=14, incidents=18, forms=4, sites=4, enum_rules=3, conflicts=2),
            "counts": dict(bridge=6, bridge_deep=3, alias=4, conditional=4, enumeration=3, event_version=3,
                           conflict=2, control=3, unanswerable=2)},
    "test": {"seed": 20261008, "tenant": "yuanfan", "name": "远帆数科", "tid": "DT",
             "world": dict(services=14, people=20, incidents=26, forms=5, sites=5, enum_rules=4, conflicts=3),
             "counts": dict(bridge=8, bridge_deep=4, alias=5, conditional=5, enumeration=4, event_version=4,
                            conflict=3, control=4, unanswerable=3)},
}


def build(split):
    cfg = SPLITS[split]
    w = world(cfg["seed"], cfg["tenant"], **cfg["world"])
    fixture = ROOT / "fixtures/dynamic" / split
    prefix = cfg["tenant"][:2]
    docs = documents(w, cfg["tenant"], prefix)
    raw = tasks(w, prefix, f"{prefix}-eng", cfg["counts"], cfg["tid"])  # fail before writing anything
    # Only generated files live here; stale files from an earlier world must not be indexed.
    for stale in (fixture / "documents").glob("*.md"):
        stale.unlink()
    spec_docs = []
    for d in docs:
        versions = d["versions"] or [{"effective_from": "2026-01-01T00:00:00Z", "body": d["body"]}]
        rows = []
        for n, v in enumerate(versions):
            path = fixture / "documents" / (d["document_id"] + (f"-v{n + 1}" if len(versions) > 1 else "") + ".md")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {d['title']}\n\n{DISCLAIMER}\n\n{v['body']}\n")
            rows.append({"path": path.relative_to(ROOT).as_posix(), "effective_from": v["effective_from"]})
        spec_docs.append({"document_id": d["document_id"], "title": d["title"], "groups": d["groups"],
                          "tenant_public": d["tenant_public"], "versions": rows})
    eng, admin = f"{prefix}-eng", f"{prefix}-admin"
    spec = {"tenant": {"id": cfg["tenant"], "name": cfg["name"]},
            "users": [{"id": eng, "tenant_id": cfg["tenant"], "username": f"engineer@{cfg['tenant']}.dynamic",
                       "display_name": f"{cfg['name']} · 工程", "role": "member", "groups": ["engineering"]},
                      {"id": admin, "tenant_id": cfg["tenant"], "username": f"admin@{cfg['tenant']}.dynamic",
                       "display_name": f"{cfg['name']} · 管理者", "role": "admin", "groups": ["engineering", "hr", "finance"]}],
            "owner": admin, "documents": spec_docs}
    atomic_json(fixture / "documents.json", spec)
    atomic_json(fixture / "world.json", w)
    atomic_json(fixture / "tasks.json", {"version": f"dynamic-{split}-v1", "split": split, "tasks": raw,
                                         "purpose": "observation-dependent tasks; necessity measured separately"})
    by_id = {d["document_id"]: d for d in spec_docs}
    annotated = [annotate(t, by_id) for t in raw]
    suite_dir = ROOT / "benchmarks/enterprise_rag/v1/dynamic" / split
    v4 = json.loads((ROOT / ".runtime/benchmark-package/v1/dev-five-repairs-20261006-v4/tasks.json").read_text())
    suite = {"version": f"enterprise-rag-benchmark-v1-dynamic-{split}", "split": "dev" if split == "dev" else "dynamic_test",
             "status": "candidate_pending_GPT_annotation_review", "evaluation_protocol": "enterprise-benchmark-v1",
             "arms": ["rag", "workflow", "dynamic", "hybrid"], "trials": 1, "tasks": annotated, "safety_tasks": [],
             "categories": dict(Counter(t["category"] for t in annotated)),
             "dynamic_types": dict(Counter(t["dynamic_type"] for t in annotated)),
             "review_packet": (suite_dir / "review.json").relative_to(ROOT).as_posix(),
             "reviewed_packet": (suite_dir / "reviewed.json").relative_to(ROOT).as_posix(),
             "package_manifest": "benchmarks/enterprise_rag/v1/manifest.json",
             "shared_configuration": v4["shared_configuration"], "method_matrix": v4["method_matrix"],
             "scorer": "strict_facts_scope_citations_abstention_GPT_review",
             "execution_policy": "repeatable_development" if split == "dev" else "single_registered_matrix_per_freeze",
             "generator": "scripts/build_dynamic_benchmark.py"}
    atomic_json(suite_dir / "tasks.json", suite)
    atomic_json(suite_dir / "documents.json", spec)
    atomic_json(suite_dir / "review.json", task_review_packet(suite, spec, ROOT))
    print(split, len(annotated), "tasks", dict(Counter(t["dynamic_type"] for t in annotated)), len(spec_docs), "documents")


def measure(split):
    """One-shot retrieval of the question alone: how many gold documents it already finds."""
    from app.clients import Models, Search
    from app.config import settings
    from app.db import SessionLocal
    from app.models import User
    from app.retrieval import retrieve_authorized

    suite_dir = ROOT / "benchmarks/enterprise_rag/v1/dynamic" / split
    suite = json.loads((suite_dir / "tasks.json").read_text())
    for key, value in suite["shared_configuration"].items():
        setattr(settings(), key, value)
    rows = []
    with SessionLocal() as db:
        for t in suite["tasks"]:
            found = retrieve_authorized(db, db.get(User, t["user"]), t["goal"], cfg=settings(), models=Models(), search=Search())
            got = {e["document_id"] for e in found.evidence}
            gold = set(t["gold_documents"])
            rows.append({"task_id": t["id"], "dynamic_type": t["dynamic_type"], "gold": len(gold),
                         "one_shot_gold_found": len(gold & got), "missing": sorted(gold - got),
                         "needs_dynamic": bool(gold - got) if gold else None})
    summary = {k: f"{sum(r['needs_dynamic'] for r in rs)}/{len(rs)}" for k, rs in
               ((k, [r for r in rows if r["dynamic_type"] == k and r["needs_dynamic"] is not None])
                for k in dict.fromkeys(r["dynamic_type"] for r in rows)) if rs}
    out = {"split": split, "rule": "needs_dynamic = one-shot retrieval of the question misses a gold document",
           "summary_needs_dynamic": summary, "rows": rows}
    atomic_json(ROOT / f"artifacts/dynamic-benchmark-{split}-necessity.json", out)
    print(split, json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("build", "measure"))
    p.add_argument("--split", choices=tuple(SPLITS), required=True)
    args = p.parse_args()
    (build if args.command == "build" else measure)(args.split)
