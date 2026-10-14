"""Fictional Dev47 replay; never opens services or real benchmark artifacts."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.execution_budget import BudgetExceeded, ExecutionBudget, TaskCancelled
from scripts.research_review import digest
from scripts.run_retrieval_quality_dev import sha

try:
    runner = importlib.import_module("scripts.run_policy_ablation_replay_dev")
except ModuleNotFoundError:
    runner = None


@pytest.fixture(autouse=True)
def forbid_real_models(monkeypatch):
    import app.rerank
    import app.db
    import httpx

    def forbidden(*args, **kwargs):
        pytest.fail("Real model, database or network transport forbidden")

    monkeypatch.setattr(app.rerank, "_model", forbidden)
    monkeypatch.setattr(app.db, "SessionLocal", forbidden)
    monkeypatch.setattr(httpx, "get", forbidden)
    monkeypatch.setattr(httpx, "post", forbidden)
    monkeypatch.setattr(httpx.Client, "request", forbidden)


def test_registered_runner_exists():
    assert runner is not None, "Missing registered complete Dev47 source and answer replay"


@pytest.fixture
def replay(tmp_path, monkeypatch):
    assert runner is not None, "Runner missing"
    r = runner
    monkeypatch.setattr(r, "ROOT", tmp_path)
    base = tmp_path / ".runtime/evidence-first-20261011"
    base.mkdir(parents=True)
    stage3_runner = tmp_path / "scripts/run_bound_citation_dev.py"
    stage3_runner.parent.mkdir(parents=True)
    stage3_runner.write_text("# synthetic Stage3 runner after formatting\n")
    archived_stage3_runner = base / "citation-style-before-20261012-v1/scripts/run_bound_citation_dev.py"
    archived_stage3_runner.parent.mkdir(parents=True)
    archived_stage3_runner.write_text("# synthetic original Stage3 runner\n")
    tasks = [
        dict(
            id=f"T{i:02}"
            if i < 40
            else sorted(
                {
                    "U12",
                    "U22",
                    "SF-3e8219f58317",
                    "SF-56d3160a9c64",
                    "SF-59679085cc4a",
                    "SF-3ea856eeec7b",
                    "SF-3fb3599dcc99",
                }
            )[i - 40],
            goal=f"Where is fictional parcel Z{i} carrier phone?",
            user="lt-eng",
            required_facts=[{"id": "f", "text": "SECRET GOLD"}],
            gold_documents=["d"],
            gold_evidence=[],
            forbidden=[],
            expected_behavior="SECRET PATH",
        )
        for i in range(47)
    ]
    suite = {"split": "dev", "tasks": tasks}
    selection = {
        "split": "dev",
        "count": 47,
        "items": [
            dict(id=t["id"], query_sha256=sha(t["goal"].encode()), annotation_sha256=digest(t))
            for t in tasks
        ],
    }
    evidence = dict(
        chunk_id="c",
        document_id="d",
        version_id="v",
        source_sha256="a" * 64,
        title="SOURCE TITLE",
        text="SOURCE TEXT",
        metadata={},
        locator={},
        rank=1,
        parent_rank=1,
        retrieval_rank=1,
        bm25_rank=1,
        dense_rank=None,
        fusion_score=1,
        rerank_score=1,
        excluded_because=None,
        admitted=True,
        lane_type="global",
        lane_sha256=digest("global"),
    )
    telemetry = dict(
        context=[evidence],
        candidates=[evidence],
        admitted=[evidence],
        context_tokens=r.replay.evidence_tokens([evidence]),
    )
    corpus, index = (
        {"documents": []},
        {
            "index": "synthetic",
            "population": {"count": 0},
            "mapping_sha256": digest({}),
            "chunks": [],
        },
    )
    parent = {
        "identity": {
            "corpus": corpus,
            "corpus_sha256": digest(corpus),
            "source_index": index,
            "source_index_sha256": digest(index),
        }
    }
    baseline = {
        "rows": {"relevant_doc": [dict(task_id=t["id"], telemetry=deepcopy(telemetry)) for t in tasks]}
    }
    chosen = dict(
        version="guarded-selection-dev-v2",
        split="dev",
        subset=False,
        valid=True,
        status="complete",
        error=None,
        drift={"all": False},
        selected_task_ids=[t["id"] for t in tasks],
        chosen_profile="baseline",
        execution_errors={p: 0 for p in ("baseline", "rank_floor", "guarded")},
        identity={
            "parent_sha256": digest(parent),
            "corpus_sha256": digest(corpus),
            "index_sha256": digest(index),
            "registration": {
                "version": "guarded-selection-input-v2",
                "profiles": ["baseline", "rank_floor", "guarded"],
                "caps": {"limit": 8, "token_budget": 5000},
            },
        },
        rows={
            p: [dict(task_id=t["id"], error=None, telemetry=deepcopy(telemetry)) for t in tasks]
            for p in ("baseline", "rank_floor", "guarded")
        },
    )
    context = base / "contexts"
    context.mkdir()
    private_hashes = {}
    for t in tasks:
        p = context / (t["id"] + ".json")
        p.write_text(
            json.dumps(
                dict(
                    task_id=t["id"],
                    question=t["goal"],
                    contexts={"baseline": [evidence]},
                    pools={"baseline": [evidence]},
                )
            )
        )
        private_hashes[t["id"]] = sha(p.read_bytes())
    archive = base / "archive"
    archive.mkdir()
    historical = {}
    for i in range(205):
        p = archive / f"scripts/f{i:03}.py"
        p.parent.mkdir(exist_ok=True)
        p.write_text(f"# historical {i}\n")
        historical[str(p.relative_to(archive))] = sha(p.read_bytes())
    generation = {
        "version": "slot-generation-input-v2",
        "split": "dev",
        "subset": False,
        "comparison": "fallback",
        "protocols": ["shared_contract", "literal_fallback"],
        "selected_task_ids": [t["id"] for t in tasks],
        "code": historical,
        "private_task_bytes_sha256": private_hashes,
        "shared_context": {"profile": "baseline"},
        "settings_sha256": "a" * 64,
        "generator_identity": {"name": "qwen2.5:7b-instruct"},
        "review": {"pending_task_ids": [t["id"] for t in tasks[40:]]},
    }
    args = SimpleNamespace(
        **{k: str(base / (k + ".json")) for k in r.INPUT_ARGUMENTS},
        context_root=str(context),
        historical_code_root=str(archive),
        registration=str(base / "new-registration.json"),
        private_root=str(base / "new-private"),
        output=str(tmp_path / "output.json"),
    )
    args.generation_registration = str(
        tmp_path / "benchmarks/enterprise_rag/v1/dev/evidence-first-20261011/generation-input-v2.json"
    )
    args.generation_public = str(tmp_path / "artifacts/dev-slot-generation-20261011-v2.json")
    Path(args.generation_registration).parent.mkdir(parents=True)
    Path(args.generation_public).parent.mkdir(parents=True)
    selected = base / "selection.json"
    selected.write_text(json.dumps(selection))
    monkeypatch.setattr(r, "SELECTION", selected)
    for key, data in [
        ("suite", suite),
        ("parent", parent),
        ("baseline", baseline),
        ("selection_result", chosen),
        ("manifest", {}),
        ("generation_registration", generation),
    ]:
        Path(getattr(args, key)).write_text(json.dumps(data))
    Path(args.overlay).write_text('{"SECRET":"opaque annotations"}')
    generation_public = dict(
        run_id="prior",
        valid=True,
        status="complete",
        error=None,
        split="dev",
        subset=False,
        identity=generation,
        selected_task_ids=[t["id"] for t in tasks],
        private_raw_sha256="f" * 64,
    )
    Path(args.generation_public).write_text(json.dumps(generation_public))
    metadata = {
        "version": "bound-citation-review-metadata-v1",
        "source_run_id": "prior",
        "header": r.reference_header("prior"),
        "upstream_code": historical,
        "parts": [],
        "reviewed_semantic_sha256": "a" * 64,
    }
    expected, refs, reviewreg, anchorpath = r.lineage_paths(args, generation, "prior")
    for key, p in expected.items():
        if key.startswith("production_code_"):
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.resolve() not in [
            Path(args.generation_registration).resolve(),
            Path(args.generation_public).resolve(),
        ]:
            p.write_text("SECRET prior content")
    for p in refs.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("SECRET grades never JSON parsed")
    bindings = {}
    for key, p in expected.items():
        relative = (
            sorted(historical)[int(key.rsplit("_", 1)[1])]
            if key.startswith("production_code_")
            else None
        )
        actual = archive / relative if relative else p
        bindings[key] = {"path": str(p), "bytes_sha256": sha(actual.read_bytes())}
    reviewreg.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for number, count in enumerate((21, 21, 20), 1):
        parts.append(
            {
                "name": f"answer-part-{number}.json",
                "sha256": digest(["synthetic", number]),
                "items": count,
                "reviewer": f"GPT-6 / Codex blind fallback answer review {number}",
                "reviewer_type": "model",
                "model": "GPT-6 / Codex",
                "exact_serving_variant": None,
                "human": False,
                "independent_fresh_context": True,
                "independent_of_implementation_context": True,
                "model_family_independent_of_generator": True,
                "generator_model_name": "qwen2.5:7b-instruct",
                "reviewer_identity_basis": "system GPT-6 Codex identity; exact serving variant unavailable",
                "independent_of_model_gold_audit": False,
                "tokens": None,
                "cost": None,
            }
        )
    metadata["parts"] = parts
    from scripts.run_external_answer_review import SYSTEM

    review = {
        "version": "slot-generation-review-input-v2",
        "split": "dev",
        "comparison": "fallback",
        "source_run_id": "prior",
        "protocols": ["shared_contract", "literal_fallback"],
        "candidate": "literal_fallback",
        "baseline": "shared_contract",
        "raw_sha256": "f" * 64,
        "suite_sha256": "a" * 64,
        "packet_sha256": "b" * 64,
        "mapping_sha256": "c" * 64,
        "rubric_sha256": digest(SYSTEM),
        "settings_sha256": generation["settings_sha256"],
        "production_code_sha256": historical,
        "instructions_sha256": digest(expected["instructions"].read_text()),
        "assembly_sha256": {expected["helper"].name: digest(expected["helper"].read_text())},
        "frozen_files": bindings,
        "files": parts,
        "positions": 94,
        "visible_answers": 62,
        "canonical_evidence_tasks": 47,
        "pending_task_ids": generation["review"]["pending_task_ids"],
        "strict_scoreable_tasks": 40,
        "deduplication": "identical projected fresh task, user-visible answer and evidence reviewed once",
        "presentation": "protocol names, costs, task-contract and tool traces hidden",
        "accuracy_basis": "named model review, not human labels or Core headline",
        "all_material_parts_required": True,
        "historical_answer_review_reused": False,
        "historical_evidence_review_reused": True,
        "retrieval_assessment": "exact projected task/evidence SHA reuse of frozen canonical v1 evidence grades",
        "review_cost_tokens": None,
        "review_cost_currency": None,
        "exact_serving_variant": None,
    }
    reviewreg.write_text(json.dumps(review))
    anchor = {
        "registration_path": str(reviewreg.resolve()),
        "registration_bytes_sha256": sha(reviewreg.read_bytes()),
    }
    anchorpath.write_text(json.dumps(anchor))
    metadata.update(
        frozen_files=bindings,
        anchor=anchor,
        reference_file_bytes_sha256={k: sha(p.read_bytes()) for k, p in refs.items()},
        review_registration_bytes_sha256=sha(reviewreg.read_bytes()),
        generation_registration_bytes_sha256=sha(Path(args.generation_registration).read_bytes()),
        raw_bytes_sha256=sha(expected["raw"].read_bytes()),
        suite_bytes_sha256=sha(expected["suite"].read_bytes()),
        packet_bytes_sha256=sha(expected["packet"].read_bytes()),
        runner_code_sha256=sha(archived_stage3_runner.read_bytes()),
        stage2_helper_bytes_sha256=sha(expected["helper"].read_bytes()),
    )
    Path(args.review_metadata).write_text(json.dumps(metadata))
    cfg = Settings(
        _env_file=None,
        answer_quality_enabled=False,
        source_facts_enabled=False,
        chat_model="synthetic-generator",
    )
    monkeypatch.setattr(r, "settings", lambda: cfg)
    monkeypatch.setattr(r, "code_fingerprints", lambda: {"changed-production.py": "b" * 64})
    monkeypatch.setattr(r, "extra_code_bindings", lambda: {"test.py": "c" * 64})
    models = {
        "policy": {"name": "synthetic-policy", "digest": "d" * 64},
        "embedding": {"name": "synthetic-embed", "digest": "e" * 64},
        "libraries": {"langgraph": "1.2.12"},
    }
    monkeypatch.setattr(r, "model_bindings", lambda config: deepcopy(models))
    validators = []
    monkeypatch.setattr(
        r.replay,
        "validate_parent",
        lambda *a: validators.append("parent")
        or [dict(task_id=t["id"], telemetry=deepcopy(telemetry)) for t in tasks],
    )
    monkeypatch.setattr(r, "validate_seed", lambda *a: validators.append("seed") or "relevant_doc")
    statements = []

    class DB:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql):
            statements.append(str(sql))

    import app.db
    import app.clients

    monkeypatch.setattr(app.db, "SessionLocal", DB)
    monkeypatch.setattr(
        app.clients,
        "Search",
        lambda **kw: SimpleNamespace(
            index=kw["index"],
            request=lambda *a, **k: {kw["index"]: {"mappings": {}}},
        ),
    )
    monkeypatch.setattr(
        r,
        "active_scope",
        lambda db: ({"lt-eng": SimpleNamespace(tenant_id="tenant")}, [], deepcopy(corpus)),
    )
    monkeypatch.setattr(r, "index_ledger", lambda *a: deepcopy(index))
    monkeypatch.setattr(r, "index_population", lambda *a: deepcopy(index["population"]))

    def current_scope(db):
        if callable(getattr(db, "expire_all", None)):
            db.expire_all()
        return {"lt-eng": SimpleNamespace(tenant_id="tenant")}, deepcopy(corpus)

    monkeypatch.setattr(r, "current_scope", current_scope)
    monkeypatch.setattr(r.replay, "rebind_rows", lambda db, user, rows, corpus: deepcopy(rows))
    monkeypatch.setattr(r.replay, "rehydrate_task", lambda *a: {"candidates": [deepcopy(evidence)]})
    monkeypatch.setattr(r, "contract_for", lambda db, user, q: SimpleNamespace(question=q))
    monkeypatch.setattr(r, "make_source_models", lambda **kw: SimpleNamespace(events=[]))
    monkeypatch.setattr(r, "make_backend", lambda source: SimpleNamespace(events=[]))
    monkeypatch.setattr(r, "ranker", lambda *a: None)
    source_calls, gen_calls, evaluated = [], [], []

    def transport(db, user, contract, **kw):
        return SimpleNamespace(
            reauthorize=deepcopy,
            search=lambda *a: [],
            validate_scope=lambda *a: True,
            cfg=r.source_configuration(cfg),
        )

    monkeypatch.setattr(r, "authorized_search", transport)

    def component(contract, seed, candidates, **kw):
        source_calls.append((contract.question, deepcopy(seed), kw["order"]))
        budgets = {
            arm: ExecutionBudget(cfg, saved={"limits": r.SOURCE_LIMITS}).snapshot() for arm in r.ARMS
        }
        trace = {
            "status": "completed",
            "order": list(kw["order"]),
            "errors": [],
            "budgets": budgets,
            **{
                k: {arm: [] for arm in r.ARMS}
                for k in (
                    "physical_events",
                    "backend_events",
                    "retrieval_events",
                    "ranker_events",
                    "search_events",
                )
            },
        }
        return r.ComponentResult(
            {
                arm: {
                    "status": "seed_retained",
                    "evidence": deepcopy(seed),
                    "candidates": deepcopy(candidates),
                }
                for arm in r.ARMS
            },
            trace,
            {},
        )

    monkeypatch.setattr(r, "run_components", component)

    def generate(question, evidence, *, models, cfg, protocol, enforce_zero_embedding, **kw):
        gen_calls.append((question, deepcopy(evidence), protocol, enforce_zero_embedding))
        models.answer_budget = ExecutionBudget(cfg, saved={"limits": {"generation": 3}})
        return {"status": "answered", "claims": [], "citations": [], "verdict": None}, {
            "business_budget": models.answer_budget.snapshot()
        }

    monkeypatch.setattr(r, "generate_answer", generate)

    def annotation(*a):
        assert len(gen_calls) == 141
        evaluated.append("annotations")
        return {"task_records": {t["id"]: {} for t in tasks}}

    monkeypatch.setattr(r, "validate_annotation", annotation)

    def score(*a, **kw):
        assert len(gen_calls) == 141
        evaluated.append("score")
        return {}

    monkeypatch.setattr(r, "provisional_score", score)
    monkeypatch.setattr(
        r.supplement, "score_context", lambda *a: score() or {"telemetry": deepcopy(telemetry)}
    )
    monkeypatch.setattr(r, "rescore", lambda *a: score() or {})
    monkeypatch.setattr(r, "diagnose", lambda *a: score() or {})
    return SimpleNamespace(
        r=r,
        args=args,
        tasks=tasks,
        evidence=evidence,
        base=base,
        archive=archive,
        archived_stage3_runner=archived_stage3_runner,
        stage3_runner=stage3_runner,
        generation=generation,
        metadata=metadata,
        cfg=cfg,
        source_calls=source_calls,
        gen_calls=gen_calls,
        evaluated=evaluated,
        statements=statements,
        component=component,
    )


def register(f):
    reg = f.r.registration_for(f.args)
    Path(f.args.registration).write_text(json.dumps(reg))
    return reg


def test_full_47_three_sources_then_three_fresh_generators_and_post_runtime_labels(replay):
    f = replay
    reg = register(f)
    assert reg["phases"] == ["source", "generation"] and reg["generation_limits"]["ranker"] == 0
    out = f.r.execute(f.args, {"run_id": "full"})
    assert out["valid"] and out["status"] == "complete"
    assert len(f.source_calls) == 47 and len(f.gen_calls) == 141
    assert [c[2] for c in f.source_calls[:3]] == [
        f.r.ARMS,
        f.r.ARMS[1:] + f.r.ARMS[:1],
        f.r.ARMS[2:] + f.r.ARMS[:2],
    ]
    assert all(c[2:] == ("shared_contract", True) for c in f.gen_calls)
    assert f.statements == ["SET TRANSACTION READ ONLY"]
    assert f.evaluated[0] == "annotations"
    raw = json.loads((f.base / "new-private/full/raw.json").read_bytes())
    assert all(len(rows) == 47 for rows in raw["results"].values())
    assert all(
        row["strict_task_success"] is None and row["answer_correct"] is None
        for rows in raw["results"].values()
        for row in rows
    )
    text = Path(f.args.output).read_text()
    assert not any(s in text for s in ("SOURCE TEXT", "SOURCE TITLE", "SECRET", "fictional parcel"))
    assert out["strict_task_success"] is out["semantic_recall"] is None


@pytest.mark.parametrize(
    "change", ["question", "order", "text", "user", "settings", "model", "input", "code", "registration"]
)
def test_drift_preflight_blocks_all_dispatch_with_full_unknown_denominator(replay, monkeypatch, change):
    f = replay
    register(f)
    if change in ("question", "order", "text", "user"):
        p = Path(f.args.context_root) / (f.tasks[0]["id"] + ".json")
        value = json.loads(p.read_bytes())
        if change == "question":
            value["question"] = "Other question"
        elif change == "user":
            value["user"] = "other-user"
        elif change == "text":
            value["contexts"]["baseline"][0]["text"] = "Other bytes"
        else:
            value["contexts"]["baseline"].append(deepcopy(f.evidence))
        p.write_text(json.dumps(value))
    elif change == "settings":
        monkeypatch.setattr(f.r, "settings", lambda: f.cfg.model_copy(update={"chat_model": "changed"}))
    elif change == "model":
        monkeypatch.setattr(f.r, "model_bindings", lambda c: {"model": "changed"})
    elif change == "code":
        monkeypatch.setattr(f.r, "code_fingerprints", lambda: {"changed": "f" * 64})
    elif change == "registration":
        Path(f.args.registration).write_text("{}")
    else:
        Path(f.args.overlay).write_text("changed opaque overlay")
    out = f.r.execute(f.args, {"run_id": "drift"})
    assert not out["valid"] and not f.source_calls and not f.gen_calls and not f.evaluated
    assert all(
        len(rows) == 47 and all(row["generation_cost"] is None for row in rows)
        for rows in out["rows"].values()
    )


@pytest.mark.parametrize("change", ["missing", "mutated", "extra", "traversal"])
def test_archived_205_source_lineage_rejects_before_calls(replay, change):
    f = replay
    if change == "missing":
        (f.archive / "scripts/f000.py").unlink()
    elif change == "mutated":
        (f.archive / "scripts/f000.py").write_text("mutated")
    elif change == "extra":
        (f.archive / "extra.py").write_text("extra")
    else:
        f.generation["code"]["../escape.py"] = f.generation["code"].pop("scripts/f000.py")
        Path(f.args.generation_registration).write_text(json.dumps(f.generation))
    with pytest.raises(ValueError):
        register(f)
    assert not f.source_calls and not f.gen_calls


def test_archive_valid_despite_current_production_changes(replay):
    assert register(replay)["historical_code_sha256"] == digest(replay.generation["code"])


@pytest.mark.parametrize("change", ["missing", "tampered", "redirected"])
def test_original_attestation_runner_must_exist_at_canonical_path(replay, change):
    f = replay
    if change == "missing":
        f.archived_stage3_runner.unlink()
    elif change == "tampered":
        f.archived_stage3_runner.write_text("# altered original Stage3 runner\n")
    else:
        other = f.base / "equal-byte-runner.py"
        other.write_bytes(f.archived_stage3_runner.read_bytes())
        f.archived_stage3_runner.unlink()
        f.archived_stage3_runner.symlink_to(other)
    with pytest.raises(ValueError, match="Attestation runner"):
        register(f)
    assert not f.source_calls and not f.gen_calls


def test_original_runner_and_current_code_have_independent_byte_bindings(replay, monkeypatch):
    f = replay
    assert sha(f.archived_stage3_runner.read_bytes()) != sha(f.stage3_runner.read_bytes())
    monkeypatch.setattr(
        f.r,
        "code_fingerprints",
        lambda: {"scripts/run_bound_citation_dev.py": sha(f.stage3_runner.read_bytes())},
    )
    registration = register(f)
    assert registration["lineage_bytes_sha256"]["attestation_runner"] == sha(
        f.archived_stage3_runner.read_bytes()
    )
    assert registration["code"]["scripts/run_bound_citation_dev.py"] == sha(f.stage3_runner.read_bytes())
    inputs = f.r.load_inputs(f.args)
    registration_bytes = Path(f.args.registration).read_bytes()
    f.stage3_runner.write_text("# changed current Stage3 runner\n")
    with pytest.raises(ValueError, match="Current code drift"):
        f.r.snapshot_matches(f.args, inputs, registration, registration_bytes)


def test_snapshot_rejects_equal_byte_redirect_of_original_runner(replay):
    f = replay
    registration = register(f)
    inputs = f.r.load_inputs(f.args)
    registration_bytes = Path(f.args.registration).read_bytes()
    other = f.base / "equal-byte-runner.py"
    other.write_bytes(f.archived_stage3_runner.read_bytes())
    f.archived_stage3_runner.unlink()
    f.archived_stage3_runner.symlink_to(other)
    with pytest.raises(ValueError, match="Attestation runner"):
        f.r.snapshot_matches(f.args, inputs, registration, registration_bytes)


@pytest.mark.parametrize("interrupt", ["return", "keyboard"])
def test_source_interruption_keeps_partial_cost_and_stops_every_later_generation(
    replay, monkeypatch, interrupt
):
    f = replay
    register(f)

    def component(*a, **kw):
        result = f.component(*a, **kw)
        arm = kw["order"][0]
        result.trace["status"] = "interrupted"
        result.trace["budgets"][arm]["calls"]["search"] = {"attempted": 1, "succeeded": 0, "failed": 1}
        result.arms[arm].update(status="interrupted", evidence=None)
        for later in kw["order"][1:]:
            result.trace["budgets"][later] = None
            result.arms[later].update(status="not_run", evidence=None)
        if interrupt == "keyboard":
            exc = KeyboardInterrupt()
            exc.component_result = result
            raise exc
        return result

    monkeypatch.setattr(f.r, "run_components", component)
    out = f.r.execute(f.args, {"run_id": "interrupt"})
    assert out["status"] == "interrupted" and not out["valid"]
    assert len(f.source_calls) == 1 and not f.gen_calls and not f.evaluated
    raw = json.loads((f.base / "new-private/interrupt/raw.json").read_bytes())
    assert raw["results"]["no_supp"][0]["source_budget"]["calls"]["search"]["attempted"] == 1
    assert sum(row["generation_cost"] is None for rows in out["rows"].values() for row in rows) == 141


@pytest.mark.parametrize("role", ["ranker", "embedding", "policy", "search", "judge"])
def test_real_generation_budget_adapter_blocks_forbidden_roles_before_transport(role):
    assert runner is not None
    cfg = Settings(_env_file=None)
    models = runner.GenerationModels()
    models.answer_budget = ExecutionBudget(cfg, saved={"limits": {"generation": 3}})
    with pytest.raises(BudgetExceeded):
        with models.answer_budget.call(role):
            pytest.fail("Forbidden generation role reached transport")
    assert models.answer_budget.snapshot()["calls"][role]["attempted"] == 0


def test_public_nested_projection_rejects_raw_text_status_and_private_exception():
    assert runner is not None
    raw = {
        "task_id": "T00",
        "status": "SECRET arbitrary model status",
        "execution_error_kind": "SECRET exception",
        "latency_ms": float("nan"),
        "answer_payload": {"status": "SECRET"},
        "source_trace": {"SECRET": {"query": "SECRET"}},
        "source_budget": {
            "calls": {"generation": {"attempted": 1, "SECRET": "SECRET"}},
            "limits": {"generation": 0},
            "SECRET": "SECRET",
        },
        "generation_cost": {
            "physical_call_records": [{"role": "SECRET", "status": "SECRET", "prompt": "SECRET"}]
        },
        "source_cost": {"SECRET": {"text": "SECRET"}},
    }
    projected = runner.public_row(raw)
    assert "SECRET" not in json.dumps(projected)
    assert projected["latency_ms"] is None


@pytest.mark.parametrize("kind", ["header_type", "runner", "canonical_path", "anchor", "reference"])
def test_pure_attestation_has_exact_header_and_byte_paths(replay, kind):
    f = replay
    value = deepcopy(f.metadata)
    if kind == "header_type":
        value["header"]["valid"] = 1
    elif kind == "runner":
        value["runner_code_sha256"] = "0" * 64
    elif kind == "canonical_path":
        record = value["frozen_files"]["v1_instructions"]
        other = f.base / "equal-byte-alternative.md"
        other.write_bytes(Path(record["path"]).read_bytes())
        record["path"] = str(other)
    elif kind == "anchor":
        value["anchor"]["registration_bytes_sha256"] = "0" * 64
    else:
        value["reference_file_bytes_sha256"]["reviewer_1"] = "0" * 64
    Path(f.args.review_metadata).write_text(json.dumps(value))
    with pytest.raises(ValueError):
        register(f)
    assert not f.source_calls and not f.gen_calls


@pytest.mark.parametrize("role", ["ranker", "embedding"])
def test_unchanged_real_generator_receives_budget_adapter_before_actions(monkeypatch, role):
    from app.execution_budget import current_budget
    from scripts.run_external_answers_dev import generate_answer
    import app.task_contract

    models = runner.GenerationModels()
    reached = []

    def pipeline(*args, **kwargs):
        with current_budget().call(role):
            reached.append("transport")
        return {}, {}

    monkeypatch.setattr(app.task_contract, "contract_generate", pipeline)
    with pytest.raises(BudgetExceeded):
        generate_answer(
            "Fictional carrier phone?",
            [],
            models=models,
            cfg=Settings(_env_file=None),
            check_conflict=False,
            protocol="shared_contract",
            enforce_zero_embedding=True,
        )
    assert not reached and models.answer_budget.state["limits"][role] == 0


@pytest.mark.parametrize("change", ["acl", "index", "model"])
def test_post_runtime_drift_stops_remaining_positions_preserves_observed_cost(
    replay, monkeypatch, change
):
    f = replay
    register(f)
    if change == "acl":
        n = [0]

        def rebind(db, user, rows, corpus):
            n[0] += 1
            value = deepcopy(rows)
            if n[0] == 7:
                value[0]["text"] = "changed source bytes"
            return value

        monkeypatch.setattr(f.r.replay, "rebind_rows", rebind)
    elif change == "index":
        changed = [False]
        population = f.r.index_population
        original = f.r.generate_answer

        def generate(*args, **kwargs):
            result = original(*args, **kwargs)
            changed[0] = True
            return result

        monkeypatch.setattr(
            f.r,
            "index_population",
            lambda *a: {"count": 1} if changed[0] else population(*a),
        )
        monkeypatch.setattr(f.r, "generate_answer", generate)
    else:
        original = f.r.generate_answer

        def generate(*args, **kwargs):
            result = original(*args, **kwargs)
            monkeypatch.setattr(f.r, "model_bindings", lambda cfg: {"changed": True})
            return result

        monkeypatch.setattr(f.r, "generate_answer", generate)
    out = f.r.execute(f.args, {"run_id": "post-drift"})
    assert not out["valid"] and len(f.source_calls) == 1 and len(f.gen_calls) == 1
    assert not f.evaluated
    assert out["rows"]["no_supp"][0]["generation_cost"] is not None
    assert sum(r["generation_cost"] is None for rows in out["rows"].values() for r in rows) == 140


def test_generation_cancellation_keeps_prepaid_attempt_stops_later_tasks(replay, monkeypatch):
    f = replay
    register(f)

    def generate(question, evidence, *, models, cfg, **kw):
        f.gen_calls.append(question)
        models.answer_budget = ExecutionBudget(cfg, saved={"limits": {"generation": 3}})
        with models.answer_budget.call("generation"):
            models.events.append(
                dict(
                    role="generation",
                    status="failed",
                    wall_ms=4,
                    prompt_tokens=None,
                    completion_tokens=None,
                    server_ms=None,
                )
            )
            raise TaskCancelled("PRIVATE interruption")

    monkeypatch.setattr(f.r, "generate_answer", generate)
    out = f.r.execute(f.args, {"run_id": "gen-cancel"})
    assert out["status"] == "interrupted" and len(f.gen_calls) == len(f.source_calls) == 1
    budget = out["rows"]["no_supp"][0]["generation_cost"]["business_budget"]
    assert budget["calls"]["generation"]["attempted"] == budget["calls"]["generation"]["failed"] == 1
    assert "PRIVATE" not in Path(f.args.output).read_text()
    assert not f.evaluated


def test_source_cost_distinguishes_backend_fanout_reuse_and_unknown_tokens():
    cfg = Settings(_env_file=None)
    allocation = ExecutionBudget(cfg, saved={"limits": runner.SOURCE_LIMITS})
    allocation.state["calls"] = {
        role: {"attempted": 1, "succeeded": 1, "failed": 0} for role in ("search", "embedding", "ranker")
    }
    trace = {
        "budgets": {"original_supp": allocation.snapshot()},
        "physical_events": {
            "original_supp": [
                dict(
                    role="embedding",
                    status="ok",
                    wall_ms=3,
                    prompt_tokens=None,
                    completion_tokens=None,
                    server_ms=None,
                )
            ]
        },
        "backend_events": {
            "original_supp": [dict(status="ok", wall_ms=1), dict(status="ok", wall_ms=2)]
        },
        "retrieval_events": {"original_supp": []},
        "search_events": {"original_supp": [dict(status="ok", physical_search_calls=1)]},
        "ranker_events": {
            "original_supp": [
                dict(
                    status="ok",
                    physical_ranker_calls=1,
                    fresh_pairs=2,
                    cached_pairs=3,
                    cache_shared_across_arms=True,
                )
            ]
        },
    }
    cost = runner.public_source_cost(runner.source_cost(trace, "original_supp"))
    assert cost["transport_budget_matched"] is True
    assert cost["logical_search_calls"] == 1 and cost["physical_backend_calls"] == 2
    assert cost["fresh_ranker_pairs"] == 2 and cost["cached_ranker_pairs"] == 3
    assert cost["physical_cost_owner"] == "original_supp"
    assert cost["physical_events"][0]["prompt_tokens"] is None


def test_proxy_gain_loss_is_per_reference_separate_and_never_adoption():
    raw = {
        "proxy_diagnostics": {
            arm: [
                {
                    "task_id": "T00",
                    "original": {
                        "phases": {
                            "context": {
                                "total": 2,
                                "delivered": 1,
                                "matched_reference_ids": ["a" if arm == "no_supp" else "b"],
                            }
                        }
                    },
                    "revised": {
                        "phases": {
                            "context": {
                                "total": 2,
                                "delivered": 1,
                                "matched_reference_ids": ["a" if arm == "no_supp" else "b"],
                            }
                        }
                    },
                }
            ]
            for arm in runner.ARMS
        }
    }
    result = runner.proxy_summary(raw)
    assert result["paired"]["original_supp"]["original"][0]["gained_reference_sha256"] == [digest("b")]
    assert result["paired"]["original_supp"]["original"][0]["lost_reference_sha256"] == [digest("a")]
    assert result["semantic_recall"] is None and result["adoption"] is None


def test_ranker_scores_all_supplied_pairs_with_explicit_window(monkeypatch):
    import app.rerank

    monkeypatch.setattr(app.rerank, "_model", lambda: pytest.fail("Real model loading forbidden"))
    assert runner is not None
    calls = []

    def ranked(question, rows, **kw):
        calls.append((question, len(rows), kw))
        return deepcopy(rows)

    monkeypatch.setattr(runner, "rerank", ranked, raising=False)
    rows = [dict(chunk_id=str(i), text="Fictional source") for i in range(58)]
    assert runner.ranker("Fictional query", rows) == rows
    assert calls == [("Fictional query", 58, {"window": 58})]


def test_cli_reserves_output_before_any_execution(monkeypatch, replay):
    f = replay
    calls = []
    monkeypatch.setattr(
        f.r,
        "reserve_output",
        lambda p: calls.append(("reserve", p)) or {"run_id": "reserved"},
        raising=False,
    )
    monkeypatch.setattr(
        f.r,
        "execute",
        lambda args, reg: calls.append(("execute", reg)) or {"valid": True, "status": "complete"},
    )
    argv = ["execute"]
    for key in (
        *f.r.INPUT_ARGUMENTS,
        "context_root",
        "historical_code_root",
        "registration",
        "private_root",
        "output",
    ):
        argv.extend(["--" + key.replace("_", "-"), getattr(f.args, key)])
    assert f.r.main(argv) == 0
    assert calls == [("reserve", f.args.output), ("execute", {"run_id": "reserved"})]


def test_declined_controlled_arm_keeps_original_and_fresh_answers_independent(replay, monkeypatch):
    f = replay
    register(f)

    def component(*a, **kw):
        result = f.component(*a, **kw)
        result.arms["controlled_supp"]["status"] = "declined"
        result.arms["original_supp"]["status"] = "selected"
        return result

    monkeypatch.setattr(f.r, "run_components", component)
    out = f.r.execute(f.args, {"run_id": "declined"})
    assert out["valid"] and len(f.gen_calls) == 141 and len(f.source_calls) == 47
    assert out["source_tasks"][0]["source_group_wall_ms"] >= 0
    assert out["source_tasks"][0]["cached_pair_owner"] == "unknown"


def test_additional_attestation_fields_cannot_be_silently_accepted(replay):
    f = replay
    f.metadata["grades"] = "SECRET semantic grades"
    Path(f.args.review_metadata).write_text(json.dumps(f.metadata))
    with pytest.raises(ValueError):
        register(f)


def test_changed_suite_preserves_trusted_selection_full_denominator(replay):
    f = replay
    register(f)
    suite = json.loads(Path(f.args.suite).read_bytes())
    suite["tasks"][0]["goal"] = "Changed fictional question"
    Path(f.args.suite).write_text(json.dumps(suite))
    out = f.r.execute(f.args, {"run_id": "suite-drift"})
    assert not out["valid"] and not f.source_calls and not f.gen_calls
    assert all(len(rows) == 47 for rows in out["rows"].values())
    assert out["selected_task_ids"] == [t["id"] for t in f.tasks]


def test_transport_effective_configuration_must_match_registered_caps(replay, monkeypatch):
    f = replay
    register(f)
    factory = f.r.authorized_search

    def transport(*a, **kw):
        result = factory(*a, **kw)
        result.cfg = result.cfg.model_copy(update={"document_quota": 2})
        return result

    monkeypatch.setattr(f.r, "authorized_search", transport)
    out = f.r.execute(f.args, {"run_id": "cfg-drift"})
    assert not out["valid"] and not f.source_calls and not f.gen_calls


def test_legacy_anchor_alias_resolves_same_original_file_without_rewriting(replay):
    f = replay
    alias = f.base.parent.parent / "legacy-alias"
    alias.symlink_to(f.base.parent.parent, target_is_directory=True)
    metadata = deepcopy(f.metadata)
    original = Path(metadata["anchor"]["registration_path"])
    metadata["anchor"]["registration_path"] = str(alias / original.relative_to(f.base.parent.parent))
    _, _, _, anchorpath = f.r.lineage_paths(f.args, f.generation, "prior")
    anchorpath.write_text(json.dumps(metadata["anchor"]))
    Path(f.args.review_metadata).write_text(json.dumps(metadata))
    assert register(f)["historical_code_sha256"] == digest(f.generation["code"])
    assert json.loads(anchorpath.read_bytes()) == metadata["anchor"]


def test_readonly_scope_refreshes_sqlalchemy_identity_map_before_acl_check(monkeypatch):
    seen = []
    db = SimpleNamespace(expire_all=lambda: seen.append("expire"))
    corpus, index = {"documents": []}, {"index": "fictional", "chunks": []}

    def scope(session):
        assert seen == ["expire"], "Stale identity map read before ACL refresh"
        return {}, [], corpus

    monkeypatch.setattr(runner, "active_scope", scope)
    monkeypatch.setattr(runner, "index_ledger", lambda *a: index)
    assert runner.unchanged_scope(db, None, corpus, index) == {}


def test_group_callbacks_do_not_repeat_full_chunk_and_vector_scan(replay, monkeypatch):
    f = replay
    register(f)
    full = f.r.index_ledger
    calls = []

    def counted(*args):
        calls.append("full")
        return full(*args)

    monkeypatch.setattr(f.r, "index_ledger", counted)

    def component(contract, seed, candidates, **kw):
        kw["validate_scope"](contract.question, contract.question)
        kw["reauthorize"](seed)
        kw["search"](SimpleNamespace(query=contract.question))
        return f.component(contract, seed, candidates, **kw)

    monkeypatch.setattr(f.r, "run_components", component)
    out = f.r.execute(f.args, {"run_id": "bounded-scope"})
    assert out["valid"]
    assert len(calls) == 2, "Full index scans belong at whole-run boundaries"


@pytest.mark.parametrize("mutation", ["acl", "version", "metadata", "population", "mapping"])
@pytest.mark.parametrize("when", ["before", "after"])
def test_source_callback_quick_guard_rejects_current_drift(replay, monkeypatch, mutation, when):
    f = replay
    register(f)
    state = {"changed": False}
    calls = []
    original_scope = f.r.current_scope
    original_population = f.r.index_population

    def active(db):
        users, corpus = original_scope(db)
        if state["changed"] and mutation in ("acl", "version", "metadata"):
            corpus = deepcopy(corpus)
            corpus["documents"].append({"document_id": mutation})
        return users, corpus

    def population(*args):
        value = original_population(*args)
        if state["changed"] and mutation == "population":
            value = deepcopy(value)
            value["count"] = 1
        return value

    monkeypatch.setattr(f.r, "current_scope", active)
    monkeypatch.setattr(f.r, "index_population", population)
    if mutation == "mapping":
        import app.clients

        monkeypatch.setattr(
            app.clients,
            "Search",
            lambda **kw: SimpleNamespace(
                index=kw["index"],
                request=lambda *a, **k: {
                    kw["index"]: {"mappings": {"changed": True} if state["changed"] else {}}
                },
            ),
        )
    factory = f.r.authorized_search

    def transport(*args, **kw):
        result = factory(*args, **kw)
        original = result.validate_scope

        def operation(*a):
            calls.append("model")
            value = original(*a)
            if when == "after":
                state["changed"] = True
            return value

        result.validate_scope = operation
        return result

    monkeypatch.setattr(f.r, "authorized_search", transport)

    def component(contract, seed, candidates, **kw):
        if when == "before":
            state["changed"] = True
        kw["validate_scope"](contract.question, contract.question)
        return f.component(contract, seed, candidates, **kw)

    monkeypatch.setattr(f.r, "run_components", component)
    out = f.r.execute(f.args, {"run_id": f"quick-{mutation}-{when}"})
    assert not out["valid"] and not f.gen_calls
    assert calls == ([] if when == "before" else ["model"])


@pytest.mark.parametrize("mutation", ["user_acl", "readable", "document_acl", "version", "metadata"])
def test_current_scope_rebuilds_complete_acl_and_version_snapshot(monkeypatch, mutation):
    users = {
        uid: SimpleNamespace(
            id=uid, tenant_id="tenant", username=uid, role="member", groups=["engineering"], active=True
        )
        for uid in ("lt-eng", "mh-eval")
    }
    document = SimpleNamespace(
        id="d",
        tenant_id="tenant",
        active_version_id="v",
        owner_id="owner",
        read_groups=["engineering"],
        tenant_public=False,
        deleted=False,
        revision=1,
        title="Original",
        metadata_json={"region": "east"},
    )
    version = SimpleNamespace(
        id="v",
        document_id="d",
        status="ready",
        content_hash="a" * 64,
        pipeline={"chunking": "fixed"},
    )
    state = {"readable": True}

    class DB:
        def expire_all(self):
            pass

        def get(self, model, key, **kwargs):
            if model is runner.User:
                return users.get(key)
            if model is runner.DocumentVersion:
                return version if key == version.id else None
            raise AssertionError(model)

    monkeypatch.setattr(
        runner, "readable_documents", lambda db, user: [document] if state["readable"] else []
    )
    db = DB()
    _, snapshot = runner.current_scope(db)
    expected = deepcopy(snapshot)
    expected["documents"][0]["chunks"] = []
    source = SimpleNamespace(index="source", request=lambda *a: {"source": {"mappings": {}}})
    frozen_index = {"index": "source", "population": {"count": 1}, "mapping_sha256": digest({})}
    monkeypatch.setattr(runner, "index_population", lambda *a: {"count": 1})
    assert runner.quick_scope(db, source, expected, frozen_index)["lt-eng"] is users["lt-eng"]
    if mutation == "user_acl":
        users["lt-eng"].groups = ["support"]
    elif mutation == "readable":
        state["readable"] = False
    elif mutation == "document_acl":
        document.revision += 1
    elif mutation == "version":
        document.active_version_id = version.id = "v2"
    else:
        document.metadata_json = {"region": "west"}
    with pytest.raises(ValueError, match="ACL/version/metadata drift"):
        runner.quick_scope(db, source, expected, frozen_index)


def test_generation_postcheck_catches_change_during_model_action(replay, monkeypatch):
    f = replay
    register(f)
    changed = [False]
    original = f.r.current_scope

    def scope(db):
        users, corpus = original(db)
        if changed[0]:
            corpus = deepcopy(corpus)
            corpus["documents"].append({"document_id": "changed-during-generation"})
        return users, corpus

    monkeypatch.setattr(f.r, "current_scope", scope)
    generation = f.r.generate_answer

    def generate(*args, **kwargs):
        result = generation(*args, **kwargs)
        changed[0] = True
        return result

    monkeypatch.setattr(f.r, "generate_answer", generate)
    out = f.r.execute(f.args, {"run_id": "generation-post-drift"})
    assert not out["valid"] and len(f.gen_calls) == 1
    assert not f.evaluated


@pytest.mark.parametrize("when", ["before", "after"])
def test_ranker_action_checks_index_marker_on_both_sides(replay, monkeypatch, when):
    f = replay
    register(f)
    changed = [False]
    calls = []
    population = f.r.index_population
    monkeypatch.setattr(
        f.r,
        "index_population",
        lambda *a: {"count": 1} if changed[0] else population(*a),
    )

    def ranking(query, rows):
        calls.append("ranker")
        if when == "after":
            changed[0] = True
        return []

    monkeypatch.setattr(f.r, "ranker", ranking)

    def component(contract, seed, candidates, **kw):
        if when == "before":
            changed[0] = True
        kw["ranker"](contract.question, seed)
        return f.component(contract, seed, candidates, **kw)

    monkeypatch.setattr(f.r, "run_components", component)
    out = f.r.execute(f.args, {"run_id": "ranker-" + when})
    assert not out["valid"] and not f.gen_calls
    assert calls == ([] if when == "before" else ["ranker"])


def test_whole_run_end_still_full_scans_chunks_and_vectors(replay, monkeypatch):
    f = replay
    register(f)
    original = f.r.index_ledger
    calls = []

    def ledger(*args):
        calls.append(True)
        value = original(*args)
        if len(calls) == 2:
            value = deepcopy(value)
            value["chunks"].append({"id": "changed-only-at-full-scan"})
        return value

    monkeypatch.setattr(f.r, "index_ledger", ledger)
    out = f.r.execute(f.args, {"run_id": "final-full-drift"})
    assert len(calls) == 2 and not out["valid"]
    assert out["drift"]["runtime"] is True


def test_direct_source_reads_and_generation_have_immediate_guard_brackets(replay, monkeypatch):
    f = replay
    register(f)
    events = []
    quick = f.r.quick_scope
    rehydrate = f.r.replay.rehydrate_task
    authorize = f.r.reauthorized
    generate = f.r.generate_answer

    def guard(*args):
        events.append("guard")
        return quick(*args)

    def recorded(name, operation):
        def call(*args, **kwargs):
            events.append(name)
            return operation(*args, **kwargs)

        return call

    monkeypatch.setattr(f.r, "quick_scope", guard)
    monkeypatch.setattr(f.r.replay, "rehydrate_task", recorded("rehydrate", rehydrate))
    monkeypatch.setattr(f.r, "reauthorized", recorded("reauthorize", authorize))
    monkeypatch.setattr(f.r, "generate_answer", recorded("generate", generate))
    out = f.r.execute(f.args, {"run_id": "direct-action-brackets"})
    assert out["valid"]
    for i, event in enumerate(events):
        if event in {"rehydrate", "reauthorize", "generate"}:
            assert events[i - 1] == events[i + 1] == "guard", (event, events[max(i - 2, 0) : i + 3])


@pytest.mark.parametrize("outcome", ["success", "cancelled"])
def test_source_model_http_dispatch_has_immediate_guard_brackets(monkeypatch, outcome):
    events = []

    def transport(self, path, body, base=None):
        events.append("dispatch")
        if outcome == "cancelled":
            raise TaskCancelled("synthetic cancellation")
        return {"prompt_eval_count": 1, "eval_count": 1}

    monkeypatch.setattr(runner.supplement.Models, "_post", transport)
    model = runner.make_source_models(
        before=lambda: events.append("guard-before"),
        after=lambda primary: events.append(
            ("guard-after", type(primary).__name__ if primary else None)
        ),
    )
    if outcome == "cancelled":
        with pytest.raises(TaskCancelled):
            model._post("/api/chat", {"messages": []})
    else:
        model._post("/api/chat", {"messages": []})
    assert events == [
        "guard-before",
        "dispatch",
        ("guard-after", "TaskCancelled" if outcome == "cancelled" else None),
    ]
    assert len(model.events) == 1
    assert model.events[0]["status"] == ("failed" if outcome == "cancelled" else "ok")


@pytest.mark.parametrize("when", ["before", "after"])
def test_source_model_dispatch_fails_closed_on_live_guard_drift(monkeypatch, when):
    changed = [when == "before"]
    calls = []

    def transport(self, path, body, base=None):
        calls.append("dispatch")
        changed[0] = True
        return {"prompt_eval_count": 1, "eval_count": 1}

    def guard():
        if changed[0]:
            raise ValueError("current scope drift")

    monkeypatch.setattr(runner.supplement.Models, "_post", transport)
    model = runner.make_source_models(before=guard, after=lambda primary: guard())
    with pytest.raises(ValueError, match="current scope drift"):
        model._post("/api/chat", {"messages": []})
    assert calls == ([] if when == "before" else ["dispatch"])


def test_generation_reauthorization_refreshes_full_source_records(monkeypatch):
    seen = []
    db = SimpleNamespace(expire_all=lambda: seen.append("expire"))
    evidence = [
        dict(
            chunk_id="c",
            document_id="d",
            version_id="v",
            source_sha256="a" * 64,
            title="",
            text="Fictional source",
            metadata={},
            locator={},
        )
    ]

    def rebind(*args):
        assert seen == ["expire"], "Stale source record read before refresh"
        return deepcopy(evidence)

    monkeypatch.setattr(runner.replay, "rebind_rows", rebind)
    assert runner.reauthorized(db, None, evidence, {}) == evidence


def test_known_210_archive_layout_binds_five_nonproduction_snapshot_files(replay):
    f = replay
    names = (
        "benchmarks/enterprise_rag/v1/dev/evidence-first-20261011/generation-input-v2.json",
        "tests/test_evidence_focus.py",
        "tests/test_external_answer_generation.py",
        "tests/test_slot_generation_replay.py",
        "tests/test_task_contract.py",
    )
    for name in names:
        path = f.archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            Path(f.args.generation_registration).read_bytes()
            if name.startswith("benchmarks/")
            else b"# synthetic historical test\n"
        )
    value = register(f)
    assert len(value["archive_extra_bytes_sha256"]) == 5
    assert len(value["code"]) == 1  # current mocked source map remains separate
    (f.archive / "tests/test_task_contract.py").write_text("changed historical test")
    out = f.r.execute(f.args, {"run_id": "archive-extra-drift"})
    assert not out["valid"] and not f.source_calls and not f.gen_calls


def test_budget_internal_zero_tokens_never_claim_provider_token_observation():
    value = {
        "calls": {
            "policy": {
                "attempted": 1,
                "succeeded": 1,
                "failed": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        },
        "limits": {"policy": 2},
    }
    public = runner.public_budget(value)
    assert "prompt_tokens" not in public["calls"]["policy"]
    assert "completion_tokens" not in public["calls"]["policy"]


def test_each_component_authorization_and_scope_check_refreshes_readonly_records(replay, monkeypatch):
    f = replay
    register(f)
    factory = f.r.authorized_search

    def transport(db, *a, **kw):
        result = factory(db, *a, **kw)
        state = {"fresh": False}
        db.expire_all = lambda: state.update(fresh=True)

        def authorize(rows):
            assert state["fresh"], "Source arm reused stale authorization records"
            state["fresh"] = False
            return deepcopy(rows)

        def scope(*args):
            assert state["fresh"], "Source arm reused stale routing scope"
            state["fresh"] = False
            return True

        result.reauthorize = authorize
        result.validate_scope = scope
        return result

    monkeypatch.setattr(f.r, "authorized_search", transport)

    def component(contract, seed, candidates, **kw):
        kw["reauthorize"](seed)
        kw["reauthorize"](seed)
        assert kw["validate_scope"](contract.question, contract.question)
        assert kw["validate_scope"](contract.question, contract.question)
        return f.component(contract, seed, candidates, **kw)

    monkeypatch.setattr(f.r, "run_components", component)
    out = f.r.execute(f.args, {"run_id": "source-refresh"})
    assert out["valid"] and len(f.gen_calls) == 141


def test_public_identity_projects_complete_frozen_protocol_without_secrets(replay):
    f = replay
    registration = register(f)
    out = f.r.execute(f.args, {"run_id": "public-identity"})
    assert out["valid"] and out["identity"] == registration
    malicious = deepcopy(registration)
    malicious["question"] = "SECRET QUESTION"
    malicious["models"]["policy"]["status"] = "SECRET STATUS"
    malicious["models"]["policy"]["cfg"] = {"api_key": "SECRET KEY"}
    malicious["models"]["policy"]["name"] = "SECRET SOURCE TITLE"
    projected = f.r.public_identity(malicious)
    assert "SECRET" not in json.dumps(projected)
    assert projected["caps"] == {
        "limit": 8,
        "token_budget": 5000,
        "document_quota": None,
        "source_quota": None,
    }
    assert projected["source_limits"] == f.r.SOURCE_LIMITS
    assert projected["generation_limits"] == f.r.GENERATION_LIMITS
    assert projected["generation_protocol"] == "shared_contract"
    assert projected["models"]["policy"]["digest"] == "d" * 64
    assert projected["cache_correlation"]["cached_pair_owner"] is None


def test_archive_same_byte_alias_to_other_declared_key_is_rejected(replay):
    f = replay
    expected = deepcopy(f.generation["code"])
    first, second = sorted(expected)[:2]
    a, b = f.archive / first, f.archive / second
    a.unlink()
    a.symlink_to(b)
    expected[first] = expected[second]
    with pytest.raises(ValueError):
        f.r.archived_code(f.args, expected)


def test_empty_attested_and_anchored_review_parts_cannot_claim_complete_lineage(replay):
    f = replay
    value = deepcopy(f.metadata)
    _, _, review_path, anchor_path = f.r.lineage_paths(f.args, f.generation, "prior")
    review = json.loads(review_path.read_bytes())
    review["files"] = []
    value["parts"] = []
    review_path.write_text(json.dumps(review))
    value["anchor"]["registration_bytes_sha256"] = sha(review_path.read_bytes())
    value["review_registration_bytes_sha256"] = sha(review_path.read_bytes())
    anchor_path.write_text(json.dumps(value["anchor"]))
    Path(f.args.review_metadata).write_text(json.dumps(value))
    with pytest.raises(ValueError):
        register(f)


@pytest.mark.parametrize("capability", ["validate_scope", "reauthorize", "search"])
@pytest.mark.parametrize("when", ["before", "after"])
def test_every_source_capability_checks_index_marker_before_and_after_dispatch(
    replay, monkeypatch, capability, when
):
    f = replay
    register(f)
    changed = [False]
    calls = []
    population = f.r.index_population
    monkeypatch.setattr(
        f.r,
        "index_population",
        lambda *a: {"count": 1} if changed[0] else population(*a),
    )
    factory = f.r.authorized_search

    def transport(*a, **kw):
        result = factory(*a, **kw)
        original = getattr(result, capability)

        def operation(*args):
            calls.append(capability)
            value = original(*args)
            if when == "after":
                changed[0] = True
            return value

        setattr(result, capability, operation)
        return result

    monkeypatch.setattr(f.r, "authorized_search", transport)

    def component(contract, seed, candidates, **kw):
        if when == "before":
            changed[0] = True
        args = (contract.question, contract.question) if capability == "validate_scope" else (seed,)
        if capability == "search":
            args = (SimpleNamespace(query=contract.question),)
        value = kw[capability](*args)
        calls.append("later transport permitted")
        assert value is not None
        return f.component(contract, seed, candidates, **kw)

    monkeypatch.setattr(f.r, "run_components", component)
    out = f.r.execute(f.args, {"run_id": capability + "-" + when})
    assert not out["valid"] and not f.gen_calls
    assert calls == ([] if when == "before" else [capability])


@pytest.mark.parametrize(
    "mutation",
    [
        "part_count",
        "part_order",
        "part_name",
        "part_sha",
        "reviewer",
        "human_type",
        "independence",
        "pending",
        "visible",
        "material_parts",
        "unknown_key",
    ],
)
def test_complete_metadata_only_blind_declaration_rejects_self_consistent_tamper(replay, mutation):
    f = replay
    metadata = deepcopy(f.metadata)
    _, _, review_path, anchor_path = f.r.lineage_paths(f.args, f.generation, "prior")
    review = json.loads(review_path.read_bytes())
    if mutation == "part_count":
        review["files"][0]["items"] = 20
    elif mutation == "part_order":
        review["files"][0], review["files"][1] = review["files"][1], review["files"][0]
    elif mutation == "part_name":
        review["files"][0]["name"] = "other-part.json"
    elif mutation == "part_sha":
        review["files"][0]["sha256"] = "SECRET HASH"
    elif mutation == "reviewer":
        review["files"][0]["reviewer"] = "SECRET reviewer alias"
    elif mutation == "human_type":
        review["files"][0]["human"] = 0
    elif mutation == "independence":
        review["files"][0]["independent_fresh_context"] = False
    elif mutation == "pending":
        review["pending_task_ids"][0] = "T00"
    elif mutation == "visible":
        review["visible_answers"] = 61
    elif mutation == "material_parts":
        review["all_material_parts_required"] = 1
    else:
        review["grade"] = "SECRET GRADE"
    metadata["parts"] = deepcopy(review["files"])
    review_path.write_text(json.dumps(review))
    metadata["review_registration_bytes_sha256"] = sha(review_path.read_bytes())
    metadata["anchor"]["registration_bytes_sha256"] = metadata["review_registration_bytes_sha256"]
    anchor_path.write_text(json.dumps(metadata["anchor"]))
    Path(f.args.review_metadata).write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        register(f)
    assert not f.source_calls and not f.gen_calls


def test_known_archive_extra_entry_cannot_alias_another_relative_file(replay):
    f = replay
    names = (
        "benchmarks/enterprise_rag/v1/dev/evidence-first-20261011/generation-input-v2.json",
        "tests/test_evidence_focus.py",
        "tests/test_external_answer_generation.py",
        "tests/test_slot_generation_replay.py",
        "tests/test_task_contract.py",
    )
    for name in names:
        path = f.archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            Path(f.args.generation_registration).read_bytes()
            if name.startswith("benchmarks/")
            else b"# same bytes\n"
        )
    source, target = f.archive / names[-1], f.archive / names[-2]
    source.unlink()
    source.symlink_to(target)
    with pytest.raises(ValueError):
        register(f)


@pytest.mark.parametrize("interruption", [TaskCancelled, KeyboardInterrupt])
@pytest.mark.parametrize("capability", ["reauthorize", "validate_scope", "search"])
def test_cancel_remains_primary_when_source_postcheck_also_drifts(
    replay, monkeypatch, interruption, capability
):
    from app.supplement_contract import build_supplement_contract
    from scripts.run_policy_ablation_dev import run_components as actual_components

    f = replay
    register(f)
    document = SimpleNamespace(id="d", active_version_id="v", source_sha256="a" * 64, metadata_json={})
    monkeypatch.setattr(
        f.r, "contract_for", lambda db, user, q: build_supplement_contract(q, [document])
    )
    source_calls = []

    def component(*args, **kw):
        source_calls.append("first task")
        return actual_components(*args, **kw)

    monkeypatch.setattr(f.r, "run_components", component)
    changed = [False]
    population = f.r.index_population
    monkeypatch.setattr(
        f.r,
        "index_population",
        lambda *a: {"count": 1} if changed[0] else population(*a),
    )
    factory = f.r.authorized_search
    called = []

    def transport(*args, **kw):
        result = factory(*args, **kw)

        def cancelled(*a):
            called.append(capability)
            changed[0] = True
            raise interruption("PRIVATE original cancellation")

        setattr(result, capability, cancelled)
        return result

    monkeypatch.setattr(f.r, "authorized_search", transport)
    out = f.r.execute(f.args, {"run_id": "cancel-" + capability + "-" + interruption.__name__})
    assert out["status"] == "interrupted" and not out["valid"]
    assert out["drift"]["runtime"] is True
    assert source_calls == ["first task"] and called == [capability]
    assert not f.gen_calls and not f.evaluated
    raw = json.loads((f.base / "new-private" / out["run_id"] / "raw.json").read_bytes())
    trace = raw["source_tasks"][f.tasks[0]["id"]]["trace"]
    assert trace["status"] == "interrupted"
    current = "original_supp" if capability == "search" else "no_supp"
    expected = {"no_supp": False, "original_supp": capability != "search", "controlled_supp": True}
    assert {arm: budget is None for arm, budget in trace["budgets"].items()} == expected
    if capability == "search":
        assert trace["budgets"][current]["calls"]["search"]["attempted"] == 1
        assert trace["budgets"][current]["calls"]["search"]["failed"] == 1
    errors = raw["source_scope_post_errors"]
    assert errors[-1]["primary_type"] == interruption.__name__
    assert errors[-1]["callback"] == capability
    assert all(
        len(rows) == 47 and all(r["generation_cost"] is None for r in rows)
        for rows in out["rows"].values()
    )
    assert "PRIVATE original cancellation" not in Path(f.args.output).read_text()
