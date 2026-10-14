"""Registered complete Dev47 source-policy replay, with fresh shared-contract answers.

This is an experiment CLI, never a production policy or semantic judge. Historical
review files are opaque byte bindings. All runtime arms finish before annotations
or proxy diagnostics are parsed. Source and generation execute as separate groups.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.config import settings
from app.execution_budget import TaskCancelled
from app.models import DocumentVersion, User
from app.rerank import rerank
from app.security import readable_documents
from scripts import run_evidence_selection_dev as replay
from scripts import run_evidence_supplement_dev as supplement
from scripts.benchmark_package_scoring import answer_review_packet, provisional_score
from scripts.benchmark_runtime import atomic_json
from scripts.dev_evidence_annotations import validate_annotation
from scripts.research_review import digest
from scripts.retrieval_quality_index import DEV_USERS, active_scope, index_ledger, index_population
from scripts.retrieval_quality_metrics import diagnose
from scripts.run_external_answers_dev import generate_answer
from scripts.run_guarded_selection_dev import active_reranker_identity
from scripts.run_policy_ablation_dev import (
    ARMS,
    CAPS,
    SOURCE_LIMITS,
    ComponentResult as ComponentResult,
    authorized_search,
    run_components,
)
from scripts.run_retrieval_quality_dev import (
    ROOT,
    SELECTION,
    _publish,
    code_fingerprints,
    reserve_output,
    sha,
    validate_suite,
)
from scripts.run_reviewed_supplement_dev import rescore, validate_seed
from scripts.run_slot_generation_dev import GenerationModels as ExistingGenerationModels
from scripts.run_slot_generation_dev import cost_projection, selected_profile
from scripts.run_supplement_contract_dev import contract_for

INPUT_ARGUMENTS = (
    "suite",
    "parent",
    "baseline",
    "selection_result",
    "overlay",
    "manifest",
    "generation_registration",
    "generation_public",
    "review_metadata",
)
GENERATION_LIMITS = {
    "generation": 3,
    "policy": 0,
    "search": 0,
    "embedding": 0,
    "ranker": 0,
    "judge": 0,
    "judge_tokens": 0,
}
GENERATION_CAPS = {
    "seconds": 180,
    "first_output_tokens": 700,
    "conflict_tokens": 200,
    "verdict_max_tokens": 500,
}
STAGES = ("preflight", "source", "generation", "post_runtime_evaluation", "complete")
STATUSES = ("not_observed", "source_observed", "completed", "execution_failed", "interrupted")
ROLES = ("policy", "search", "embedding", "ranker", "generation", "judge")
ATTESTATION_RUNNER_ARCHIVE = Path(
    ".runtime/evidence-first-20261011/"
    "citation-style-before-20261012-v1/scripts/run_bound_citation_dev.py"
)
MEASURES = (
    "attempted",
    "succeeded",
    "failed",
    "prompt_tokens",
    "completion_tokens",
    "wall_ms",
    "server_ms",
    "input_pairs",
    "fresh_pairs",
    "cached_pairs",
    "physical_ranker_calls",
    "physical_search_calls",
    "input_count",
)
SCOPE_VALIDATION = {
    "run_boundaries": "full_active_corpus_chunks_vectors_and_index_ledger",
    "action_boundaries": "current_users_readable_documents_acl_versions_metadata_index_population_and_mapping",
    "evidence": "each_used_row_live_reauthorized_with_source_bytes",
    "index_marker_limit": "population_and_mapping_are_change_markers_not_full_source_byte_checks",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def attestation_runner_path():
    path = ROOT / ATTESTATION_RUNNER_ARCHIVE
    require(path.is_file() and path.resolve() == path, "Attestation runner path changed")
    return path


def private_root(path):
    result = Path(path).resolve()
    require(result.is_relative_to((ROOT / ".runtime").resolve()), "Ignored project runtime required")
    return result


def safe_id(value):
    require(type(value) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value), "Invalid identifier")
    return value


def source_configuration(cfg):
    return supplement.supplement_configuration(cfg).model_copy(
        update={"focused_generation_enabled": True, "document_quota": None}
    )


def generator_configuration(cfg):
    return cfg.model_copy(update={"agent_policy_model": cfg.chat_model, "agent_policy_url": ""})


def model_bindings(cfg):
    source = supplement.active_model_identity(source_configuration(cfg))
    generator = supplement.active_model_identity(generator_configuration(cfg))
    return {
        "source": source,
        "generator": generator["policy"],
        "generator_libraries": generator["libraries"],
        "ranker": active_reranker_identity(),
    }


def extra_code_bindings():
    paths = [
        ROOT / "tests/test_policy_ablation_dev.py",
        ROOT / "tests/test_policy_ablation_replay_dev.py",
        ROOT / "tests/test_slot_generation_replay.py",
        ROOT / "tests/test_guarded_selection_replay.py",
        ROOT / "tests/test_bound_citation_dev.py",
        ROOT / ".runtime/evidence-first-20261011/generation_review_v2.py",
    ]
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in paths}


class GenerationModels(ExistingGenerationModels):
    """Tighten the existing budget at assignment, before generate_answer enters it."""

    @property
    def answer_budget(self):
        return getattr(self, "_replay_answer_budget", None)

    @answer_budget.setter
    def answer_budget(self, value):
        value.state["limits"].update(GENERATION_LIMITS)
        self._replay_answer_budget = value


def ranker(query, rows):
    return rerank(query, rows, window=len(rows))


class GuardedSourceModels(supplement.MeteredModels):
    """Bracket actual policy and embedding HTTP dispatch with live scope checks."""

    def __init__(self, before, after):
        super().__init__()
        self._scope_before = before
        self._scope_after = after

    def _post(self, path, body, base=None):
        self._scope_before()
        primary = None
        try:
            return super()._post(path, body, base=base)
        except (TaskCancelled, KeyboardInterrupt) as exc:
            primary = exc
            raise
        finally:
            try:
                self._scope_after(primary)
            except BaseException:
                if primary is None:
                    raise


def make_source_models(*, before=None, after=None):
    if before is None and after is None:
        return supplement.MeteredModels()
    require(callable(before) and callable(after), "Both source model scope guards required")
    return GuardedSourceModels(before, after)


def make_backend(source):
    return supplement.MeteredSearch(source)


def reference_header(run_id):
    return {
        "version": "slot-generation-review-dev-v2",
        "split": "dev",
        "comparison": "fallback",
        "valid": True,
        "status": "complete",
        "run_id": run_id,
        "reviewer_model": "GPT-6 / Codex",
        "reviewer_human": False,
        "exact_serving_variant": None,
        "positions": 94,
        "strict_scoreable_tasks": 40,
        "canonical_evidence_tasks": 47,
        "reviewer_independent_fresh_context": True,
        "reviewer_independent_of_implementation_context": True,
        "model_family_independent_of_generator": True,
        "independent_of_model_gold_audit": False,
        "historical_answer_review_reused": False,
        "historical_evidence_review_reused": True,
        "headline_eligible": False,
        "historical_retrieval_gain_claim": False,
    }


def lineage_paths(args, generation, run_id):
    """Exact legacy declaration; paths resolve aliases to the same original files."""
    base = ROOT / ".runtime/evidence-first-20261011"
    dev = ROOT / "benchmarks/enterprise_rag/v1/dev/evidence-first-20261011"
    run, out = base / "generation" / safe_id(run_id), base / "generation-review-v2"
    old = base / "generation/0ce3a066137e4a0eae0499d13e21be83"
    old_out = base / "generation-review-v1"
    paths = {
        "raw": run / "raw.json",
        "suite": run / "suite.json",
        "packet": run / "review-packet.json",
        "runtime_registration": run / "registration.json",
        "generation_registration": dev / "generation-input-v2.json",
        "source_artifact": ROOT / "artifacts/dev-slot-generation-20261011-v2.json",
        "v1_registration": dev / "generation-review-input-v1.json",
        "v1_artifact": ROOT / "artifacts/dev-slot-generation-review-20261011-v1.json",
        "v1_raw": old / "raw.json",
        "v1_suite": old / "suite.json",
        "v1_packet": old / "review-packet.json",
        "v1_mapping": old_out / "mapping.json",
        "v1_instructions": base / "generation-review-instructions-v1.md",
        "helper": base / "generation_review_v2.py",
        "instructions": base / "generation-review-instructions-v2.md",
        "mapping": out / "mapping.json",
        "canonical": out / "canonical-evidence.json",
    }
    for name in ("prepare_generation_review_v1.py", "score_generation_review_v1.py"):
        paths["v1_helper_" + name] = base / name
    for kind, count in (("answer", 4), ("evidence", 2)):
        for number in range(1, count + 1):
            name = f"{kind}-part-{number}.json"
            paths["v1_input_" + name] = old_out / name
            if kind == "evidence":
                name = f"{kind}-part-{number}-reviews.json"
                paths["v1_review_" + name] = old_out / name
    for number in (1, 2, 3):
        paths["answer_part_" + str(number)] = out / f"answer-part-{number}.json"
    for number, relative in enumerate(sorted(generation["code"])):
        paths[f"production_code_{number:04d}"] = ROOT / relative
    refs = {
        "review_public": ROOT / "artifacts/dev-slot-generation-review-20261011-v2.json",
        "reviewed": out / "reviewed.json",
        "scored": out / "scored-primary.json",
        "canonical": out / "canonical-evidence.json",
        **{f"reviewer_{i}": out / f"answer-part-{i}-reviews.json" for i in (1, 2, 3)},
    }
    return paths, refs, dev / "generation-review-input-v2.json", out / "registration-sha256.json"


def archived_code(args, expected):
    archive = private_root(args.historical_code_root)
    require(type(expected) is dict and len(expected) == 205, "Complete historical 205 files required")
    paths = {}
    for relative, fingerprint in expected.items():
        require(
            type(relative) is str
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts
            and str(Path(relative)) == relative,
            "Canonical contained archive path required",
        )
        path = (archive / relative).resolve()
        require(
            path.is_relative_to(archive)
            and str(path.relative_to(archive)) == relative
            and path.is_file()
            and sha(path.read_bytes()) == fingerprint,
            "Historical source bytes changed",
        )
        paths[relative] = path
    actual = {str(p.relative_to(archive)) for p in archive.rglob("*") if p.is_file()}
    allowed_extras = {
        "benchmarks/enterprise_rag/v1/dev/evidence-first-20261011/generation-input-v2.json",
        "tests/test_evidence_focus.py",
        "tests/test_external_answer_generation.py",
        "tests/test_slot_generation_replay.py",
        "tests/test_task_contract.py",
    }
    extras = actual - set(expected)
    require(
        set(expected) <= actual and (not extras or extras == allowed_extras),
        "Historical archive key set changed",
    )
    for relative in extras:
        path = (archive / relative).resolve()
        require(
            path.is_relative_to(archive) and str(path.relative_to(archive)) == relative,
            "Canonical archive extra path required",
        )
    return paths


PENDING_IDS = frozenset(
    {
        "U12",
        "U22",
        "SF-3e8219f58317",
        "SF-56d3160a9c64",
        "SF-59679085cc4a",
        "SF-3ea856eeec7b",
        "SF-3fb3599dcc99",
    }
)


def validate_review_declaration(review, generation, public, metadata, expected):
    """Verify the original blind protocol using metadata and opaque byte bindings.

    Semantic part SHA values are prior attested metadata, not recomputed by parsing
    question/answer part bodies. Their exact file bytes are independently bound.
    """
    from scripts.run_external_answer_review import SYSTEM

    fixed = {
        "version": "slot-generation-review-input-v2",
        "split": "dev",
        "comparison": "fallback",
        "source_run_id": metadata["source_run_id"],
        "protocols": ["shared_contract", "literal_fallback"],
        "candidate": "literal_fallback",
        "baseline": "shared_contract",
        "positions": 94,
        "strict_scoreable_tasks": 40,
        "canonical_evidence_tasks": 47,
        "visible_answers": 62,
        "settings_sha256": generation["settings_sha256"],
        "production_code_sha256": generation["code"],
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
    required = set(fixed) | {
        "raw_sha256",
        "suite_sha256",
        "packet_sha256",
        "mapping_sha256",
        "rubric_sha256",
        "instructions_sha256",
        "assembly_sha256",
        "frozen_files",
        "files",
        "pending_task_ids",
    }
    require(
        type(review) is dict
        and set(review) == required
        and all(type(review.get(k)) is type(v) and review[k] == v for k, v in fixed.items()),
        "Complete original blind review declaration required",
    )
    pending = generation["review"]["pending_task_ids"]
    require(
        type(pending) is list
        and len(pending) == 7
        and all(type(tid) is str for tid in pending)
        and set(pending) == PENDING_IDS
        and PENDING_IDS <= set(generation["selected_task_ids"])
        and review["pending_task_ids"] == pending,
        "Original pending seven bindings required",
    )
    for key in (
        "raw_sha256",
        "suite_sha256",
        "packet_sha256",
        "mapping_sha256",
        "rubric_sha256",
        "settings_sha256",
        "instructions_sha256",
    ):
        require(
            type(review[key]) is str and re.fullmatch(r"[0-9a-f]{64}", review[key]),
            "Review metadata SHA required",
        )
    require(
        review["raw_sha256"] == public["private_raw_sha256"]
        and review["rubric_sha256"] == digest(SYSTEM)
        and review["instructions_sha256"] == digest(expected["instructions"].read_text())
        and review["assembly_sha256"]
        == {expected["helper"].name: digest(expected["helper"].read_text())},
        "Original review rubric or source metadata differs",
    )
    require(
        review["files"] == metadata["parts"]
        and type(review["files"]) is list
        and len(review["files"]) == 3,
        "Three original review parts required",
    )
    generator = generation["generator_identity"]["name"]
    require(generator == "qwen2.5:7b-instruct", "Original historical generator required")
    for number, (part, count) in enumerate(zip(review["files"], (21, 21, 20), strict=True), 1):
        declaration = {
            "name": f"answer-part-{number}.json",
            "items": count,
            "reviewer": f"GPT-6 / Codex blind fallback answer review {number}",
            "reviewer_type": "model",
            "model": "GPT-6 / Codex",
            "exact_serving_variant": None,
            "human": False,
            "independent_fresh_context": True,
            "independent_of_implementation_context": True,
            "model_family_independent_of_generator": True,
            "generator_model_name": generator,
            "reviewer_identity_basis": "system GPT-6 Codex identity; exact serving variant unavailable",
            "independent_of_model_gold_audit": False,
            "tokens": None,
            "cost": None,
        }
        require(
            type(part) is dict
            and set(part) == set(declaration) | {"sha256"}
            and all(type(part.get(k)) is type(v) and part[k] == v for k, v in declaration.items())
            and type(part["sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", part["sha256"]),
            "Ordered complete review part identities required",
        )


def validate_lineage(args, generation, public, metadata):
    """Validate attestation and bytes only; never load reference decisions."""
    required_keys = {
        "version",
        "source_run_id",
        "header",
        "review_registration_bytes_sha256",
        "generation_registration_bytes_sha256",
        "raw_bytes_sha256",
        "suite_bytes_sha256",
        "packet_bytes_sha256",
        "reviewed_semantic_sha256",
        "reference_file_bytes_sha256",
        "frozen_files",
        "anchor",
        "parts",
        "upstream_code",
        "runner_code_sha256",
        "stage2_helper_bytes_sha256",
    }
    require(
        type(metadata) is dict and set(metadata) == required_keys, "Exact attestation fields required"
    )
    run_id = safe_id(metadata["source_run_id"])
    require(
        metadata["version"] == "bound-citation-review-metadata-v1"
        and metadata["header"] == reference_header(run_id)
        and all(type(metadata["header"].get(k)) is type(v) for k, v in reference_header(run_id).items()),
        "Review attestation header changed",
    )
    require(
        generation["version"] == "slot-generation-input-v2"
        and generation["split"] == "dev"
        and generation["subset"] is False
        and generation["comparison"] == "fallback"
        and generation["protocols"] == ["shared_contract", "literal_fallback"]
        and generation["shared_context"]["profile"] == "baseline",
        "Historical generation declaration differs",
    )
    require(
        public["run_id"] == run_id
        and public["valid"] is True
        and public["status"] == "complete"
        and public["error"] is None
        and public["split"] == "dev"
        and public["subset"] is False
        and public["identity"] == generation
        and public["selected_task_ids"] == generation["selected_task_ids"],
        "Historical public source differs",
    )
    attestation_runner = attestation_runner_path()
    require(
        metadata["runner_code_sha256"] == sha(attestation_runner.read_bytes()),
        "Attestation runner bytes changed",
    )
    archive = archived_code(args, generation["code"])
    require(metadata["upstream_code"] == generation["code"], "Attested source declaration differs")
    expected, refs, review_path, anchor_path = lineage_paths(args, generation, run_id)
    require(
        Path(args.generation_registration).resolve() == expected["generation_registration"].resolve()
        and Path(args.generation_public).resolve() == expected["source_artifact"].resolve(),
        "Original generation lineage paths required",
    )
    require(
        len(expected) == 235 and set(metadata["frozen_files"]) == set(expected),
        "Complete 235 metadata bindings required",
    )
    frozen = metadata["frozen_files"]
    archive_root = private_root(args.historical_code_root)
    bytepaths = {
        "archive_extra_" + sha(str(p.relative_to(archive_root)).encode()): p
        for p in archive_root.rglob("*")
        if p.is_file() and str(p.relative_to(archive_root)) not in generation["code"]
    }
    archived_registration = (
        archive_root
        / "benchmarks/enterprise_rag/v1/dev/evidence-first-20261011/generation-input-v2.json"
    )
    if archived_registration.exists():
        require(
            archived_registration.read_bytes() == Path(args.generation_registration).read_bytes(),
            "Archived generation registration differs",
        )
    relatives = sorted(generation["code"])
    for key, declared in expected.items():
        record = frozen[key]
        production = key.startswith("production_code_")
        relative = relatives[int(key.rsplit("_", 1)[1])] if production else None
        # Historical app files may have changed. Verify their declared canonical
        # path, then verify precisely that relative file in the 205-file archive.
        declared_path = declared.resolve()
        require(
            set(record) == {"path", "bytes_sha256"} and Path(record["path"]).resolve() == declared_path,
            "Attested canonical path differs",
        )
        actual = archive[relative] if production else declared_path
        require(
            actual.is_file() and sha(actual.read_bytes()) == record["bytes_sha256"],
            "Attested upstream bytes changed",
        )
        bytepaths[key] = actual
    review_bytes = review_path.read_bytes()
    review = json.loads(review_bytes)  # Registration metadata, never reviewer decisions.
    require(
        review["frozen_files"] == frozen
        and review["files"] == metadata["parts"]
        and review["production_code_sha256"] == generation["code"],
        "Review registration differs",
    )
    validate_review_declaration(review, generation, public, metadata, expected)
    anchor = metadata["anchor"]
    require(
        type(anchor) is dict
        and set(anchor) == {"registration_path", "registration_bytes_sha256"}
        and type(anchor["registration_path"]) is str
        and Path(anchor["registration_path"]).resolve() == review_path.resolve()
        and anchor["registration_bytes_sha256"] == sha(review_bytes)
        and json.loads(anchor_path.read_bytes()) == anchor,
        "Review registration anchor changed",
    )
    require(
        metadata["review_registration_bytes_sha256"] == sha(review_bytes),
        "Review registration bytes changed",
    )
    for key, path in [
        ("generation_registration", Path(args.generation_registration)),
        ("raw", expected["raw"]),
        ("suite", expected["suite"]),
        ("packet", expected["packet"]),
    ]:
        require(
            metadata[key + "_bytes_sha256"] == sha(path.read_bytes()), "Attested source bytes changed"
        )
    require(
        set(metadata["reference_file_bytes_sha256"]) == set(refs)
        and metadata["reference_file_bytes_sha256"] == {k: sha(p.read_bytes()) for k, p in refs.items()},
        "Reference file bytes changed",
    )
    require(
        metadata["stage2_helper_bytes_sha256"] == sha(expected["helper"].read_bytes()),
        "Historical helper changed",
    )
    bytepaths.update({"reference_" + key: path for key, path in refs.items()})
    bytepaths.update(
        review_registration=review_path,
        review_anchor=anchor_path,
        attestation_runner=attestation_runner,
    )
    return bytepaths


def input_paths(args):
    return {key: Path(getattr(args, key)).resolve() for key in INPUT_ARGUMENTS} | {
        "selection": Path(SELECTION).resolve()
    }


def task_contexts(args, tasks):
    root = private_root(args.context_root)
    result = {}
    for task in tasks:
        path = (root / (safe_id(task["id"]) + ".json")).resolve()
        require(path.parent == root and path.is_file(), "Private context missing or escaping")
        result[task["id"]] = path
    require(
        {p.name for p in root.glob("*.json")} == {p.name for p in result.values()},
        "Private task context set differs",
    )
    return result


def identities(rows):
    return [[row[key] for key in replay.IDENTITY_FIELDS] for row in rows]


def load_inputs(args):
    paths = input_paths(args)
    data = {key: path.read_bytes() for key, path in paths.items()}
    suite, selection = json.loads(data["suite"]), json.loads(data["selection"])
    tasks = validate_suite(suite, selection, ROOT)
    require(
        [t["id"] for t in tasks] == [t["id"] for t in selection["items"]], "Trusted Dev order differs"
    )
    parent, baseline, chosen = (
        json.loads(data[key]) for key in ("parent", "baseline", "selection_result")
    )
    parent_rows = replay.validate_parent(parent, suite, selection, data["suite"], data["selection"])
    # Frozen prior diagnostics are lineage metadata. Current overlay records and
    # reference tests are deliberately unavailable until runtime is complete.
    manifest = json.loads(data["manifest"])
    require(
        validate_seed(baseline, parent, suite, manifest) == "relevant_doc",
        "Accepted relevant_doc seed required",
    )
    require(
        selected_profile(chosen, parent, tasks) == "baseline",
        "Accepted Stage1 original baseline required",
    )
    require(
        all(
            identities(a["telemetry"]["context"]) == identities(b["telemetry"]["context"])
            for a, b in zip(chosen["rows"]["baseline"], baseline["rows"]["relevant_doc"], strict=True)
        ),
        "Stage1 baseline differs from original relevant_doc",
    )
    generation, public, metadata = (
        json.loads(data[key])
        for key in ("generation_registration", "generation_public", "review_metadata")
    )
    ids = [t["id"] for t in tasks]
    require(generation["selected_task_ids"] == ids, "Historical full Dev order differs")
    lineage = validate_lineage(args, generation, public, metadata)
    contexts = task_contexts(args, tasks)
    context_bytes = {key: path.read_bytes() for key, path in contexts.items()}
    require(
        generation["private_task_bytes_sha256"]
        == {key: sha(value) for key, value in context_bytes.items()},
        "Stage2 frozen private context bytes differ",
    )
    originals = []
    for offset, task in enumerate(tasks):
        original = json.loads(context_bytes[task["id"]])
        require(
            original["task_id"] == task["id"]
            and original["question"] == task["goal"]
            and original.get("user", task["user"]) == task["user"],
            "Private task/question/user differs",
        )
        context = original["contexts"]["baseline"]
        telemetry = chosen["rows"]["baseline"][offset]["telemetry"]
        require(identities(context) == identities(telemetry["context"]), "Frozen source order differs")
        require(
            identities(original["pools"]["baseline"]) == identities(telemetry["candidates"]),
            "Frozen candidate order differs",
        )
        replay.validate_context(context, telemetry["context_tokens"])
        originals.append(original)
    return {
        "paths": paths,
        "bytes": data,
        "suite": suite,
        "tasks": tasks,
        "parent": parent,
        "parent_rows": parent_rows,
        "baseline": baseline,
        "chosen": chosen,
        "manifest": manifest,
        "generation": generation,
        "originals": originals,
        "contexts": contexts,
        "context_bytes": context_bytes,
        "lineage_paths": lineage,
        "lineage_bytes": {k: p.read_bytes() for k, p in lineage.items()},
    }


def registration_for(args):
    inputs = load_inputs(args)
    cfg = settings()
    require(
        not cfg.answer_quality_enabled and not cfg.source_facts_enabled,
        "Extra generation pipeline enabled",
    )
    private_root(args.private_root)
    private_root(args.registration)
    return {
        "version": "policy-ablation-replay-input-v1",
        "split": "dev",
        "subset": False,
        "selected_task_ids": [t["id"] for t in inputs["tasks"]],
        "arms": list(ARMS),
        "input_bytes_sha256": {k: sha(v) for k, v in inputs["bytes"].items()},
        "private_task_bytes_sha256": {k: sha(v) for k, v in inputs["context_bytes"].items()},
        "lineage_bytes_sha256": {k: sha(v) for k, v in inputs["lineage_bytes"].items()},
        "historical_code_sha256": digest(inputs["generation"]["code"]),
        "archive_extra_bytes_sha256": {
            k: sha(v) for k, v in inputs["lineage_bytes"].items() if k.startswith("archive_extra_")
        },
        "code": code_fingerprints(),
        "extra_code": extra_code_bindings(),
        "settings_sha256": digest(cfg.model_dump(mode="json")),
        "source_configuration_sha256": digest(source_configuration(cfg).model_dump(mode="json")),
        "generator_configuration_sha256": digest(generator_configuration(cfg).model_dump(mode="json")),
        "models": model_bindings(cfg),
        "phases": ["source", "generation"],
        "source_limits": deepcopy(SOURCE_LIMITS),
        "source_seconds": 180,
        "scope_validation": deepcopy(SCOPE_VALIDATION),
        "caps": deepcopy(CAPS),
        "generation_limits": deepcopy(GENERATION_LIMITS),
        "generation_caps": deepcopy(GENERATION_CAPS),
        "generation_protocol": "shared_contract",
        "order": "task_index_rotated_source_group_then_generation_group",
        "rank_cache": "task_local_exact_query_complete_source_identity",
        "cache_latency_comparison": "shared_cache_no_independent_arm_latency_claim",
        "cache_correlation": {
            "scope": "same_task_across_source_arms",
            "physical_cost_owner": "dispatching_arm",
            "cached_pair_owner": None,
            "latency": "not_independent",
        },
        "baseline": "original_relevant_doc",
        "annotation_access": "post_runtime_only",
        "strict_task_success": None,
        "semantic_recall": None,
        "adoption": None,
    }


def public_hash(value):
    return value if type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) else None


def public_hashes(value, *, paths=False):
    if type(value) is not dict:
        return {}
    result = {}
    for key, fingerprint in value.items():
        valid_key = type(key) is str and re.fullmatch(r"[A-Za-z0-9_./-]{1,240}", key)
        if paths and valid_key:
            valid_key = (
                not Path(key).is_absolute() and ".." not in Path(key).parts and str(Path(key)) == key
            )
        if valid_key and public_hash(fingerprint) is not None:
            result[key] = fingerprint
    return result


def public_libraries(value):
    if type(value) is not dict:
        return {}
    return {
        key: version
        if version is None
        or type(version) is str
        and re.fullmatch(r"[0-9][0-9A-Za-z_.+-]{0,79}", version)
        else None
        for key, version in value.items()
        if key
        in (
            "httpx",
            "pydantic",
            "sqlalchemy",
            "langgraph",
            "torch",
            "transformers",
            "tokenizers",
            "tiktoken",
        )
    }


def public_model(value):
    if type(value) is not dict:
        return {}
    result = {}
    for key in (
        "name",
        "digest",
        "endpoint_sha256",
        "status",
        "runtime",
        "revision",
        "snapshot_path_sha256",
        "files",
        "batch",
        "max_tokens",
        "libraries",
    ):
        if key not in value:
            continue
        v = value[key]
        if key in ("digest", "endpoint_sha256", "snapshot_path_sha256"):
            result[key] = public_hash(v)
        elif key == "name":
            result[key] = v if type(v) is str and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,160}", v) else None
        elif key == "status":
            result[key] = enum(v, ("active", "unused"))
        elif key == "runtime":
            version = v.get("version") if type(v) is dict else None
            result[key] = {
                "version": version
                if type(version) is str and re.fullmatch(r"[0-9][0-9A-Za-z_.+-]{0,79}", version)
                else None
            }
        elif key == "revision":
            result[key] = (
                v if v is None or type(v) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", v) else None
            )
        elif key == "files":
            result[key] = public_hashes(v, paths=True)
        elif key == "libraries":
            result[key] = public_libraries(v)
        else:
            result[key] = v if type(v) is int and v >= 0 else None
    return result


def public_models(value):
    if type(value) is not dict:
        return {}
    result = {}
    for key in (
        "source",
        "generator",
        "generator_libraries",
        "ranker",
        "embedding",
        "policy",
        "libraries",
    ):
        if key not in value:
            continue
        if key == "source":
            source = value[key] if type(value[key]) is dict else {}
            result[key] = {
                role: public_libraries(model) if role == "libraries" else public_model(model)
                for role, model in source.items()
                if role in ("embedding", "policy", "generation", "supplement_reranker", "libraries")
            }
        elif key in ("generator_libraries", "libraries"):
            result[key] = public_libraries(value[key])
        else:
            result[key] = public_model(value[key])
    return result


def public_identity(registration):
    """Complete frozen protocol, projected through explicit typed public schemas."""
    if type(registration) is not dict:
        return None
    result = {
        "version": "policy-ablation-replay-input-v1",
        "split": "dev",
        "subset": False,
        "arms": list(ARMS),
        "phases": ["source", "generation"],
        "source_limits": dict(SOURCE_LIMITS),
        "source_seconds": 180,
        "scope_validation": deepcopy(SCOPE_VALIDATION),
        "caps": dict(CAPS),
        "generation_limits": dict(GENERATION_LIMITS),
        "generation_caps": dict(GENERATION_CAPS),
        "generation_protocol": "shared_contract",
        "order": "task_index_rotated_source_group_then_generation_group",
        "rank_cache": "task_local_exact_query_complete_source_identity",
        "cache_latency_comparison": "shared_cache_no_independent_arm_latency_claim",
        "cache_correlation": {
            "scope": "same_task_across_source_arms",
            "physical_cost_owner": "dispatching_arm",
            "cached_pair_owner": None,
            "latency": "not_independent",
        },
        "baseline": "original_relevant_doc",
        "annotation_access": "post_runtime_only",
        "strict_task_success": None,
        "semantic_recall": None,
        "adoption": None,
    }
    result["selected_task_ids"] = [safe_id(tid) for tid in registration.get("selected_task_ids", [])]
    result["input_bytes_sha256"] = {
        key: public_hash(value)
        for key, value in (registration.get("input_bytes_sha256") or {}).items()
        if key in (*INPUT_ARGUMENTS, "selection")
    }
    for key in (
        "private_task_bytes_sha256",
        "lineage_bytes_sha256",
        "archive_extra_bytes_sha256",
        "code",
        "extra_code",
    ):
        result[key] = public_hashes(registration.get(key), paths=key in ("code", "extra_code"))
    for key in (
        "historical_code_sha256",
        "settings_sha256",
        "source_configuration_sha256",
        "generator_configuration_sha256",
    ):
        result[key] = public_hash(registration.get(key))
    result["models"] = public_models(registration.get("models"))
    return result


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def enum(value, values):
    return value if type(value) is str and value in values else "unknown"


def counters(value):
    value = value if type(value) is dict else {}
    return {key: number(value.get(key)) for key in MEASURES if key in value}


def public_budget(value):
    if type(value) is not dict:
        return None
    return {
        "calls": {
            role: {
                key: metric
                for key, metric in counters(value.get("calls", {}).get(role, {})).items()
                if key not in ("prompt_tokens", "completion_tokens")
            }
            for role in ROLES
        },
        "limits": {role: number(value.get("limits", {}).get(role)) for role in (*ROLES, "judge_tokens")},
    }


def public_event(event):
    event = event if type(event) is dict else {}
    result = counters(event)
    result["status"] = enum(event.get("status"), ("ok", "failed", "pending"))
    if "role" in event:
        result["role"] = enum(event["role"], ROLES)
    if "operation" in event:
        result["operation"] = enum(event["operation"], ("generation", "conflict", "verdict"))
    for key in ("query_sha256", "request_sha256"):
        v = event.get(key)
        result[key] = v if type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v) else None
    for key in ("cache_shared_across_arms",):
        result[key] = event.get(key) if type(event.get(key)) is bool else None
    return result


def public_generation_cost(value):
    if type(value) is not dict:
        return None
    return {
        "physical_generation": counters(value.get("physical_generation")),
        "business_budget": public_budget(value.get("business_budget")),
        "physical_call_records": [public_event(e) for e in value.get("physical_call_records") or []],
        "transport_budget_matched": value.get("transport_budget_matched")
        if type(value.get("transport_budget_matched")) is bool
        else None,
    }


def public_source_cost(value):
    if type(value) is not dict:
        return None
    result = {
        key: number(value.get(key))
        for key in (
            "logical_search_calls",
            "physical_backend_calls",
            "physical_ranker_calls",
            "fresh_ranker_pairs",
            "cached_ranker_pairs",
        )
    }
    for key in (
        "physical_events",
        "backend_events",
        "retrieval_events",
        "ranker_events",
        "search_events",
    ):
        result[key] = [public_event(e) for e in value.get(key) or []]
    result["physical_cost_owner"] = enum(value.get("physical_cost_owner"), ARMS)
    result["cache_shared_across_arms"] = (
        value.get("cache_shared_across_arms")
        if type(value.get("cache_shared_across_arms")) is bool
        else None
    )
    result["transport_budget_matched"] = (
        value.get("transport_budget_matched")
        if type(value.get("transport_budget_matched")) is bool
        else None
    )
    return result


def public_row(row):
    return {
        "task_id": safe_id(row["task_id"]),
        "status": enum(row.get("status"), STATUSES),
        "stage": enum(row.get("stage"), STAGES),
        "error": None if row.get("execution_error_kind") is None else "execution_error",
        "latency_ms": number(row.get("latency_ms")),
        "answer_sha256": digest(row["answer_payload"])
        if row.get("generation_cost") is not None
        else None,
        "source_budget": public_budget(row.get("source_budget")),
        "source_cost": public_source_cost(row.get("source_cost")),
        "generation_cost": public_generation_cost(row.get("generation_cost")),
        "strict_task_success": None,
        "answer_correct": None,
        "semantic_recall": None,
    }


def missing(task_id, run_id):
    return {
        "task_id": task_id,
        "run_id": run_id,
        "status": "not_observed",
        "stage": "preflight",
        "not_observed": True,
        "answer_payload": {"status": "not_observed", "claims": [], "citations": [], "verdict": None},
        "retrieved_evidence": None,
        "first_retrieval_documents": None,
        "steps": None,
        "trace_complete": False,
        "tool_trace": [],
        "scenario_events": {},
        "latency_ms": None,
        "execution_error_kind": "NotObserved",
        "source_budget": None,
        "source_cost": None,
        "generation_cost": None,
        "generation_usage": None,
        "usage": None,
        "transport_budget_matched": None,
        "strict_task_success": None,
        "answer_correct": None,
        "task_success": None,
    }


def source_cost(trace, arm):
    budget = trace["budgets"][arm]
    if budget is None:
        return None
    events = {
        key: deepcopy(trace.get(key, {}).get(arm))
        for key in (
            "physical_events",
            "backend_events",
            "retrieval_events",
            "ranker_events",
            "search_events",
        )
    }
    physical = events["physical_events"]
    searches = events["search_events"]
    ranking = events["ranker_events"]
    calls = budget["calls"]

    def attempted(role):
        return calls.get(role, {}).get("attempted", 0)

    matched = (
        physical is not None
        and searches is not None
        and ranking is not None
        and all(
            attempted(role) == sum(e.get("role") == role for e in physical)
            for role in ("policy", "embedding")
        )
        and attempted("search") == sum(e.get("physical_search_calls", 0) for e in searches)
        and attempted("ranker") == sum(e.get("physical_ranker_calls", 0) for e in ranking)
        and all(attempted(role) == 0 for role in ("generation", "judge"))
    )
    return {
        **events,
        "logical_search_calls": attempted("search"),
        "physical_backend_calls": len(events["backend_events"])
        if events["backend_events"] is not None
        else None,
        "physical_ranker_calls": sum(e.get("physical_ranker_calls", 0) for e in ranking or []),
        "fresh_ranker_pairs": sum(e.get("fresh_pairs", 0) for e in ranking or []),
        "cached_ranker_pairs": sum(e.get("cached_pairs", 0) for e in ranking or []),
        "physical_cost_owner": arm,
        "cache_shared_across_arms": any(e.get("cache_shared_across_arms") for e in ranking or []),
        "transport_budget_matched": matched,
    }


def preserve_components(raw, offset, result):
    raw.setdefault("source_tasks", {})[raw["results"][ARMS[0]][offset]["task_id"]] = deepcopy(
        {"trace": result.trace, "private": result.private, "arms": result.arms}
    )
    for arm in ARMS:
        row = raw["results"][arm][offset]
        observed = result.trace["budgets"][arm] is not None
        row.update(
            stage="source",
            status="source_observed" if observed else "not_observed",
            not_observed=True,
            source_status=result.arms[arm]["status"],
            retrieved_evidence=deepcopy(result.arms[arm]["evidence"]),
            source_candidates=deepcopy(result.arms[arm].get("candidates")),
            source_budget=deepcopy(result.trace["budgets"][arm]),
            source_cost=source_cost(result.trace, arm),
        )
        if result.arms[arm]["status"] in ("failed", "interrupted"):
            row.update(
                status="interrupted"
                if result.arms[arm]["status"] == "interrupted"
                else "execution_failed",
                execution_error_kind="SourceExecutionFailure",
            )


def unchanged_scope(db, source, expected_corpus, expected_index):
    if callable(getattr(db, "expire_all", None)):
        db.expire_all()
    users, records, corpus = active_scope(db)
    current_index = index_ledger(source, records)
    require(corpus == expected_corpus and current_index == expected_index, "Active corpus/index drift")
    return users


def current_scope(db):
    """Rebuild every user's readable document and active-version ledger, without chunks."""
    if callable(getattr(db, "expire_all", None)):
        db.expire_all()
    users, documents, bindings = {}, {}, {}
    for uid in DEV_USERS:
        user = db.get(User, uid, populate_existing=True)
        require(user is not None and user.active, "Dev user missing or inactive")
        users[uid] = user
        readable = readable_documents(db, user)
        bindings[uid] = sorted(doc.id for doc in readable if doc.active_version_id)
        documents.update((doc.id, doc) for doc in readable if doc.active_version_id)
    ledger = []
    for did, doc in sorted(documents.items()):
        version = db.get(DocumentVersion, doc.active_version_id, populate_existing=True)
        require(
            version is not None and version.document_id == did and version.status == "ready",
            "Active version unavailable",
        )
        ledger.append(
            {
                "document_id": did,
                "tenant_id": doc.tenant_id,
                "active_version_id": version.id,
                "source_sha256": version.content_hash,
                "acl_sha256": digest(
                    [doc.owner_id, sorted(doc.read_groups), doc.tenant_public, doc.deleted, doc.revision]
                ),
                "document_sha256": digest([doc.title, doc.metadata_json]),
                "pipeline_sha256": digest(version.pipeline),
            }
        )
    user_ledger = [
        {
            "id": user.id,
            "tenant_id": user.tenant_id,
            "identity_sha256": digest([user.username, user.role, sorted(user.groups), user.active]),
            "readable_document_ids": bindings[uid],
        }
        for uid, user in sorted(users.items())
    ]
    return users, {"documents": ledger, "users": user_ledger}


def quick_scope(db, source, expected_corpus, expected_index):
    """Check live ACL/version metadata and index mutation markers at action boundaries."""
    users, corpus = current_scope(db)
    expected = {
        **expected_corpus,
        "documents": [
            {key: value for key, value in document.items() if key != "chunks"}
            for document in expected_corpus["documents"]
        ],
    }
    require(corpus == expected, "Active corpus ACL/version/metadata drift")
    require(source.index == expected_index["index"], "Source index identity drift")
    require(index_population(source) == expected_index["population"], "Source index population drift")
    mapping = source.request("GET", f"/{source.index}/_mapping")
    require(
        digest(mapping[source.index]["mappings"]) == expected_index["mapping_sha256"],
        "Source index mapping drift",
    )
    return users


def reauthorized(db, user, evidence, corpus):
    if callable(getattr(db, "expire_all", None)):
        db.expire_all()
    bound = replay.rebind_rows(db, user, evidence, corpus)
    # Rebind may normalize selection-only fields; all source bytes and identity
    # fields must still match exactly, including ordered metadata and locator.
    fields = (*replay.IDENTITY_FIELDS, "title", "text", "metadata", "locator")
    require(
        [{k: r.get(k) for k in fields} for r in bound]
        == [{k: r.get(k) for k in fields} for r in evidence],
        "Source identity or bytes drift",
    )
    return deepcopy(evidence)


def snapshot_matches(args, inputs, registration, registration_bytes):
    require(Path(args.registration).read_bytes() == registration_bytes, "Registration bytes drift")
    require(
        inputs["lineage_paths"].get("attestation_runner") == attestation_runner_path(),
        "Attestation runner path changed",
    )
    require(
        code_fingerprints() == registration["code"]
        and extra_code_bindings() == registration["extra_code"],
        "Current code drift",
    )
    cfg = settings()
    require(digest(cfg.model_dump(mode="json")) == registration["settings_sha256"], "Settings drift")
    require(model_bindings(cfg) == registration["models"], "Model identity drift")
    require(
        all(p.read_bytes() == inputs["bytes"][key] for key, p in inputs["paths"].items()),
        "Input bytes drift",
    )
    require(
        all(p.read_bytes() == inputs["context_bytes"][key] for key, p in inputs["contexts"].items()),
        "Private context drift",
    )
    require(
        all(
            p.read_bytes() == inputs["lineage_bytes"][key] for key, p in inputs["lineage_paths"].items()
        ),
        "Historical lineage bytes drift",
    )


def post_runtime(args, inputs, raw, db, users, corpus):
    annotations = validate_annotation(
        inputs["suite"], inputs["bytes"]["overlay"], inputs["manifest"], ROOT, inputs["bytes"]["suite"]
    )
    diagnostics = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for task, row in zip(inputs["tasks"], raw["results"][arm], strict=True):
            user = users[task["user"]]
            phases = {
                "candidates": row["source_candidates"],
                "admitted": row["retrieved_evidence"],
                "context": row["retrieved_evidence"],
            }
            source_row = supplement.score_context(
                db, user, task, phases["candidates"], phases["context"], corpus
            )
            original = diagnose(task, phases)
            revised = rescore(
                db, user, task, source_row, corpus, annotations["task_records"][task["id"]]
            )
            diagnostics[arm].append({"task_id": task["id"], "original": original, "revised": revised})
            provisional_score(
                task,
                row["answer_payload"],
                [],
                0,
                row["latency_ms"],
                execution_error_kind=None,
                scenario_events={},
            )
            # Proxy diagnostics cannot establish semantic correctness.
            row.update(
                strict_task_success=None,
                answer_correct=None,
                task_success=None,
                early_stop=None,
                extra_calls_after_sufficient_evidence=None,
            )
    raw["proxy_diagnostics"] = diagnostics


def proxy_summary(raw):
    result = {arm: {"original": [], "revised": []} for arm in ARMS}
    for arm in ARMS:
        for item in raw.get("proxy_diagnostics", {}).get(arm, []):
            for name in ("original", "revised"):
                diagnostic = item.get(name) or {}
                phase = diagnostic.get("phases", {}).get("context") or {}
                result[arm][name].append(
                    {
                        "task_id": safe_id(item["task_id"]),
                        "total": number(phase.get("total")),
                        "delivered": number(phase.get("delivered")),
                        "all_delivered": phase.get("all_delivered")
                        if type(phase.get("all_delivered")) is bool
                        else None,
                    }
                )
    paired = {arm: {name: [] for name in ("original", "revised")} for arm in ARMS[1:]}
    diagnostics = raw.get("proxy_diagnostics", {})
    baseline = {r["task_id"]: r for r in diagnostics.get("no_supp", [])}
    for arm in ARMS[1:]:
        for item in diagnostics.get(arm, []):
            for name in ("original", "revised"):
                first = (
                    (baseline.get(item["task_id"], {}).get(name) or {}).get("phases", {}).get("context")
                )
                second = (item.get(name) or {}).get("phases", {}).get("context")
                observed = (
                    first is not None
                    and second is not None
                    and all(
                        type(phase.get("matched_reference_ids")) is list for phase in (first, second)
                    )
                )
                gains = losses = None
                if observed:
                    old, new = set(first["matched_reference_ids"]), set(second["matched_reference_ids"])
                    gains = sorted(digest(ref) for ref in new - old)
                    losses = sorted(digest(ref) for ref in old - new)
                paired[arm][name].append(
                    {
                        "task_id": safe_id(item["task_id"]),
                        "gained_reference_sha256": gains,
                        "lost_reference_sha256": losses,
                    }
                )
    result.update(paired=paired, semantic_recall=None, adoption=None)
    return result


def execute(args, reservation):
    from app.clients import Search
    from app.db import SessionLocal

    run_id = safe_id(reservation["run_id"])
    runtime = private_root(args.private_root) / run_id
    runtime.mkdir(parents=True, exist_ok=False)
    raw = {
        "version": "policy-ablation-replay-dev-v1",
        "run_id": run_id,
        "split": "dev",
        "subset": False,
        "results": {arm: [] for arm in ARMS},
        "corpus_sha256": None,
        "scope_validation": deepcopy(SCOPE_VALIDATION),
        "runtime_completed": False,
    }
    stage = "preflight"
    inputs = registration = registration_bytes = None
    interrupted = False
    error = None
    drift = {"runtime": None, "bindings": None}
    try:
        # Establish the trusted denominator before any other preflight dependency.
        selection = json.loads(Path(SELECTION).read_bytes())
        trusted_ids = [safe_id(item["id"]) for item in selection["items"]]
        require(
            selection["split"] == "dev"
            and selection["count"] == 47
            and len(trusted_ids) == len(set(trusted_ids)) == 47,
            "Trusted full Dev selection required",
        )
        raw["results"] = {arm: [missing(tid, run_id) for tid in trusted_ids] for arm in ARMS}
        atomic_json(runtime / "raw.json", raw)
        suite_bytes = Path(args.suite).read_bytes()
        suite = json.loads(suite_bytes)
        tasks = validate_suite(suite, selection, ROOT)
        require(
            [t["id"] for t in tasks]
            == [t["id"] for t in json.loads(Path(SELECTION).read_bytes())["items"]],
            "Trusted task order differs",
        )
        raw["results"] = {arm: [missing(t["id"], run_id) for t in tasks] for arm in ARMS}
        atomic_json(runtime / "raw.json", raw)
        inputs = load_inputs(args)
        registration_bytes = Path(args.registration).read_bytes()
        registration = json.loads(registration_bytes)
        require(registration == registration_for(args), "Frozen registration differs")
        raw["registration"] = deepcopy(registration)
        atomic_json(runtime / "registration.json", registration)
        cfg = settings()
        corpus = inputs["parent"]["identity"]["corpus"]
        index = inputs["parent"]["identity"]["source_index"]
        raw["corpus_sha256"] = digest(corpus)
        source = Search(index=index["index"])
        with SessionLocal() as db:
            db.execute(text("SET TRANSACTION READ ONLY"))
            users = unchanged_scope(db, source, corpus, index)
            try:
                for offset, (task, original) in enumerate(
                    zip(inputs["tasks"], inputs["originals"], strict=True)
                ):
                    rotation = offset % 3
                    order = ARMS[rotation:] + ARMS[:rotation]
                    stage = "source"
                    snapshot_matches(args, inputs, registration, registration_bytes)
                    users = quick_scope(db, source, corpus, index)
                    user = users[task["user"]]

                    def source_postcheck(callback, primary):
                        try:
                            quick_scope(db, source, corpus, index)
                        except BaseException as exc:
                            drift["runtime"] = True
                            raw.setdefault("source_scope_post_errors", []).append(
                                {
                                    "stage": stage,
                                    "callback": callback,
                                    "type": type(exc).__name__,
                                    "reason": str(exc),
                                    "primary_type": type(primary).__name__ if primary else None,
                                    "primary_reason": str(primary) if primary else None,
                                }
                            )
                            if primary is None:
                                raise
                            # Cancellation owns the stop; the failed audit remains recorded.

                    def checked_action(callback, action, *a, **kw):
                        quick_scope(db, source, corpus, index)
                        primary = None
                        try:
                            return action(*a, **kw)
                        except (TaskCancelled, KeyboardInterrupt) as exc:
                            primary = exc
                            raise
                        finally:
                            source_postcheck(callback, primary)

                    # Authenticate complete parent cached candidates with the
                    # existing rehydration API, then the frozen Stage1 pool/seed.
                    checked_action(
                        "rehydrate",
                        replay.rehydrate_task,
                        db,
                        user,
                        inputs["parent_rows"][offset],
                        corpus,
                    )
                    seed = checked_action(
                        "reauthorize",
                        reauthorized,
                        db,
                        user,
                        original["contexts"]["baseline"],
                        corpus,
                    )
                    replay.validate_context(
                        seed, inputs["chosen"]["rows"]["baseline"][offset]["telemetry"]["context_tokens"]
                    )
                    candidates = checked_action(
                        "reauthorize",
                        reauthorized,
                        db,
                        user,
                        original["pools"]["baseline"],
                        corpus,
                    )
                    contract = contract_for(db, user, task["goal"])
                    models = make_source_models(
                        before=lambda: quick_scope(db, source, corpus, index),
                        after=lambda primary: source_postcheck("model", primary),
                    )
                    transport = authorized_search(
                        db,
                        user,
                        contract,
                        cfg=source_configuration(cfg),
                        models=models,
                        backend=make_backend(source),
                        corpus=corpus,
                    )
                    require(
                        digest(transport.cfg.model_dump(mode="json"))
                        == registration["source_configuration_sha256"],
                        "Actual source configuration drift",
                    )

                    def live_reauthorize(rows):
                        quick_scope(db, source, corpus, index)
                        primary = None
                        try:
                            return transport.reauthorize(rows)
                        except (TaskCancelled, KeyboardInterrupt) as exc:
                            primary = exc
                            raise
                        finally:
                            source_postcheck("reauthorize", primary)

                    def live_scope(original_question, query):
                        quick_scope(db, source, corpus, index)
                        primary = None
                        try:
                            return transport.validate_scope(original_question, query)
                        except (TaskCancelled, KeyboardInterrupt) as exc:
                            primary = exc
                            raise
                        finally:
                            source_postcheck("validate_scope", primary)

                    def live_search(search_args):
                        quick_scope(db, source, corpus, index)
                        primary = None
                        try:
                            return transport.search(search_args)
                        except (TaskCancelled, KeyboardInterrupt) as exc:
                            primary = exc
                            raise
                        finally:
                            source_postcheck("search", primary)

                    def live_ranker(query, rows):
                        quick_scope(db, source, corpus, index)
                        primary = None
                        try:
                            return ranker(query, rows)
                        except (TaskCancelled, KeyboardInterrupt) as exc:
                            primary = exc
                            raise
                        finally:
                            source_postcheck("ranker", primary)

                    # Retain the frozen transport's exact physical event lists.
                    live_search.events = getattr(transport.search, "events", None)
                    live_search.backend_events = getattr(transport.search, "backend_events", None)

                    source_started = time.monotonic()
                    try:
                        component = run_components(
                            contract,
                            seed,
                            candidates,
                            models=models,
                            reauthorize=live_reauthorize,
                            search=live_search,
                            validate_scope=live_scope,
                            ranker=live_ranker,
                            cfg=transport.cfg,
                            order=order,
                        )
                    except BaseException as exc:
                        component = getattr(exc, "component_result", None)
                        if component is not None:
                            component.trace["scope_validation"] = deepcopy(SCOPE_VALIDATION)
                            component.trace["source_group_wall_ms"] = (
                                time.monotonic() - source_started
                            ) * 1000
                            preserve_components(raw, offset, component)
                            atomic_json(runtime / "raw.json", raw)
                        raise
                    component.trace["scope_validation"] = deepcopy(SCOPE_VALIDATION)
                    component.trace["source_group_wall_ms"] = (time.monotonic() - source_started) * 1000
                    preserve_components(raw, offset, component)
                    atomic_json(runtime / "raw.json", raw)
                    if component.trace["status"] == "interrupted":
                        raise TaskCancelled("Source component interrupted")
                    require(
                        not component.trace.get("errors")
                        and all(
                            component.arms[a]["evidence"] is not None
                            and raw["results"][a][offset]["source_cost"]["transport_budget_matched"]
                            is True
                            for a in ARMS
                        ),
                        "Source execution or budget failure",
                    )
                    for arm in ARMS:
                        checked_action(
                            "reauthorize",
                            reauthorized,
                            db,
                            user,
                            component.arms[arm]["evidence"],
                            corpus,
                        )
                    users = quick_scope(db, source, corpus, index)
                    snapshot_matches(args, inputs, registration, registration_bytes)
                    stage = "generation"
                    for arm in order:
                        row = raw["results"][arm][offset]
                        user = quick_scope(db, source, corpus, index)[task["user"]]
                        snapshot_matches(args, inputs, registration, registration_bytes)
                        evidence = checked_action(
                            "reauthorize",
                            reauthorized,
                            db,
                            user,
                            row["retrieved_evidence"],
                            corpus,
                        )
                        replay.validate_context(evidence, replay.evidence_tokens(evidence))
                        generation = GenerationModels()
                        usage = None
                        began = time.monotonic()
                        row.update(
                            stage=stage,
                            status="execution_failed",
                            not_observed=False,
                            execution_error_kind="GenerationExecutionFailure",
                        )
                        try:
                            payload, usage = checked_action(
                                "generate_answer",
                                generate_answer,
                                task["goal"],
                                evidence,
                                models=generation,
                                cfg=cfg,
                                check_conflict=user.tenant_id not in cfg.conflict_check_disabled_tenants,
                                protocol="shared_contract",
                                enforce_zero_embedding=True,
                            )
                            row["answer_payload"] = payload
                            checked_action("reauthorize", reauthorized, db, user, evidence, corpus)
                            quick_scope(db, source, corpus, index)
                            snapshot_matches(args, inputs, registration, registration_bytes)
                            row.update(status="completed", execution_error_kind=None)
                        except BaseException as exc:
                            row["private_error"] = {
                                "type": type(exc).__name__,
                                "reason": str(exc),
                                "stage": stage,
                            }
                            if isinstance(exc, (TaskCancelled, KeyboardInterrupt)):
                                row["status"] = "interrupted"
                            raise
                        finally:
                            if generation.answer_budget is not None:
                                usage = (usage or {}) | {
                                    "business_budget": generation.answer_budget.snapshot()
                                }
                            cost = cost_projection(generation, usage)
                            budget = (usage or {}).get("business_budget")
                            # Existing cost projection covers transports; this
                            # experiment adds the previously omitted ranker role.
                            cost["transport_budget_matched"] = (
                                cost["transport_budget_matched"] is True
                                and budget is not None
                                and all(
                                    budget["calls"].get(role, {}).get("attempted", 0) == 0
                                    for role in ("policy", "search", "embedding", "ranker", "judge")
                                )
                            )
                            row.update(
                                generation_usage=usage,
                                generation_cost=cost,
                                usage=cost["physical_generation"],
                                transport_budget_matched=cost["transport_budget_matched"],
                                latency_ms=(time.monotonic() - began) * 1000,
                            )
                            atomic_json(runtime / "raw.json", raw)
                        require(row["transport_budget_matched"] is True, "Generation budget mismatch")
                raw["runtime_completed"] = True
                drift["runtime"] = False
                snapshot_matches(args, inputs, registration, registration_bytes)
                stage = "post_runtime_evaluation"
                post_runtime(args, inputs, raw, db, users, corpus)
                stage = "complete"
            finally:
                try:
                    unchanged_scope(db, source, corpus, index)
                    drift["runtime"] = bool(raw.get("source_scope_post_errors"))
                except Exception:
                    drift["runtime"] = True
    except BaseException as exc:
        interrupted = isinstance(exc, (KeyboardInterrupt, TaskCancelled))
        error = {"stage": stage, "type": "interrupted" if interrupted else "execution_error"}
        raw["private_error"] = {"stage": stage, "type": type(exc).__name__, "reason": str(exc)}
    finally:
        if inputs is not None and registration is not None and registration_bytes is not None:
            try:
                snapshot_matches(args, inputs, registration, registration_bytes)
                drift["bindings"] = False
            except Exception:
                drift["bindings"] = True
        rows = raw["results"]
        complete = len(rows[ARMS[0]]) == 47 and all(
            r["status"] == "completed" and r["transport_budget_matched"] is True
            for arm in ARMS
            for r in rows[arm]
        )
        valid = (
            raw["runtime_completed"]
            and complete
            and error is None
            and all(v is False for v in drift.values())
        )
        status = (
            "complete"
            if valid
            else "interrupted"
            if interrupted
            else "invalid_drift"
            if any(v is True for v in drift.values())
            else "failed"
        )
        raw.update(valid=valid, status=status, error=error, drift=drift, stage=stage)
        atomic_json(runtime / "raw.json", raw)
        if inputs is not None:
            suite = deepcopy(inputs["suite"])
            suite["arms"] = list(ARMS)
            atomic_json(runtime / "suite.json", suite)
            if valid:
                atomic_json(runtime / "review-packet.json", answer_review_packet(raw, suite))
        public = {
            "version": raw["version"],
            "run_id": run_id,
            "split": "dev",
            "subset": False,
            "valid": valid,
            "status": status,
            "error": error,
            "drift": drift,
            "stage": stage,
            "runtime_completed": raw["runtime_completed"],
            "selected_task_ids": [r["task_id"] for r in rows[ARMS[0]]],
            "rows": {arm: [public_row(r) for r in rows[arm]] for arm in ARMS},
            "strict_task_success": None,
            "semantic_recall": None,
            "answer_correctness": None,
            "semantic_review": "pending_fresh_named_blind_review",
            "adoption": None,
            "phases": ["source", "generation"],
            "cache_latency_comparison": "shared_cache_no_independent_arm_latency_claim",
            "source_tasks": [
                {
                    "task_id": safe_id(tid),
                    "source_group_wall_ms": number(record["trace"].get("source_group_wall_ms")),
                    "cached_pair_owner": "unknown",
                    "cache_latency": "not_independent",
                    "source_order": [enum(arm, ARMS) for arm in record["trace"].get("order", [])],
                }
                for tid, record in raw.get("source_tasks", {}).items()
            ],
            "proxy_diagnostics": proxy_summary(raw),
            "proxy_basis": "source_delivery_only_not_semantic_correctness",
            "private_raw_sha256": sha((runtime / "raw.json").read_bytes()),
            "identity": public_identity(raw.get("registration")),
            "registration_bytes_sha256": sha(registration_bytes)
            if registration_bytes is not None
            else None,
        }
        _publish(args.output, public)
    return public


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "execute"))
    for key in (
        *INPUT_ARGUMENTS,
        "context_root",
        "historical_code_root",
        "registration",
        "private_root",
        "output",
    ):
        parser.add_argument("--" + key.replace("_", "-"), required=True)
    args = parser.parse_args(argv)
    if args.action == "register":
        destination = private_root(args.registration)
        require(not Path(args.output).exists(), "Fresh output required")
        registration = registration_for(args)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _publish(destination, registration)
        print(json.dumps({"status": "registered", "registration_sha256": sha(destination.read_bytes())}))
    else:
        require(not Path(args.output).exists(), "Fresh output required")
        reservation = reserve_output(args.output)
        result = execute(args, reservation)
        print(json.dumps({"status": result["status"], "valid": result["valid"]}))
        return 0 if result["valid"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
