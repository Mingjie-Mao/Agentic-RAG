"""Synthetic source review must be exhaustive, bound and text-free when published."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts.dev_evidence_annotations import (
    build_manifest, compile_annotation, references_for_task, validate_annotation,
)
from scripts.research_review import digest
from scripts.retrieval_quality_metrics import supporting_facts
from scripts.multihop_retrieval_eval import document_id


def sha(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    text = "PRIVATE SOURCE DISCLAIMER.\nThe current limit is 30 minutes, excluding outages.\n"
    path = tmp_path / "source.md"
    path.write_text(text)
    source_sha = sha(path.read_bytes())
    title = "Synthetic Source"
    did = document_id(title)
    parent = {"id": "GE-source", "document_id": did, "source_path": "source.md",
              "source_sha256": source_sha, "quote": text,
              "locator": {"start_char": 0, "end_char": len(text)},
              "effective_from": "2026-01-01", "fact_ids": ["limit"]}
    tasks, records, reviews = [], [], []
    for i in range(47):
        task = {"id": f"D{i:02}", "question": "PRIVATE QUESTION", "category": "comparison",
                "required_facts": [{"id": "limit", "document_ids": [did]}],
                "gold_answer": "PRIVATE ANSWER", "gold_evidence": [deepcopy(parent)],
                "requires_version": 36 <= i < 41}
        if 26 <= i < 36:
            task["upstream_gold_evidence"] = [
                {"title": title, "fact": f"Original unique reference {j}: the limit is thirty minutes."}
                for j in range(3 if i < 32 else 2)]
        if i >= 41:
            task["gold_evidence"] = []
            task["required_facts"] = []
        kind = ("active_native" if i < 26 else "active_upstream" if i < 36 else
                "history_not_applicable" if i < 41 else "no_references")
        atoms, decisions = [], []
        if i < 26:
            start = text.index("The current")
            end = len(text) - 1
            atoms.append({"id": "NA-" + digest([task['id'], did, source_sha, start, end])[:20],
                          "document_id": did, "source_path": "source.md", "source_sha256": source_sha,
                          "start_char": start, "end_char": end, "quote": text[start:end],
                          "supports_fact_ids": ["limit"], "reason": "PRIVATE ATOM REASON"})
        if kind == "active_upstream":
            decisions = [{"reference_id": f["id"], "original_index": j, "label": "keep",
                          "source_sha256": source_sha, "reason": "PRIVATE REFERENCE REASON",
                          "source_quote_found": True, "proposed_replacement": None}
                         for j, f in enumerate(supporting_facts(task))]
        record = {"task_id": task["id"], "input_sha256": digest(task), "applicability": kind,
                  "native_atoms": atoms, "upstream_decisions": decisions,
                  "answer_audit": {"label": "correct" if i < 36 else "not_applicable",
                                   "reason": "PRIVATE AUDIT REASON", "proposed_answer": None}}
        review = {"task_id": task["id"], "input_sha256": digest(task), "verdict": "approved",
                  "reason": "PRIVATE REVIEW REASON",
                  "native_atom_reviews": [{"id": a['id'], "label": "keep", "reason": "PRIVATE"}
                                          for a in atoms],
                  "upstream_reviews": [{"reference_id": a['reference_id'], "label": "keep",
                                         "reason": "PRIVATE"} for a in decisions],
                  "answer_audit": {"label": record['answer_audit']['label'], "reason": "PRIVATE"}}
        tasks.append(task)
        records.append(record)
        reviews.append(review)
    suite = {"split": "dev", "version": "synthetic-dev-v1", "tasks": tasks}
    suite_bytes = json.dumps(suite).encode()
    draft = {"version": "dev-evidence-annotation-draft-v2", "split": "dev",
             "suite_bytes_sha256": sha(suite_bytes), "task_ids47": [t['id'] for t in tasks],
             "provenance": {"reviewer_type": "model", "model": "gpt-6-astra", "role": "proposer",
                            "independent_of_retrieval_results": True,
                            "independent_of_authored_annotation": False, "human": False},
             "audit_scope": {}, "source_inputs": [], "records": records}
    draft_sha = sha(json.dumps(draft).encode())
    review = {"version": "dev-evidence-source-review-v2", "draft_bytes_sha256": draft_sha,
              "suite_bytes_sha256": sha(suite_bytes),
              "provenance": {"reviewer_type": "model", "model": "gpt-6-astra",
                             "role": "source_reviewer", "independent_of_authored_annotation": True,
                             "independent_of_retrieval_results": True, "human": False,
                             "model_family_independent": False}, "records": reviews}
    return suite, draft, review, tmp_path, suite_bytes, draft_sha


def compile_inputs(inputs):
    suite, draft, review, root, suite_bytes, draft_sha = inputs
    return compile_annotation(suite, draft, review, root, draft_sha256=draft_sha,
                              suite_bytes_sha256=sha(suite_bytes))


def test_roundtrip_preserves_original_and_public_whitelist(inputs):
    suite = inputs[0]
    before = deepcopy(suite)
    overlay = compile_inputs(inputs)
    overlay_bytes = json.dumps(overlay, ensure_ascii=False).encode()
    manifest = build_manifest(overlay, sha(overlay_bytes))
    loaded = validate_annotation(suite, overlay_bytes, manifest, inputs[3], inputs[4])
    assert list(loaded['task_records']) == [t['id'] for t in suite['tasks']]
    assert manifest['counts']['eligible_tasks'] == 36
    assert suite == before
    public = json.dumps([manifest, loaded['public_ledger']])
    for secret in ('PRIVATE', 'source.md', 'Synthetic Source', 'The current', 'minutes', 'question'):
        assert secret not in public
    assert len(references_for_task(suite['tasks'][0], loaded['task_records']['D00'])) == 1


@pytest.mark.parametrize('mutation', ['split', 'count', 'order', 'task', 'suite_sha', 'draft_sha',
                                       'source_sha', 'span_bool', 'quote', 'path', 'duplicate',
                                       'fact', 'verdict', 'differing_review', 'review_count'])
def test_invalid_binding_or_review_fails(inputs, mutation):
    suite, draft, review, root, suite_bytes, draft_sha = inputs
    atom = draft['records'][0]['native_atoms'][0]
    if mutation == 'split':
        suite['split'] = 'core'
    elif mutation == 'count':
        draft['records'].pop()
    elif mutation == 'order':
        draft['records'].reverse()
    elif mutation == 'task':
        suite['tasks'][0]['question'] = 'changed'
    elif mutation == 'suite_sha':
        draft['suite_bytes_sha256'] = '0' * 64
    elif mutation == 'draft_sha':
        review['draft_bytes_sha256'] = '0' * 64
    elif mutation == 'source_sha':
        (root / 'source.md').write_text('changed')
    elif mutation == 'span_bool':
        atom['start_char'] = True
    elif mutation == 'quote':
        atom['quote'] = '30 minutes'
    elif mutation == 'path':
        atom['source_path'] = '../source.md'
    elif mutation == 'duplicate':
        draft['records'][0]['native_atoms'].append(deepcopy(atom))
    elif mutation == 'fact':
        atom['supports_fact_ids'] = ['invented']
    elif mutation == 'verdict':
        review['records'][0]['verdict'] = 'changes_required'
    elif mutation == 'differing_review':
        review['records'][0]['native_atom_reviews'][0]['label'] = 'unclear'
    elif mutation == 'review_count':
        review['records'].pop()
    with pytest.raises(ValueError):
        compile_inputs(inputs)


def test_symlink_escape_rejected(inputs, tmp_path):
    root = inputs[3]
    outside = tmp_path.parent / 'annotation-outside-source.md'
    outside.write_bytes((root / 'source.md').read_bytes())
    (root / 'source.md').unlink()
    (root / 'source.md').symlink_to(outside)
    with pytest.raises(ValueError, match='path'):
        compile_inputs(inputs)


def test_upstream_exhaustive_original_ids_and_indices(inputs):
    decision = inputs[1]['records'][26]['upstream_decisions']
    decision[0]['original_index'] = 1
    with pytest.raises(ValueError):
        compile_inputs(inputs)


def test_unclear_and_rejected_do_not_change_original_eligibility(inputs):
    draft, review = inputs[1:3]
    draft['records'][0]['native_atoms'][0]['label'] = 'unclear'
    review['records'][0]['native_atom_reviews'][0]['label'] = 'unclear'
    for decision, reviewed in zip(draft['records'][26]['upstream_decisions'],
                                  review['records'][26]['upstream_reviews']):
        decision['label'] = reviewed['label'] = 'reject'
    overlay = compile_inputs(inputs)
    first = overlay['records'][0]
    assert references_for_task(inputs[0]['tasks'][0], first) == []
    assert len(references_for_task(inputs[0]['tasks'][0], first, include_unclear=True)) == 1
    manifest = build_manifest(overlay, 'a' * 64)
    assert manifest['counts']['eligible_tasks'] == 36
    assert manifest['counts']['annotation_pending_tasks'] == 1


def test_manifest_integrity_and_nested_private_fields(inputs):
    inputs[1]['audit_scope']['private'] = {'deep': 'PRIVATE SENTINEL'}
    overlay = compile_inputs(inputs)
    raw = json.dumps(overlay).encode()
    manifest = build_manifest(overlay, sha(raw))
    assert 'PRIVATE SENTINEL' not in json.dumps(manifest)
    bad = deepcopy(manifest)
    bad['private'] = {'deep': 'PRIVATE SENTINEL'}
    with pytest.raises(ValueError):
        validate_annotation(inputs[0], raw, bad, inputs[3], inputs[4])
    with pytest.raises(ValueError):
        validate_annotation(inputs[0], raw + b' ', manifest, inputs[3], inputs[4])


def test_pending_task_keeps_eligibility_and_null_completion(inputs):
    from scripts.dev_evidence_annotations import diagnose_revised, summarize_revised
    draft, review = inputs[1:3]
    for decision, checked in zip(draft['records'][26]['upstream_decisions'],
                                 review['records'][26]['upstream_reviews']):
        decision['label'] = checked['label'] = 'unclear'
    overlay = compile_inputs(inputs)
    task, record = inputs[0]['tasks'][26], overlay['records'][26]
    row = diagnose_revised(task, {p: [] for p in ('candidates', 'admitted', 'context')}, record)
    assert row['eligible'] and row['status'] == 'measured'
    assert row['phases']['context']['delivered'] == 0
    assert row['phases']['context']['all_delivered'] is None
    summary = summarize_revised([row])['by_basis']['upstream_word4_proxy']
    assert summary['phases']['context']['proxy_coverage'] is None
    assert summary['phases']['context']['completion_pending_tasks'] == 1
    assert summary['annotation_counts']['unclear_reference_count'] == 3
    missing = diagnose_revised(task, None, record)
    assert missing['phases']['context'] is None and missing['losses'] is None


def test_actual_review_independence_is_not_overstated(inputs):
    review = inputs[2]
    review['provenance'].update(model='GPT-6', independent_of_retrieval_results=False,
        independent_of_authored_annotation=False, model_identity_precision='family_from_session_instructions',
        serving_model_id=None)
    overlay = compile_inputs(inputs)
    manifest = build_manifest(overlay, 'b' * 64)
    assert not manifest['review_provenance']['independent_of_retrieval_results']
    assert manifest['model_identity_precision'] == 'family_from_session_instructions'
