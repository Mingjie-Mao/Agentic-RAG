"""Synthetic source contracts: no evaluation data or external writes."""

from types import SimpleNamespace
import importlib
import importlib.util
import json

import pytest

import app.retrieval as retrieval


def document(key, source="Publisher P", published="2026-01-03"):
    return SimpleNamespace(id=key, active_version_id=f"v-{key}",
                           metadata_json={"source": source, "published_at": published})


def test_source_mentions_keeps_occurrences_and_legacy_routing_deduplicates():
    docs = [document("jan"), document("feb", published="2026-02-08")]
    question = "Publisher P report on January 3, 2026 and Publisher P report on January 3, 2026"
    assert hasattr(retrieval, "source_mentions"), "occurrence helper must exist"
    mentions = retrieval.source_mentions(question, docs)
    assert len(mentions) == 2
    assert [(m["start"], m["end"], m["mention"]) for m in mentions] == [
        (0, 11, "publisher p"), (42, 53, "publisher p")]
    assert all(m["documents"] == docs for m in mentions)
    assert retrieval.route_sources(question, docs, strict_dates=True) == [{
        "mention": "publisher p", "clause": "Publisher P report on January 3, 2026 and",
        "dated": True, "bounded": False, "key": ["jan"], "versions": ["v-jan"]}]


def contract(question, docs=()):
    assert importlib.util.find_spec("app.supplement_contract"), "pure contract module must exist"
    return importlib.import_module("app.supplement_contract").build_supplement_contract(question, docs)


def test_distinct_publication_dates_resolve_independent_source_slots():
    question = "Publisher P report on January 3, 2026 and Publisher P report on February 8, 2026"
    result = contract(question, [document("jan"), document("feb", published="2026-02-08")])
    assert result.mode == "independent"
    assert len(result.slots) == len(result.source_requests) == 2
    jan, feb = result.source_requests
    assert jan.document_ids == ("jan",) and jan.version_ids == ("v-jan",)
    assert feb.document_ids == ("feb",) and feb.version_ids == ("v-feb",)
    assert jan.publication_constraints == (("eq", "2026-01-03"),)
    assert feb.publication_constraints == (("eq", "2026-02-08"),)
    assert jan.occurrence_id != feb.occurrence_id
    assert question[jan.start:jan.end] == jan.original_span == "Publisher P"
    assert question[jan.clause_start:jan.clause_end] == jan.query_clause
    assert result.slots[0].source_request_ids == (jan.request_id,)


def test_coordinated_dated_reports_inherit_one_explicit_publisher_with_separate_clauses():
    question = ("Did Publisher P report on January 3, 2026, that Acme paid billions, "
                "and then on February 8, 2026, reported a lawsuit?")
    result = contract(question, [document("jan"), document("feb", published="2026-02-08")])
    assert result.mode == "independent"
    assert len(result.source_requests) == len(result.slots) == 2
    jan, feb = result.source_requests
    assert jan.occurrence_id == feb.occurrence_id
    assert (jan.start, jan.end, jan.original_span) == (feb.start, feb.end, feb.original_span)
    assert jan.original_span == "Publisher P"
    assert jan.document_versions == (("jan", "v-jan"),)
    assert feb.document_versions == (("feb", "v-feb"),)
    assert jan.publication_constraints == (("eq", "2026-01-03"),)
    assert feb.publication_constraints == (("eq", "2026-02-08"),)
    assert jan.clause_end <= feb.clause_start
    for request, slot in zip(result.source_requests, result.slots):
        assert question[request.clause_start:request.clause_end] == request.query_clause == slot.query
        assert slot.source_request_ids == (request.request_id,)
    assert "billions" in jan.query_clause and "lawsuit" not in jan.query_clause
    assert "lawsuit" in feb.query_clause and "billions" not in feb.query_clause


def test_coordinated_dated_reports_keep_missing_date_separate():
    question = ("Publisher P report on January 3, 2026: Acme paid billions, "
                "and then on February 8, 2026, reported a lawsuit")
    jan, feb = contract(question, [document("jan")]).source_requests
    assert jan.document_ids == ("jan",) and jan.resolution == "supported"
    assert feb.document_ids == () and feb.resolution == "missing"
    assert feb.reason == "no_matching_publication_date"


def test_coordinated_dated_reports_allow_named_subject_complementizers_in_both_clauses():
    question = ("Did Publisher P report on January 3, 2026, that Acme paid billions, "
                "and then on February 8, 2026, reported that Beta faced a lawsuit?")
    result = contract(question, [document("jan"), document("feb", published="2026-02-08")])
    assert result.mode == "independent" and result.reason == "explicit_source_requests"
    jan, feb = result.source_requests
    assert jan.occurrence_id == feb.occurrence_id
    assert jan.document_versions == (("jan", "v-jan"),)
    assert feb.document_versions == (("feb", "v-feb"),)
    assert jan.publication_constraints == (("eq", "2026-01-03"),)
    assert feb.publication_constraints == (("eq", "2026-02-08"),)
    assert "billions" in jan.query_clause and "lawsuit" not in jan.query_clause
    assert "lawsuit" in feb.query_clause and "billions" not in feb.query_clause
    assert question[feb.clause_start:feb.clause_end] == feb.query_clause


@pytest.mark.parametrize("content", [
    "that company faced a lawsuit", "that Beta faced its lawsuit", "that Beta faced their lawsuit",
])
def test_coordinated_continuation_keeps_demonstratives_and_anaphora_unknown(content):
    question = ("Did Publisher P report on January 3, 2026, that Acme paid billions, "
                f"and then on February 8, 2026, reported {content}?")
    result = contract(question, [document("jan"), document("feb", published="2026-02-08")])
    assert result.mode == "unknown" and result.reason == "ambiguous_anaphora"
    assert [r.document_ids for r in result.source_requests] == [("jan",), ("feb",)]


@pytest.mark.parametrize("continuation", [
    "and then on February 8, 2026, Acme faced a lawsuit",
    "and then on February 8, 2026, a lawsuit was reported",
    "and then on February 8, 2026, Publisher Q reported a lawsuit",
    "or then on February 8, 2026, reported a lawsuit",
    "and on February 8, 2026, reported a lawsuit",
])
def test_other_coordinated_or_event_grammars_do_not_inherit_publisher(continuation):
    question = f"Publisher P report on January 3, 2026: Acme paid billions, {continuation}"
    result = contract(question, [document("jan"), document("feb", published="2026-02-08"),
                                 document("q", source="Publisher Q", published="2026-02-08")])
    assert not any(r.resolution == "supported" and r.document_ids == ("feb",)
                   for r in result.source_requests)
    assert result.source_requests[0].resolution == "unknown"


@pytest.mark.parametrize("date_text", [
    "Feb 8, 2026", "2026-02-30", "February 8, 2026 at 12:00", "February 8, 2026 at 5 PM",
    "February 8, 2026 or February 9, 2026",
])
def test_coordinated_invalid_or_ambiguous_dates_cannot_bind_documents(date_text):
    question = ("Publisher P report on January 3, 2026: Acme paid billions, "
                f"and then on {date_text}, reported a lawsuit")
    result = contract(question, [document("jan"), document("feb", published="2026-02-08")])
    assert all(r.resolution == "unknown" for r in result.source_requests
               if "lawsuit" in r.query_clause)
    assert not any(r.document_ids == ("feb",) for r in result.source_requests)


def test_coordinated_event_date_in_report_content_is_not_a_publication_constraint():
    question = ("Publisher P report on January 3, 2026: Acme paid billions, "
                "and then on February 8, 2026, reported a lawsuit filed on March 9, 2026")
    jan, feb = contract(question, [document("jan"), document("feb", published="2026-02-08")]).source_requests
    assert jan.resolution == "supported"
    assert feb.resolution == "unknown" and feb.reason == "ambiguous_date_attachment"
    assert feb.document_ids == ()


def test_coordinated_reports_cannot_inherit_ambiguous_publisher_alias():
    question = ("Publisher P report on January 3, 2026: Acme paid billions, "
                "and then on February 8, 2026, reported a lawsuit")
    docs = [document("jan", source="Publisher P - Technology"),
            document("feb", source="Publisher P - Business", published="2026-02-08")]
    result = contract(question, docs)
    assert all(r.resolution == "unknown" and r.reason == "ambiguous_date_attachment"
               and not r.document_ids for r in result.source_requests)


@pytest.mark.parametrize("published", [None, "nonsense", "2026-02-30", "2026-02-08garbage"])
def test_coordinated_reports_keep_invalid_publication_metadata_unknown(published):
    question = ("Publisher P report on January 3, 2026: Acme paid billions, "
                "and then on February 8, 2026, reported a lawsuit")
    result = contract(question, [document("jan"), document("feb", published=published)])
    assert all(r.resolution == "unknown" and r.reason == "missing_publication_metadata"
               and not r.document_ids for r in result.source_requests)


def test_coordinated_report_dates_resolve_even_with_genuine_anaphora():
    question = ("Publisher P report on January 3, 2026, that Acme paid its bills, "
                "and then on February 8, 2026, reported a lawsuit")
    result = contract(question, [document("jan"), document("feb", published="2026-02-08")])
    assert result.mode == "unknown" and result.reason == "ambiguous_anaphora"
    assert [r.document_ids for r in result.source_requests] == [("jan",), ("feb",)]


def test_coordinated_report_demonstrative_that_stays_unknown():
    question = ("Publisher P report on January 3, 2026, that company paid billions, "
                "and then on February 8, 2026, reported a lawsuit")
    result = contract(question, [document("jan"), document("feb", published="2026-02-08")])
    assert result.mode == "unknown" and result.reason == "ambiguous_anaphora"
    assert [r.document_ids for r in result.source_requests] == [("jan",), ("feb",)]


def test_coordinated_event_only_first_date_does_not_inherit_publisher():
    question = ("Publisher P report the incident on January 3, 2026, "
                "and then on February 8, 2026, reported a lawsuit")
    result = contract(question, [document("jan"), document("feb", published="2026-02-08")])
    assert len(result.source_requests) == 1
    assert result.source_requests[0].publication_constraints == ()
    assert result.source_requests[0].document_ids == ("jan", "feb")


@pytest.mark.parametrize("conjunction", ["and", "or"])
def test_coordinated_reports_do_not_inherit_shared_publishers(conjunction):
    question = (f"Publisher P {conjunction} Publisher Q report on January 3, 2026: Acme paid billions, "
                "and then on February 8, 2026, reported a lawsuit")
    result = contract(question, [document("jan"), document("q", source="Publisher Q")])
    assert all(r.resolution == "unknown" and not r.document_ids for r in result.source_requests)


def test_unavailable_explicit_date_never_widens():
    result = contract("Publisher P report on February 8, 2026", [document("jan")])
    request = result.source_requests[0]
    assert request.resolution == "missing"
    assert request.reason == "no_matching_publication_date"
    assert request.document_ids == request.version_ids == ()


def test_repeated_same_publisher_same_date_keeps_two_slots():
    result = contract("Publisher P report on January 3, 2026 and Publisher P report on January 3, 2026", [document("jan")])
    assert len(result.source_requests) == len(result.slots) == 2
    assert len({s.slot_id for s in result.slots}) == 2
    assert all(r.document_ids == ("jan",) for r in result.source_requests)


@pytest.mark.parametrize("tail", [
    "say the incident on February 8, 2026 occurred?",
    "report the event on February 8, 2026?",
    "describe the policy effective on February 8, 2026?",
])
def test_event_and_version_effective_dates_are_not_publication_constraints(tail):
    request = contract("Did Publisher P " + tail, [document("jan")]).source_requests[0]
    assert request.publication_constraints == ()
    assert request.document_ids == ("jan",)


@pytest.mark.parametrize("word,selected", [("before", "earlier"), ("after", "later")])
def test_new_bounds_are_strict_and_legacy_bounds_stay_inclusive(word, selected):
    docs = [document("earlier", published="2026-01-02"), document("equal"), document("later", published="2026-01-04")]
    question = f"Did Publisher P report {word} January 3, 2026?"
    request = contract(question, docs).source_requests[0]
    assert request.document_ids == (selected,)
    legacy = retrieval.route_sources(question, docs, strict_dates=True)
    assert set(legacy[0]["key"]) == {"equal", selected}


@pytest.mark.parametrize("published", [None, "", "nonsense", "2026-02-30"])
def test_missing_or_invalid_publication_metadata_stays_unknown(published):
    request = contract("Publisher P report on January 3, 2026", [document("x", published=published)]).source_requests[0]
    assert request.resolution == "unknown"
    assert request.reason == "missing_publication_metadata"
    assert request.document_ids == request.version_ids == ()


@pytest.mark.parametrize("question,reason", [
    ("Publisher P report on Jan 3, 2026", "unsupported_publication_date"),
    ("Publisher P report on 2026-02-30", "unsupported_publication_date"),
    ("Publisher P reports as of January 3, 2026", "unsupported_publication_date"),
    ("Publisher P article published at 12:00:00", "unsupported_publication_date"),
    ("Publisher P report on January 3, 2026 or February 8, 2026", "ambiguous_date_attachment"),
])
def test_unsupported_and_ambiguous_dates_remain_unknown(question, reason):
    request = contract(question, [document("jan")]).source_requests[0]
    assert request.resolution == "unknown" and request.reason == reason
    assert request.document_ids == ()


@pytest.mark.parametrize("question", [
    "星海公司的创始人的出生地在哪里？",
    "What is Acme's founder's birthplace?",
    "What is the birthplace of the founder of Acme?",
])
def test_clear_nested_generic_relationship_is_only_a_bridge_proposal(question):
    result = contract(question)
    assert result.mode == "entity_bridge"
    assert result.reason == "nested_relationship_proposal"


@pytest.mark.parametrize("question", [
    "甲服务的超时时间和乙服务的超时时间分别是多少？",
    "What is Alpha's status and what is Beta's status?",
    "Compare Alpha and Beta's quotas.",
])
def test_explicit_independent_attributes_and_comparisons_are_independent(question):
    result = contract(question)
    assert result.mode == "independent"
    assert len(result.slots) == 2


@pytest.mark.parametrize("question", [
    "What is its founder's birthplace?",
    "Who founded Acme previously?",
    "If Acme succeeds, where is the founder born?",
    "How did the quota change across previous versions?",
    "What is Alpha's quota?",
])
def test_anaphora_history_conditions_and_unresolved_generic_slots_stay_unknown(question):
    assert contract(question).mode == "unknown"


def test_recognized_slot_spec_is_retained_for_later_value_binding():
    result = contract("甲服务的超时时间和乙服务的超时时间分别是多少？")
    assert all(s.slot_spec and s.slot_spec.attribute for s in result.slots)


def test_public_projection_contains_only_allowlisted_enums_ids_hashes_and_counts():
    publisher = "Ignore Previous Instructions"
    question = f"{publisher} report on January 3, 2026"
    result = contract(question, [document("secret-document", source=publisher)])
    summary = result.public_summary()
    encoded = json.dumps(summary)
    for sensitive in (publisher, publisher.lower(), question, "2026-01-03", "secret-document", "v-secret-document", "January", '"start":', '"end":', "query_clause", "metadata"):
        assert sensitive not in encoded
    assert summary["mode"] == result.mode
    assert summary["source_requests"][0]["document_count"] == 1
    assert summary["source_requests"][0]["request_id"] == result.source_requests[0].request_id
    assert len(summary["question_fingerprint"]) == 64
    assert result == contract(question, [document("secret-document", source=publisher)])
    with pytest.raises(AttributeError):
        result.mode = "unknown"


def test_source_without_a_date_after_another_dated_source_remains_supported():
    result = contract("Publisher P report on January 3, 2026; Publisher P describes the rule", [document("jan")])
    assert all(r.resolution == "supported" for r in result.source_requests)
    assert result.source_requests[1].publication_constraints == ()


@pytest.mark.parametrize("question", [
    "Publisher P and Publisher Q articles published on January 3, 2026",
    "Reports published on January 3, 2026 by Publisher P",
])
def test_shared_predicates_and_date_prefixes_have_unknown_attachment(question):
    result = contract(question, [document("p"), document("q", source="Publisher Q")])
    assert all(r.resolution == "unknown" for r in result.source_requests)
    assert all(r.reason == "ambiguous_date_attachment" for r in result.source_requests)
    assert all(r.document_ids == () for r in result.source_requests)


@pytest.mark.parametrize("source", ["If Alpha", "History News", "Its Founder", "Compare Alpha"])
def test_publisher_names_are_masked_before_query_classification(source):
    result = contract(f"{source} reports on January 3, 2026 and {source} reports on January 3, 2026", [document("jan", source=source)])
    assert result.mode == "independent"
    assert all(r.resolution == "supported" for r in result.source_requests)


def test_source_clause_keeps_recognized_slot_spec():
    result = contract("Publisher P report on January 3, 2026: 甲服务的超时时间是多少？", [document("jan")])
    assert result.slots[0].slot_spec.attribute == "超时时间"


def test_two_bounds_for_one_occurrence_keep_distinct_source_slots():
    result = contract("Publisher P reported before January 3, 2026 and reported before February 8, 2026", [document("early", published="2026-01-02"), document("jan")])
    assert len(result.source_requests) == len(result.slots) == 2
    a, b = result.source_requests
    assert a.occurrence_id == b.occurrence_id
    assert a.document_ids == ("early",)
    assert b.document_ids == ("early", "jan")


def test_malformed_metadata_with_valid_date_prefix_is_unknown():
    request = contract("Publisher P report on January 3, 2026", [document("x", published="2026-01-03garbage")]).source_requests[0]
    assert request.resolution == "unknown"


def test_public_projection_rejects_arbitrary_strings_in_enum_and_id_fields():
    from dataclasses import replace

    original = contract("Publisher P report on January 3, 2026", [document("jan")])
    injected = replace(original, reason="LEAK", mode="LEAK", source_requests=(
        replace(original.source_requests[0], request_id="LEAK", occurrence_id="LEAK", reason="LEAK", resolution="LEAK"),))
    assert "LEAK" not in json.dumps(injected.public_summary())


def test_single_explicit_source_request_is_independent():
    assert contract("Publisher P article published on January 3, 2026", [document("jan")]).mode == "independent"


def test_single_recognized_subject_attribute_is_independent():
    assert contract("甲服务的超时时间是多少？").mode == "independent"


def presence_row(cid='x', text='Product A 的超时阈值为 9 秒。', **extra):
    return dict(chunk_id=cid, document_id=cid, version_id=cid+'-v1', source_sha256=cid+'-sha',
                title='', text=text, **extra)


def test_literal_presence_and_missing_are_distinct():
    from app.supplement_contract import assess_slot_presence, build_supplement_contract
    c = build_supplement_contract('Product A 的超时阈值是多少？', [])
    assert assess_slot_presence(c, [presence_row()])['slots'][0]['state'] == 'supported'
    assert assess_slot_presence(c, [])['slots'][0]['state'] == 'missing'


def test_subject_attribute_cannot_cross_bind():
    from app.supplement_contract import assess_slot_presence, build_supplement_contract
    c = build_supplement_contract('Product A 的超时阈值是多少？', [])
    rows = [presence_row(text='Product B 的超时阈值为 9 秒。'),
            presence_row('y', 'Product A 是一个产品。')]
    assert assess_slot_presence(c, rows)['slots'][0]['state'] == 'missing'
    rows[0]['title'] = 'Product A and Product B'
    assert assess_slot_presence(c, rows)['slots'][0]['state'] != 'supported'


def test_generic_prose_stays_unknown():
    from app.supplement_contract import assess_slot_presence, build_supplement_contract
    c = build_supplement_contract("What was the article's position on privacy?", [])
    assert assess_slot_presence(c, [presence_row(text='Privacy was mentioned.')])['slots'][0]['state'] == 'unknown'


def test_stale_source_binding_and_invalid_metadata_never_support():
    from app.supplement_contract import assess_slot_presence, build_supplement_contract
    c = build_supplement_contract('Product A 的超时阈值是多少？', [])
    for key in ['version_id', 'source_sha256', 'document_id']:
        r = presence_row()
        r[key] = ''
        assert assess_slot_presence(c, [r])['slots'][0]['state'] == 'unknown'
    assert assess_slot_presence(c, [presence_row(authorized=False)])['slots'][0]['state'] == 'unknown'


def test_ambiguous_or_negated_literal_is_not_support():
    from app.supplement_contract import assess_slot_presence, build_supplement_contract
    c = build_supplement_contract('Product A 的超时阈值是多少？', [])
    for text in ['Product A 的超时阈值为 9 秒或 10 秒。', 'Product A 的超时阈值不是 9 秒。']:
        assert assess_slot_presence(c, [presence_row(text=text)])['slots'][0]['state'] == 'unknown'


def test_same_publisher_wrong_date_cannot_supply_source_slot():
    from app.supplement_contract import assess_slot_presence
    c = contract('Publisher P report published February 8, 2026',
                 [document('jan'), document('feb', published='2026-02-08')])
    r = presence_row('jan', 'Privacy was discussed.')
    r['version_id'] = 'v-jan'
    p = assess_slot_presence(c, [r])['slots'][0]
    assert p['source'] == p['date'] == p['state'] == 'missing'
    assert p['scoped_ids'] == []


def test_frozen_source_sha_detects_drift():
    from app.supplement_contract import assess_slot_presence
    d = document('x')
    d.source_sha256 = 'frozen-sha'
    c = contract('Publisher P report', [d])
    r = presence_row()
    r['version_id'] = 'v-x'
    assert assess_slot_presence(c, [r])['invalid_row_count'] == 1
    r['source_sha256'] = 'frozen-sha'
    assert assess_slot_presence(c, [r])['invalid_row_count'] == 0


def test_table_header_and_exact_row_bind_literal():
    from app.supplement_contract import assess_slot_presence
    c = contract('Product A 的超时阈值是多少？')
    r = presence_row(text='| 产品 | 超时阈值 |\n| --- | --- |\n| Product A | 9 秒 |\n| Product B | 10 秒 |')
    assert assess_slot_presence(c, [r])['slots'][0]['state'] == 'supported'


def test_condition_unknown_and_inactive_are_explicit():
    from app.supplement_contract import assess_slot_presence
    c = contract('回滚服务的阈值是多少？如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。')
    assert all(r['state'] == 'unknown' for r in assess_slot_presence(c, [])['slots'])
    p = assess_slot_presence(c, [presence_row(text='回滚服务的阈值为 9 秒。')])
    assert any(r['state'] == 'inactive' for r in p['slots'])


def test_public_presence_is_hash_only():
    from app.supplement_contract import assess_slot_presence, public_presence
    c = contract('Product A 的超时阈值是多少？')
    raw = assess_slot_presence(c, [presence_row('SECRET')])
    public = json.dumps(public_presence(raw))
    assert 'SECRET' not in public and 'Product A' not in public and '9 秒' not in public


def test_explicit_publication_clock_is_not_silently_discarded():
    c = contract('Publisher P article published January 3, 2026 at 11:57 AM', [document('jan')])
    assert c.source_requests[0].resolution == 'unknown'
    assert c.source_requests[0].document_ids == ()


def test_nonstring_document_identity_remains_unknown():
    d = document('x')
    d.id = 123
    assert contract('Publisher P report', [d]).source_requests[0].resolution == 'unknown'


def test_condition_cannot_borrow_actor_from_title():
    from app.supplement_contract import assess_slot_presence
    c = contract('回滚服务的阈值是多少？如果超过 10 秒，再查日志服务的阈值；否则查通知服务的阈值。')
    r = presence_row(text='通知服务的阈值为 9 秒。')
    r['title'] = '回滚服务与通知服务'
    assert not any(s['state'] == 'inactive' for s in assess_slot_presence(c, [r])['slots'])


def test_nested_entity_literal_cannot_certify_top_level_attribute():
    from app.supplement_contract import assess_slot_presence
    c = contract('A数据库的承运商的阈值是多少？')
    p = assess_slot_presence(c, [presence_row(text='A数据库的阈值为 9 秒。')])
    assert p['slots'][0]['state'] == 'unknown'
    assert p['gate']


def test_unresolved_pronoun_literal_cannot_close_gate():
    from app.supplement_contract import assess_slot_presence
    c = contract('its 的阈值是多少？')
    p = assess_slot_presence(c, [presence_row(text='its 的阈值为 9 秒。')])
    assert p['slots'][0]['state'] == 'unknown'
    assert p['gate']
