from app.source_fact_guards import date_role, inclusive_end, literal_field


def test_number_and_identifier_boundaries_reject_partial_values():
    assert not literal_field('50','The limit is 250%')
    assert not literal_field('1','Value is 16')
    assert not literal_field('1','Value is 1.5')
    assert not literal_field('1','Value is -1')
    assert not literal_field('Alex','Alexander said yes')
    assert literal_field('2%','失败阈值为2%。')
    assert literal_field('event_id','Use event_id for deduplication.')


def test_numeric_unit_cannot_be_silently_dropped():
    assert not literal_field('15','Policy limit is 15 minutes.',complete_number=True)
    assert literal_field('15 minutes','Policy limit is 15 minutes.',complete_number=True)
    assert literal_field('15','There are 15 tickets.',complete_number=True)


def test_date_role_uses_nearest_label_in_either_order():
    quote='Policy effective September 1, 2026; published August 20, 2026.'
    assert date_role('September 1, 2026',quote)=='effective'
    assert date_role('August 20, 2026',quote)=='publication'
    assert date_role('2026-09-01','发布日期为2026-08-20，2026-09-01起生效。')=='effective'


def test_until_is_not_assumed_inclusive_without_explicit_support():
    assert inclusive_end('August 31, 2026','Effective January 1 through August 31, 2026.')
    assert inclusive_end('2026-08-31','自2026-01-01起生效，至2026-08-31止。')
    assert not inclusive_end('September 1, 2026','Effective until September 1, 2026.')
    assert inclusive_end('September 1, 2026','Effective until September 1, 2026, inclusive.')
