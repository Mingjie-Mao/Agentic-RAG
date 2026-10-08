from app.retrieval_queries import retrieval_facets


def test_explicit_between_creates_two_article_queries():
    parts = retrieval_facets(
        "Between the TechCrunch report on the trial and the subsequent TechCrunch report on fraud allegations, was the portrayal different?"
    )
    assert len(parts) == 2
    assert 'trial' in parts[0] and 'fraud allegations' in parts[1]


def test_from_to_chain_creates_three_queries():
    parts = retrieval_facets(
        "Has the regulator's focus changed from addressing Amazon competition concerns to facilitating dialogue over Meta subscriptions to probing X moderation practices?"
    )
    assert len(parts) == 3
    assert all(len(part.split()) >= 4 for part in parts)


def test_compound_actions_keep_entity_anchor():
    parts = retrieval_facets(
        'Who is the individual associated with FTX that was alleged to have committed fraud, advised another trader on withdrawals, and made a decision to use customer funds?'
    )
    assert len(parts) == 3 and all('FTX' in part for part in parts)


def test_short_or_chinese_question_keeps_existing_retrieval():
    assert retrieval_facets('What is the RPO?') == []
    assert retrieval_facets('甲报告和乙报告中的时间与阈值分别是多少？') == []
