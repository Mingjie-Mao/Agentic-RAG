"""Human gold-source labels must cover the frozen source text without edits."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_required_document_review import validate


def test_required_document_review_rejects_missing_or_changed_sources():
    source = {'title': 'Report A', 'fact': 'The threshold is 2%',
              'label': None, 'reason': ''}
    original = [{'id': 'Q1', 'question': 'What threshold?', 'gold_answer': '2%',
                 'evidence': [source]}]
    reviewed = {'reviewer': 'Independent reviewer', 'reviewer_type': 'human',
                'items': [{**original[0], 'evidence': [{**source, 'label': 'required',
                                                       'reason': 'Contains the answer'}]}]}
    validate(reviewed, original)
    reviewed['items'][0]['evidence'][0]['fact'] = 'The threshold is 3%'
    with pytest.raises(ValueError, match='Source text changed'):
        validate(reviewed, original)
    reviewed['items'] = []
    with pytest.raises(ValueError, match='incomplete'):
        validate(reviewed, original)
