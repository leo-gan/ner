import pytest

from ne_extractor_app.app.name_normalizer import Normalizer


@pytest.fixture
def normalizer():
    return Normalizer()


def test_compact_name(normalizer):
    name = "John Doe, Inc."
    expected = "doe john"
    assert normalizer.compact_name(name) == expected


def test_is_included(normalizer):
    normalized_to_original = {"doe john": "John Doe"}
    normalized = "john"
    assert normalizer.is_included(normalized_to_original, normalized) == "doe john"

    normalized = "john doe"
    assert normalizer.is_included(normalized_to_original, normalized) == "doe john"

    normalized = "john smith"
    assert normalizer.is_included(normalized_to_original, normalized) is None


def test_normalize_names(normalizer):
    name_list = (
        "John Doe, Inc.;Jane Smith LLC;John Smith;Jane Smith;John Smith;John Doe"
    )
    expected = "John Doe, Inc.;Jane Smith LLC;John Smith"
    assert normalizer.normalize_names(name_list) == expected
