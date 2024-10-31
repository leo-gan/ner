import pytest

from ne_extractor_app.app.models.ensemble import EnsembleNERModel
from ne_extractor_app.app.name_normalizer import Normalizer
from ne_extractor_app.app.ne_extractor import NEExtractor


@pytest.fixture
def ne_model():
    return EnsembleNERModel()


@pytest.fixture
def name_normalize_l():
    return Normalizer()


@pytest.fixture
def ne_extractor_l(ne_model, name_normalize_l):
    return NEExtractor(ne_model, name_normalize_l)


def test_extract_ne_batch(ne_extractor_l):
    texts = [
        "John Doe is from New York or Scottsdale.",
        "I am from Los Angeles.",
        "Some random text without NEs.",
    ]
    result = ne_extractor_l.extract_ne_batch(texts)
    expected = {
        "persons": ["John Doe", "", ""],
        "organizations": ["", "", ""],
        "locations": ["New York;Scottsdale", "Los Angeles", ""],
    }
    assert result == expected


def test_extract_ne(ne_extractor_l):
    text = "John Doe is from New York or Scottsdale. He works at the ABC Inc."
    result = ne_extractor_l.extract_ne(text)
    expected = {
        "persons": "John Doe",
        "organizations": "",
        "locations": "New York;Scottsdale",
    }
    assert result == expected
