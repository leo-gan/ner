import pytest

from ne_extractor_app.app.models.ensemble import EnsembleNERModel


@pytest.fixture
def ner_model():
    return EnsembleNERModel()


def test_extract_ne_batch(ner_model):
    texts = [
        "John Doe is from New York or Scottsdale.",
        "I am from Los Angeles and works in the Civil Department.",
        "Some random text without NEs.",
    ]
    result = ner_model.extract_ne_batch(texts)
    expected = {
        "locations": ["New York;Scottsdale", "Los Angeles", ""],
        "organizations": [
            "",
            "Civil Department",
            "",
        ],  # 'the' was removed by normalization
        "persons": ["John Doe", "", ""],
    }
    assert result == expected


def test_extract_ne(ner_model):
    text = "John Smith is from New York or Scottsdale. He works at ABC Inc."
    result = ner_model.extract_ne(text)
    expected = {
        "locations": "New York;Scottsdale",
        "organizations": "ABC Inc",  # '.' was removed by normalization
        "persons": "John Smith",
    }
    assert result == expected
