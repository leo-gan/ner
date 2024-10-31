import pytest

from ne_extractor_app.app.models.hf import HuggingFaceNERModel


@pytest.fixture
def ner_model():
    return HuggingFaceNERModel()


def test_extract_ne_batch(ner_model):
    texts = [
        "John Doe is from New York or Scottsdale.",
        "I am from Los Angeles.",
        "Some random text without NEs.",
    ]
    result = ner_model.extract_ne_batch(texts)
    expected = {
        "persons": ["John Doe", "", ""],
        "organizations": ["", "", ""],
        "locations": ["New York;Scottsdale", "Los Angeles", ""],
    }
    assert result == expected


def test_extract_ne(ner_model):
    text = "John Doe is from New York or Scottsdale. He works at ABC Inc."
    result = ner_model.extract_ne(text)
    expected = {
        "persons": "John Doe",
        "organizations": "ABC Inc",
        "locations": "New York;Scottsdale",
    }
    assert result == expected
