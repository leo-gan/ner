import pytest

from ne_extractor_app.app.models.nltk import NltkNERModel


@pytest.fixture
def ner_model():
    return NltkNERModel()


def test_extract_ne_batch(ner_model):
    texts = [
        "John Doe is from New York or Scottsdale.",
        "I am from Los Angeles.",
        "Some random text without NEs.",
    ]
    result = ner_model.extract_ne_batch(texts)
    expected = {
        "locations": ["New York", "Los Angeles", ""],
        "organizations": ["Doe", "", "NEs"],
        "persons": ["John", "", ""],
    }
    assert result == expected


def test_extract_ne(ner_model):
    text = "John Smith is from New York or Scottsdale. He works at ABC Inc."
    result = ner_model.extract_ne(text)
    expected = {"locations": "New York", "organizations": "ABC Inc", "persons": "John"}
    assert result == expected
