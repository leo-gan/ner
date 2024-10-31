from ne_extractor_app.app.models.hf import format_entities


def test_format_entities():
    predictions = [
        [{"entity_group": "PER", "word": "John Doe"}],
        [
            {"entity_group": "LOC", "word": "New York"},
            {"entity_group": "LOC", "word": "Scottsdale"},
        ],
        [{"entity_group": "LOC", "word": "Los Angeles"}],
    ]
    result = format_entities(predictions)
    expected = {
        "persons": ["John Doe", "", ""],
        "organizations": ["", "", ""],
        "locations": ["", "New York;Scottsdale", "Los Angeles"],
    }
    assert result == expected
