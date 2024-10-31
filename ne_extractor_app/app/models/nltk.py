from app.models.base import BaseNEModel
from nltk import ne_chunk, pos_tag, word_tokenize


class NltkNERModel(BaseNEModel):
    def __init__(self, model_name: str = None, entities: set[str] = None):
        self.model_name = model_name or "NLTK"
        super().__init__(self.model_name, entities)

    def extract_ne(self, text: str) -> dict[str, str]:
        # Tokenize the input text
        tokens = word_tokenize(text)

        # Get part of speech tags for the tokens
        pos_tags = pos_tag(tokens)

        # Perform Named Entity Chunking (NE chunking)
        chunks = ne_chunk(pos_tags, binary=False)

        # Extract only Persons, Organizations, Locations
        named_entities = {"persons": [], "organizations": [], "locations": []}
        ne_code2name = {
            "PERSON": "persons",
            "ORGANIZATION": "organizations",
            "GPE": "locations",
        }  # GPE = Geopolitical Entity (Locations)
        for chunk in chunks:
            if hasattr(chunk, "label"):
                entity_name = " ".join(c[0] for c in chunk)
                entity_type = chunk.label()
                if entity_type in ne_code2name:
                    named_entities[ne_code2name[entity_type]].append(
                        entity_name.strip()
                    )
        return {k: v[0] if v else "" for k, v in named_entities.items()}

    def extract_ne_batch(self, texts: list[str]) -> dict[str, list[str]]:
        predictions = [self.extract_ne(text) for text in texts]
        res = {"persons": [], "organizations": [], "locations": []}
        for prediction in predictions:
            res["persons"].append(prediction["persons"])
            res["organizations"].append(prediction["organizations"])
            res["locations"].append(prediction["locations"])
        return res
