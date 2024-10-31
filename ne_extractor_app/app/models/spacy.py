# import pandas as pd
import spacy
from app.models.base import BaseNEModel


def format_entities(
    predictions,
    entity_map={"PER": "persons", "ORG": "organizations", "LOC": "locations"},
):
    entity_results = {value: [] for value in entity_map.values()}
    for pred in predictions:
        entity_sets = {value: set() for value in entity_map.values()}
        for entity in pred:
            entity_group = entity["entity_group"]
            if entity_group in entity_map:
                entity_sets[entity_map[entity_group]].add(entity["word"])
        for key in entity_map.values():
            entity_results[key].append(";".join(sorted(entity_sets[key])))
    return entity_results


class SpacyNERModel(BaseNEModel):
    def __init__(self, model_name: str = "en_core_web_sm", entities: set[str] = None):
        self.model_name = model_name
        self.nlp = spacy.load(model_name)
        super().__init__(self.model_name, entities)

    def extract_ne_batch(self, texts: list[str]) -> dict[str, list[str]]:
        result = {"persons": [], "organizations": [], "locations": []}
        for doc in self.nlp.pipe(texts):
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

            # Join entities with ';' or set as empty string if no entities found
            result["persons"].append(";".join(persons) if persons else "")
            result["organizations"].append(
                ";".join(organizations) if organizations else ""
            )
            result["locations"].append(";".join(locations) if locations else "")
        return result
