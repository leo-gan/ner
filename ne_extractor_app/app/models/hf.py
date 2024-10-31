from app.models.base import BaseNEModel
from transformers import BertForTokenClassification, BertTokenizer, Pipeline, pipeline

DEFAULT_NER_MODEL = "dslim/bert-base-NER"  # Pre-trained BERT model for NER


def load_model(model_name: str = None) -> Pipeline:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name)
    return pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )


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


class HuggingFaceNERModel(BaseNEModel):
    def __init__(self, model_name: str = None, entities: set[str] = None):
        self.model_name = model_name or DEFAULT_NER_MODEL
        super().__init__(self.model_name, entities)
        self.ner_pipeline = load_model(self.model_name)

    def extract_ne_batch(self, texts: list[str]) -> dict[str, list[str]]:
        predictions = self.ner_pipeline(texts)
        return format_entities(predictions)
