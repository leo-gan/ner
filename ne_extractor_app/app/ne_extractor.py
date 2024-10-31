from app.models.base import BaseNEModel
from app.name_normalizer import Normalizer


class NEExtractor:
    def __init__(self, ne_model: BaseNEModel, normalizer: Normalizer):
        self.ne_model = ne_model
        self.normalizer = normalizer

    def extract_ne(self, text: str) -> dict[str, str]:
        ness = self.ne_model.extract_ne(text)
        ness["organizations"] = self.normalizer.normalize_names(ness["organizations"])
        return ness

    def extract_ne_batch(self, texts: list[str]) -> dict[str, list[str]]:
        nes = self.ne_model.extract_ne_batch(texts)
        nes["organizations"] = [
            self.normalizer.normalize_names(ness) for ness in nes["organizations"]
        ]
        return nes
