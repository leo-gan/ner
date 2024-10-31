from collections import defaultdict

from app.models.base import BaseNEModel
from app.models.hf import HuggingFaceNERModel
from app.models.nltk import NltkNERModel
from app.models.spacy import SpacyNERModel


class EnsembleNERModel(BaseNEModel):
    def __init__(
        self,
        model_name: str = "Ensemble",
        weights: dict[str, float] = None,
        threshold: float = 0.6,
    ):
        self.model_name = model_name
        self.models = {
            "hf": HuggingFaceNERModel(),
            "nltk": NltkNERModel(),
            "spacy": SpacyNERModel(),
        }
        self.weights = weights or {"nltk": 0.25, "spacy": 0.35, "hf": 0.4}
        self.threshold = threshold

        super().__init__(self.model_name, entities=None)

    def extract_ne(self, text: str) -> dict[str, str]:
        model2predictions = {
            k: model.extract_ne(text) for k, model in self.models.items()
        }
        return self.combine_nes(model2predictions)

    def extract_ne_batch(self, texts: list[str]) -> dict[str, list[str]]:
        predictions = [self.extract_ne(text) for text in texts]
        res = {"persons": [], "organizations": [], "locations": []}
        for prediction in predictions:
            res["persons"].append(prediction["persons"])
            res["organizations"].append(prediction["organizations"])
            res["locations"].append(prediction["locations"])
        return res

    def combine_nes(self, model2predictions):
        result = {}
        for key in ["persons", "organizations", "locations"]:
            # Dictionary to hold scores for each unique substring
            substring_scores = defaultdict(float)

            for model in ["nltk", "spacy", "hf"]:
                substrings = (
                    model2predictions[model][key].split(";")
                    if model2predictions[model][key]
                    else []
                )

                # Add the weight for each substring from the respective model
                for substring in substrings:
                    if substring:
                        substring_scores[substring] += self.weights[model]

            # Filter substrings with a cumulative score greater than the threshold
            combined_substrings = [
                substring
                for substring, score in substring_scores.items()
                if score > self.threshold
            ]
            result[key] = ";".join(combined_substrings)
        return result
