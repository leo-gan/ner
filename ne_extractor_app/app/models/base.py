from abc import abstractmethod


class BaseNEModel:
    def __init__(self, model_name: str, entities: set[str] = None):
        self.name = model_name
        self.entities = entities or {"persons", "organizations", "locations"}

    def extract_ne(self, text: str) -> dict[str, str]:
        """Extract named entities from text.

        Args:
            text (str): Input text

        Returns:
            dict: Extracted named entities
        """
        res = self.extract_ne_batch([text])
        return {k: v[0] for k, v in res.items()}

    @abstractmethod
    def extract_ne_batch(self, texts: list[str]) -> dict[str, list[str]]:
        """Extract named entities from a batch of texts.

        Args:
            texts (list[str]): List of input texts

        Returns:
            dict: Extracted named entities for each text
        """
        ...
