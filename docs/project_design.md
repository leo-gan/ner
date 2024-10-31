### Design Document for Named Entity Recognition Project

#### Project Overview
The project aims to develop a `Named Entity Recognition` (`NER`) system. 
The system will extract entities such 
as `persons`, `organizations`, and `locations` from text inputs. The project includes 
integration and unit tests to ensure the functionality and reliability of the NER system.

#### Components TODO

1. **BaseNEModel Class**
   - Abstract base class for NER models.
   - It defines the interface for extracting named entities from the text.

2. **HuggingFaceNERModel Class**
   - Inherits from `BaseNEModel`.
   - Uses Hugging Face's BERT model for NER.
   - Implements methods to load the model, predict entities, and format the results.

3. **Model Loading and Prediction**
   - `load_model`: Loads the pre-trained BERT model and tokenizer.
   - `predict_entities`: Uses the `Hugging Face pipeline` to predict entities in the text.

4. **Entity Formatting**
   - `format_entities`: Formats the raw predictions into a structured dictionary.

5. **Testing**
   - Integration tests to verify the end-to-end functionality of the NER system.
   - Unit tests to validate individual components and methods.

#### Class Definitions

**BaseNEModel Class**
```python
class BaseNEModel:
    def __init__(self, model_name: str, entities: set[str] = None):
        self.name = model_name
        self.entities = entities or {"persons", "organizations", "locations"}

    def extract_ne(self, text: str) -> dict[str, str]:
        res = self.extract_ne_batch([text])
        return {k: v[0] for k, v in res.items()}

    @abstractmethod
    def extract_ne_batch(self, texts: list[str]) -> dict[str, list[str]]:
        pass
```

**HuggingFaceNERModel Class**
```python
from transformers import BertTokenizer, BertForTokenClassification, pipeline, Pipeline

class HuggingFaceNERModel(BaseNEModel):
    ...
```

#### Testing Strategy

**Integration Tests**
- Verify the end-to-end functionality of the main classes.
- Ensure that the model correctly extracts and formats entities from text inputs.

**Unit Tests**
- Validate individual methods and components.
- Mock external dependencies to ensure tests are fast and reliable.


#### Conclusion
This design document outlines the structure and components of the NER system, including the `BaseNEModel` and `HuggingFaceNERModel` classes, model loading, prediction, entity formatting, and testing strategy. The provided code examples demonstrate how to implement and test the system effectively.