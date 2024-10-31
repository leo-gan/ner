# Named Entity Recognition (NER) Project Plan

Here’s a plan for tackling this `Named Entity Recognition` (`NER`) project, 
ensuring considerable improvement over an out-of-the-box solution:

## 1. Problem Understanding
The task requires:

* Extracting Persons, Organizations, and Locations from text.
* Improving over basic NLP models like spaCy or Transformers out-of-the-box solutions.

## 2.Data Analysis
Download the test dataset and explore it for:
* [Optional] Text distribution (length, structure).
* Label quality (check examples of persons, organizations, and locations).
* Any potential noisy data that could affect evaluation.

## 3. Preprocessing
Text Cleaning. Do this only if it's necessary:
* Remove irrelevant text like special characters, HTML tags, etc.
* Normalize text (lowercasing, removing excessive whitespace).
Note: such cleaning does not significantly improve the performance of contemporary models.

Label Handling:
* Ensure labels for persons, organizations, and locations are in the expected format.
* Consider if additional dataset transformations are required, e.g., 
handling different label formats or entity overlaps.

## 4. Baseline Model Selection
Use an out-of-the-box model (for baseline comparison):
* `spaCy` NER or `Hugging Face's` `BERT` NER to create a baseline.
* Track binary metrics: `precision`, `recall`, and `F1` score. `Accuracy` is not generally useful 
for NER tasks since FNs are hard to label.

## 5. Evaluation
* Evaluate your baseline models using standard metrics (precision, recall, F1 score) on the evaluation dataset.
* Compare your metrics for different models.

## 6. Evaluation of State-of-the-Art Models
The best models are implemented as services, like ChatGPT, Claude, or Gemini.
To use them, we have to decide on budget.
Usage of State-of-the-Art Models is possible only if the budget allows it.

We can use SOTA models for:
* Labeling the evaluation datasets
* Labeling the datasets for fine-tuning models
* Verifying the results. Currently, this is the most desired option, since
it gives us the good result for less budget
* Directly using the model for NE recognition. This gives the best result, 
but requires the most budget.

## 7. Entity Normalization
Implement a custom entity normalization function:
* For multi-word normalization (for organizations):
  * Implement deduplication the entities with slightly different wording. For example:
    "Mount Graham International Observatory", "Mount Graham International", "Mount Graham Observatory" in one text 
    should be normalized to "Mount Graham International Observatory".
* [Optional] Vocabulary normalization (for locations, organizations):
  * Match predicted NE with a dictionary of common organizations.

## 8. Testing
Create all necessary tests:
* Unit Tests: Validate individual components and methods.
* Integration Tests: Verify the end-to-end functionality of the NER system.

## 9. Documentation & Report
Documents:
* How the solution was designed and implemented.
* How the improvement over the baseline was achieved.
* Results on the evaluation dataset.

## 12. Deployment
Package the code, model, and evaluation script in a GitHub repository, with clear instructions 
on how to run the pipeline.

## [Optional] Model Training & Fine-tuning
It works only with a good amount of additional training data.

Options:
1. Fine-tune `ChatGPT`, `Claude`, or `Gemini` models.
2. Fine-tune OS model.
3. Use methods as `Low-Rank Adaptation of Large Language Models` (`LoRA`), or `QLoRA` (Quantized LoRA) to fast fine-tuning.

