# Extract Named Entity project

## Project Overview

It is a project to extract Persons, Organizations, and Locations entities from provided texts.

## Quick Install

### Local Installation

To install the project locally, you need to have `Python 3.8` or higher installed on your machine.
Run the following commands to install the project:

```bash
git clone https://git. TODO .git
cd Leonid-ganeline-2/ne_extractor_app
pip install -r requirements.txt
```

### Docker installation with CLI Application

If you have `Docker` installed, you can run these scripts to 
[build](https://git.. TODO ./-/blob/main/scripts/build.sh) and 
[run](https://git. TODO ./-/blob/main/scripts/run.sh) the CLI application.


### REST API Installation [TODO]


## Project Structure

```bash
root/
│
├── data  # Datasets and evaluation results
│   ├──  external/hf
│   └── original
├── docs  # Documentation
├── experiments  # Jupyter notebooks with experiments
├── ne_extractor_app/  # Main application folder
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   └── app/
├── scripts  # Scripts for building and running the Docker container
└── tests  # Unit tests and Integration tests
```

## Data Exploration TODO

- [Baseline with Hugging Face models](experiments/baseline_hf.ipynb)
- [NE recognition with State-of-the-art models](experiments/sota_llm_ner.ipynb)
- [Ensemble of models](experiments/ensemble_of_models.ipynb)
- [Baseline with NLTK package](experiments/baseline_nltk.ipynb)
- [Baseline with spaCy models](experiments/baseline_spacy.ipynb)
- [Named Entity datasets](experiments/ne_datasets.ipynb)
- [Name normalization](experiments/name_normalization.ipynb)

## Evaluation

Several models and packages were evaluated for the Named Entity Recognition task. 

### Evaluation Datasets

Datasets placed in the `data` folder.
Evaluation was done on the `CoNLL-2003` dataset, in the `data/external/hf` folder.

### Evaluation Results

The evaluation results are available for the following models in these folders:
- Hugging Face models: `hf`
- spaCy models: `spacy`
- NLTK models: `nltk`
- Ensemble of models: `ensemble`

Evaluation results are provided in two files:
- `with_scores.csv`: Contains original datasets with additional columns: 
  predicted entities, and metrics: `TP`, `FP`, `TN`, `FN`, `Precision`, `Recall`, `F1`.
- `scores.csv`: Contains aggregated scores for each model.

## Tests

See the Unit tests and Integration tests in the `tests` folder.
