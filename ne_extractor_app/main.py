from pathlib import Path

import typer
from app.evaluation import evaluate
from app.models.ensemble import EnsembleNERModel
from app.name_normalizer import Normalizer
from app.ne_extractor import NEExtractor
from pydantic import BaseModel

app_typer = typer.Typer()
model = EnsembleNERModel()
normalizer = Normalizer()
extractor = NEExtractor(ne_model=model, normalizer=normalizer)


class ExtractionInput(BaseModel):
    text: str


class BatchExtractionInput(BaseModel):
    texts: list[str]


class EvaluationInput(BaseModel):
    evaluation_file_path: Path
    result_output_dir: Path


@app_typer.command()
def extract_ne_cli(text: str):
    result = extractor.extract_ne(text)
    typer.echo(result)


@app_typer.command()
def extract_ne_batch_cli(texts: list[str]):
    results = extractor.extract_ne_batch(texts)
    typer.echo(results)


@app_typer.command()
def evaluate_cli(evaluation_file_path: Path, result_output_dir: Path):
    result = evaluate(evaluation_file_path, result_output_dir)
    typer.echo(result)


if __name__ == "__main__":
    app_typer()
