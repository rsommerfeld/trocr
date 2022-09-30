import typer

from .main import TrocrPredictor, main_train, main_validate

cli = typer.Typer(name="TrOCR")


@cli.command()
def train(local_model: bool = False):
    main_train(local_model)


@cli.command()
def validate(local_model: bool = True):
    main_validate(local_model)


@cli.command()
def predict(image_paths: list[str], local_model: bool = True):
    predictions = TrocrPredictor(local_model).predict_for_image_paths(image_paths)
    for path, (prediction, confidence) in zip(image_paths, predictions):
        print(f"Path:\t\t{path}\nPrediction:\t{prediction}\nConfidence:\t{confidence}\n")
