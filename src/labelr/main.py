from typing import Annotated

import typer
from openfoodfacts.utils import get_logger

from labelr.apps import datasets as dataset_app
from labelr.apps import directus as directus_app
from labelr.apps import evaluate as evaluate_app
from labelr.apps import google_batch as google_batch_app
from labelr.apps import hugging_face as hf_app
from labelr.apps import label_studio as ls_app
from labelr.apps import train as train_app
from labelr import config as _config

app = typer.Typer(pretty_exceptions_show_locals=False)

logger = get_logger()


@app.command()
def predict(
    model_name: Annotated[
        str, typer.Option(help="Name of the object detection model to run")
    ],
    label_names: Annotated[list[str], typer.Argument(help="List of label names")],
    image_url: Annotated[str, typer.Option(help="URL of the image to process")],
    triton_uri: Annotated[
        str, typer.Option(help="URI (host+port) of the Triton Inference Server")
    ],
    image_size: Annotated[
        int, typer.Option(help="Size of the image the model expects")
    ] = 640,
    threshold: Annotated[float, typer.Option(help="Detection threshold")] = 0.5,
    triton_model_version: str = "1",
):
    """Predict objects in an image using an object detection model served by
    Triton."""
    import typing

    from openfoodfacts.ml.object_detection import ObjectDetector
    from openfoodfacts.utils import get_image_from_url
    from PIL import Image

    model = ObjectDetector(
        model_name=model_name, label_names=label_names, image_size=image_size
    )
    image = typing.cast(Image.Image | None, get_image_from_url(image_url))

    if image is None:
        logger.error("Failed to download image from URL: %s", image_url)
        raise typer.Abort()

    output = model.detect_from_image(
        image,
        triton_uri=triton_uri,
        model_version=triton_model_version,
        threshold=threshold,
    )
    results = output.to_list()

    for result in results:
        typer.echo(result)


@app.command()
def config(name: str, value: str):
    """Set a Labelr configuration value.

    The configuration is stored in a JSON file at ~/.config/labelr/config.json.
    """
    typer.echo(f"Set '{name}' to '{value}'")
    _config.set_file_config(name, value)
    typer.echo(f"Configuration saved to {_config.CONFIG_PATH}")


app.add_typer(
    ls_app.app,
    name="ls",
    help="Manage Label Studio projects (create, import data, etc.).",
)
app.add_typer(
    hf_app.app,
    name="hf",
    help="Manage Hugging Face Datasets repositories.",
)
app.add_typer(
    dataset_app.app,
    name="datasets",
    help="Manage datasets (convert, export, check, etc.)",
)
app.add_typer(
    train_app.app,
    name="train",
    help="Train models.",
)
app.add_typer(
    evaluate_app.app,
    name="evaluate",
    help="Visualize and evaluate trained models.",
)
app.add_typer(
    google_batch_app.app,
    name="google-batch",
    help="Generate datasets and launch batch jobs on Google Gemini.",
)
app.add_typer(
    directus_app.app, name="directus", help="Manage directus collections and items."
)

if __name__ == "__main__":
    app()
