"""Commands to manage datasets local datasets and export between platforms
(Label Studio, HuggingFace Hub, local dataset,...)."""

import shutil
from pathlib import Path
from typing import Annotated

import typer
from openfoodfacts.utils import get_logger

app = typer.Typer(no_args_is_help=True)

logger = get_logger(__name__)


IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".avif",
    ".svg",
    ".tiff",
    ".tif",
    ".bmp",
    ".heic",
]


@app.command()
def shard_image_dir(
    dataset_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to the dataset directory containing images",
            exists=True,
            file_okay=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Path to the output directory for sharded images"),
    ],
    shard_size: Annotated[
        int,
        typer.Option(help="Number of items in each shard"),
    ] = 1000,
):
    """Shard a local image dataset directory into smaller subdirectories.

    This command looks for image files in the dataset directory and
    creates subdirectories with a specified number of items per shard
    in the output directory.

    The original dataset directory is not modified.
    """

    i = 0
    for file_path in dataset_dir.glob("**/*"):
        if file_path.suffix not in IMAGE_EXTENSIONS:
            continue

        shard_idx = i // shard_size
        shard_dir = output_dir / f"{shard_idx:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_path, shard_dir / file_path.name)
        i += 1
