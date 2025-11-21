from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


@app.command()
def show_hf_sample(
    repo_id: Annotated[
        str,
        typer.Argument(
            ...,
            help="Hugging Face Datasets repo ID. The revision can be specified by "
            "appending `@<revision>` to the repo ID.",
        ),
    ],
    image_id: Annotated[
        str,
        typer.Argument(
            ...,
            help="ID of the image associated with the sample to display (field: `image_id`)",
        ),
    ],
    output_image_path: Annotated[
        Path | None,
        typer.Option(help="Path to save the sample image (optional)", exists=False),
    ] = None,
):
    """Display a sample from a Hugging Face Datasets repository by image ID."""
    from labelr.utils import parse_hf_repo_id

    repo_id, revision = parse_hf_repo_id(repo_id)

    from datasets import load_dataset

    ds = load_dataset(repo_id, revision=revision)

    sample = None
    for split in ds.keys():
        samples = ds[split].filter(lambda x: x == image_id, input_columns="image_id")
        if len(samples) > 0:
            sample = samples[0]
            break
    if sample is None:
        typer.echo(f"Sample with image ID {image_id} not found in dataset {repo_id}")
        raise typer.Exit(code=1)

    else:
        for key, value in sample.items():
            typer.echo(f"{key}: {value}")

        if output_image_path is not None:
            image = sample["image"]
            image.save(output_image_path)
            typer.echo(f"Image saved to {output_image_path}")
