from typing import Annotated

import typer

from labelr.evaluate import visualize as _visualize

app = typer.Typer()


@app.command()
def visualize(
    hf_repo_id: Annotated[
        str,
        typer.Option(
            ...,
            help="Hugging Face repository ID of the trained model. "
            "A `predictions.parquet` file is expected in the repo. Revision can be specified "
            "by appending `@<revision>` to the repo ID.",
        ),
    ],
    dataset_name: Annotated[
        str | None, typer.Option(..., help="Name of the FiftyOne dataset to create.")
    ] = None,
    persistent: Annotated[
        bool,
        typer.Option(
            ...,
            help="Whether to make the FiftyOne dataset persistent (i.e., saved to disk).",
        ),
    ] = False,
):
    if dataset_name is None:
        dataset_name = hf_repo_id.replace("/", "-").replace("@", "-")

    _visualize(
        hf_repo_id=hf_repo_id,
        dataset_name=dataset_name,
        persistent=persistent,
    )
