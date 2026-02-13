from pathlib import Path
from typing import Annotated

import typer
from openfoodfacts.utils import get_logger

from . import typer_description
from ...config import config, check_required_field
from .common import check_label_studio_api_key

app = typer.Typer(no_args_is_help=True)

logger = get_logger(__name__)


@app.command()
def add(
    tag: Annotated[
        str,
        typer.Option(help="Value of the tag to add"),
    ],
    task_id_file: Annotated[
        Path,
        typer.Option(
            help="Path of a text file containing IDs of samples",
            exists=True,
            dir_okay=False,
            file_okay=True,
            readable=True,
        ),
    ],
    project_id: Annotated[
        int | None, typer.Option(help=typer_description.LABEL_STUDIO_PROJECT_ID)
    ] = config.label_studio_project_id,
    api_key: Annotated[
        str | None, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
    ] = config.label_studio_api_key,
    label_studio_url: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_URL)
    ] = config.label_studio_url,
):
    """Add a tag to Label Studio tasks.

    Tags are used to filter tasks on Label Studio using arbitrary criteria.
    A plain text file containing IDs of Label Studio tasks (one per line) must be
    provided with --task-id-file. The provided tag will be added in the `tags` data
    field for each task.
    """

    from label_studio_sdk.client import LabelStudio

    check_label_studio_api_key(api_key)
    check_required_field("--project-id", project_id)

    typer.echo(
        f"Adding tag '{tag}' to tasks with IDs read from file '{task_id_file}'",
        err=True,
    )
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    updated = 0

    with task_id_file.open("r") as f:
        for line in f:
            task_id = line.strip()
            if not task_id:
                continue
            task = ls.tasks.get(task_id)

            if task:
                data = task.data
                data.setdefault("tags", [])
                if not isinstance(data["tags"], list):
                    raise ValueError(
                        f"tags field for task '{task.id}' is not a list: '{data['tags']}'"
                    )
                if tag in data["tags"]:
                    continue
                task.data["tags"].append(tag)
                ls.tasks.update(task_id, data=task.data)
                updated += 1
            else:
                typer.echo(f"Task '{task.id}' not found", err=True)

    typer.echo(f"Number of tasks updated: {updated}", err=True)
