from typing import Annotated
import typing

import typer
from openfoodfacts.utils import get_logger

from . import typer_description
from ...config import config, check_required_field
from .common import check_label_studio_api_key

app = typer.Typer(no_args_is_help=True)

logger = get_logger(__name__)


@app.command()
def annotate(
    label: Annotated[
        str,
        typer.Argument(
            help="The image label (=class) to assign to the images.",
        ),
    ],
    view_id: Annotated[
        int,
        typer.Argument(
            help="The Label Studio view ID to use for the annotation.",
        ),
    ],
    updated_by: Annotated[
        int | None, typer.Option(help="User ID to declare as annotator.")
    ] = None,
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
    """Annotate images in Label Studio with a given label.

    The `view_id` is mandatory, as it allows to specify which images to annotate.
    """
    from labelr.sample.classification import format_annotation_results

    import tqdm
    from label_studio_sdk.client import LabelStudio
    from label_studio_sdk.types import LseTask

    check_label_studio_api_key(api_key)
    check_required_field("--project-id", project_id)

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    for task in tqdm.tqdm(ls.tasks.list(project=project_id, view=view_id)):
        task = typing.cast(LseTask, task)
        task_id = task.id
        if task.total_annotations == 0:
            logger.info("Creating annotation for task: %s", task_id)
            result = format_annotation_results(label)
            ls.annotations.create(
                id=task_id,
                result=result,
                project=project_id,
                updated_by=updated_by,
            )
