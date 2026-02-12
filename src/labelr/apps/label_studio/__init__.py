import json
from pathlib import Path
from typing import Annotated

import typer
from openfoodfacts.utils import get_logger

from . import typer_description
from ...config import config, check_required_field
from .common import check_label_studio_api_key
from .prediction import app as prediction_app

app = typer.Typer(no_args_is_help=True)
app.add_typer(
    prediction_app, name="prediction", help="Manage predictions on Label Studio"
)

logger = get_logger(__name__)


@app.command()
def create(
    title: Annotated[str, typer.Option(help="Project title")],
    config_file: Annotated[
        Path, typer.Option(help="Path to label config file", file_okay=True)
    ],
    api_key: Annotated[
        str | None, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
    ] = config.label_studio_api_key,
    label_studio_url: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_URL)
    ] = config.label_studio_url,
):
    """Create a new Label Studio project."""
    from label_studio_sdk.client import LabelStudio

    check_label_studio_api_key(api_key)
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    label_config = config_file.read_text()

    project = ls.projects.create(title=title, label_config=label_config)
    logger.info(f"Project created: {project}")


@app.command()
def import_data(
    dataset_path: Annotated[
        Path,
        typer.Option(
            help="Path to the Label Studio dataset JSONL file", file_okay=True
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
    batch_size: Annotated[
        int, typer.Option(help="Number of tasks to import as a single batch")
    ] = 25,
):
    """Import tasks from a dataset file to a Label Studio project.

    The dataset file must be a JSONL file: it should contain one JSON object
    per line. To generate such a file, you can use the `create-dataset-file`
    command.
    """
    import more_itertools
    import tqdm
    from label_studio_sdk.client import LabelStudio

    check_label_studio_api_key(api_key)
    check_required_field("--project-id", project_id)
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    with dataset_path.open("rt") as f:
        for batch in more_itertools.chunked(
            tqdm.tqdm(map(json.loads, f), desc="tasks"), batch_size
        ):
            ls.projects.import_tasks(id=project_id, request=batch)


@app.command()
def add_split(
    train_split: Annotated[
        float, typer.Option(help="fraction of samples to add in train split")
    ],
    project_id: Annotated[
        int | None, typer.Option(help=typer_description.LABEL_STUDIO_PROJECT_ID)
    ] = config.label_studio_project_id,
    train_split_name: Annotated[
        str,
        typer.Option(help="name of the train split"),
    ] = "train",
    val_split_name: Annotated[
        str,
        typer.Option(help="name of the validation split"),
    ] = "val",
    task_id_file: Annotated[
        Path | None,
        typer.Option(help="path of a text file containing IDs of samples"),
    ] = None,
    split_name: Annotated[
        str | None,
        typer.Option(
            help="name of the split associated "
            "with the task ID file. If --task-id-file is not provided, "
            "this field is ignored."
        ),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option(help="overwrite existing split field")
    ] = False,
    view_id: Annotated[
        int | None,
        typer.Option(
            help="ID of the Label Studio view, if any. This option is useful "
            "to filter the task to process."
        ),
    ] = None,
    api_key: Annotated[
        str | None, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
    ] = config.label_studio_api_key,
    label_studio_url: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_URL)
    ] = config.label_studio_url,
):
    """Update the split field of tasks in a Label Studio project.

    The behavior of this command depends on the `--task-id-file` option.

    If `--task-id-file` is provided, it should contain a list of task IDs,
    one per line. The split field of these tasks will be updated to the value
    of `--split-name`. The `--train-split` value is ignored in this case.

    If `--task-id-file` is not provided, the split field of all tasks in the
    project will be updated based on the `train_split` probability.
    The split field is set to "train" with probability `train_split`, and "val"
    otherwise.

    In both cases, tasks with a non-null split field are not updated unless
    the `--overwrite` flag is provided.

    The `--view-id` option can be used to only assign the split on a subset
    of the tasks.
    """
    import random

    from label_studio_sdk import Task
    from label_studio_sdk.client import LabelStudio

    check_label_studio_api_key(api_key)
    check_required_field("--project-id", project_id)
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    task_ids = None
    if task_id_file is not None:
        if split_name is None or split_name not in (train_split_name, val_split_name):
            raise typer.BadParameter(
                "--split-name is required when using --task-id-file"
            )
        task_ids = task_id_file.read_text().strip().split("\n")

    for task in ls.tasks.list(project=project_id, fields="all", view=view_id):
        task: Task
        task_id = task.id

        split = task.data.get("split")
        if split is None or overwrite:
            if task_ids:
                if str(task_id) in task_ids:
                    split = split_name
                else:
                    continue
            else:
                split = (
                    train_split_name
                    if random.random() < train_split
                    else val_split_name
                )

            logger.info("Updating task: %s, split: %s", task.id, split)
            ls.tasks.update(task.id, data={**task.data, "split": split})


@app.command()
def create_dataset_file(
    input_file: Annotated[
        Path,
        typer.Option(help="Path to a list of image URLs", exists=True),
    ],
    output_file: Annotated[
        Path, typer.Option(help="Path to the output JSONL file", exists=False)
    ],
):
    """Create a Label Studio object detection dataset file from a list of
    image URLs.

    The output file is a JSONL file. It cannot be imported directly in Label
    Studio (which requires a JSON file as input), the `import-data` command
    should be used to import the generated dataset file.
    """
    from urllib.parse import urlparse

    import tqdm
    from openfoodfacts.images import extract_barcode_from_url, extract_source_from_url
    from openfoodfacts.utils import get_image_from_url

    from labelr.sample.object_detection import format_object_detection_sample_to_ls

    logger.info("Loading dataset: %s", input_file)

    with output_file.open("wt") as f:
        for line in tqdm.tqdm(input_file.open("rt"), desc="images"):
            url = line.strip()
            if not url:
                continue

            extra_meta = {}
            image_id = Path(urlparse(url).path).stem
            if ".openfoodfacts.org" in url:
                barcode = extract_barcode_from_url(url)
                extra_meta["barcode"] = barcode
                off_image_id = Path(extract_source_from_url(url)).stem
                extra_meta["off_image_id"] = off_image_id
                image_id = f"{barcode}_{off_image_id}"

            image = get_image_from_url(url, error_raise=False)

            if image is None:
                logger.warning("Failed to load image: %s", url)
                continue

            label_studio_sample = format_object_detection_sample_to_ls(
                image_id, url, image.width, image.height, extra_meta
            )
            f.write(json.dumps(label_studio_sample) + "\n")


@app.command()
def create_config_file(
    output_file: Annotated[
        Path, typer.Option(help="Path to the output label config file", exists=False)
    ],
    labels: Annotated[
        list[str], typer.Option(help="List of class labels to use for the model")
    ],
):
    """Create a Label Studio label config file for object detection tasks."""
    from labelr.project_config import create_object_detection_label_config

    config = create_object_detection_label_config(labels)
    output_file.write_text(config)
    logger.info("Label config file created: %s", output_file)


@app.command()
def check_dataset(
    project_id: Annotated[
        int | None, typer.Option(help=typer_description.LABEL_STUDIO_PROJECT_ID)
    ] = config.label_studio_project_id,
    view_id: Annotated[int, typer.Option(help="Label Studio View ID, if any.")] = None,
    api_key: Annotated[
        str | None, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
    ] = config.label_studio_api_key,
    label_studio_url: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_URL)
    ] = config.label_studio_url,
    delete_missing_images: Annotated[
        bool,
        typer.Option(help="Delete tasks with missing images from the dataset"),
    ] = False,
    delete_duplicate_images: Annotated[
        bool, typer.Option(help="Delete duplicate images from the dataset")
    ] = False,
):
    """Perform sanity checks of a Label Studio dataset.

    This function checks for:
    - Tasks with missing images (404)
    - Duplicate images based on perceptual hash (pHash)
    - Tasks with multiple annotations

    This function doesn't perform any modifications to the dataset, except
    optionally deleting tasks with missing images if --delete-missing-images
    is provided and tasks with duplicate images if --delete-duplicate-images
    is provided.
    """
    from label_studio_sdk.client import LabelStudio

    from ..check import check_ls_dataset

    check_label_studio_api_key(api_key)
    check_required_field("--project-id", project_id)
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    check_ls_dataset(
        ls=ls,
        project_id=project_id,
        view_id=view_id,
        delete_missing_images=delete_missing_images,
        delete_duplicate_images=delete_duplicate_images,
    )


@app.command()
def list_users(
    api_key: Annotated[
        str | None, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
    ] = config.label_studio_api_key,
    label_studio_url: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_URL)
    ] = config.label_studio_url,
):
    """List all users in Label Studio."""
    from label_studio_sdk.client import LabelStudio

    check_label_studio_api_key(api_key)
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    for user in ls.users.list():
        print(f"{user.id:02d}: {user.email}")


@app.command()
def delete_user(
    user_id: int,
    api_key: Annotated[
        str | None, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
    ] = config.label_studio_api_key,
    label_studio_url: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_URL)
    ] = config.label_studio_url,
):
    """Delete a user from Label Studio."""
    from label_studio_sdk.client import LabelStudio

    check_label_studio_api_key(api_key)
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    ls.users.delete(user_id)


@app.command()
def dump_dataset(
    output_file: Annotated[
        Path, typer.Option(help="Path of the output file", writable=True)
    ],
    project_id: Annotated[
        int | None, typer.Option(help=typer_description.LABEL_STUDIO_PROJECT_ID)
    ] = config.label_studio_project_id,
    api_key: Annotated[
        str | None, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
    ] = config.label_studio_api_key,
    view_id: Annotated[
        int | None,
        typer.Option(
            help="ID of the Label Studio view, if any. This option is useful "
            "to filter the tasks to dump."
        ),
    ] = None,
    label_studio_url: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_URL)
    ] = config.label_studio_url,
):
    """Dump all the tasks of a dataset in a JSONL file.

    All fields of the tasks are exported. A subset of the tasks can be
    selected by filtering tasks based on a view (=tab) using the `--view-id`
    option.
    """
    import orjson
    import tqdm
    from label_studio_sdk.client import LabelStudio

    check_label_studio_api_key(api_key)
    check_required_field("--project-id", project_id)
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    with output_file.open("wb") as f:
        for task in tqdm.tqdm(
            ls.tasks.list(project=project_id, view=view_id), desc="tasks"
        ):
            content = orjson.dumps(task.dict())
            f.write(content + b"\n")
