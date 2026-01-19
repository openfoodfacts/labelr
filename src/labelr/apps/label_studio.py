import enum
import json
import typing
from pathlib import Path
from typing import Annotated, Optional

import typer
from openfoodfacts.utils import get_logger

from ..config import LABEL_STUDIO_DEFAULT_URL

app = typer.Typer()

logger = get_logger(__name__)


@app.command()
def create(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    title: Annotated[str, typer.Option(help="Project title")],
    config_file: Annotated[
        Path, typer.Option(help="Path to label config file", file_okay=True)
    ],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """Create a new Label Studio project."""
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    label_config = config_file.read_text()

    project = ls.projects.create(title=title, label_config=label_config)
    logger.info(f"Project created: {project}")


@app.command()
def import_data(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    dataset_path: Annotated[
        Path,
        typer.Option(
            help="Path to the Label Studio dataset JSONL file", file_okay=True
        ),
    ],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
    batch_size: int = 25,
):
    """Import tasks from a dataset file to a Label Studio project.

    The dataset file must be a JSONL file: it should contain one JSON object
    per line. To generate such a file, you can use the `create-dataset-file`
    command.
    """
    import more_itertools
    import tqdm
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    with dataset_path.open("rt") as f:
        for batch in more_itertools.chunked(
            tqdm.tqdm(map(json.loads, f), desc="tasks"), batch_size
        ):
            ls.projects.import_tasks(id=project_id, request=batch)


@app.command()
def update_prediction(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    for task in ls.tasks.list(project=project_id, fields="all"):
        for prediction in task.predictions:
            prediction_id = prediction["id"]
            if prediction["model_version"] == "":
                logger.info("Updating prediction: %s", prediction_id)
                ls.predictions.update(
                    id=prediction_id,
                    model_version="undefined",
                )


@app.command()
def add_split(
    train_split: Annotated[
        float, typer.Option(help="fraction of samples to add in train split")
    ],
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    split_name: Annotated[
        Optional[str],
        typer.Option(
            help="name of the split associated "
            "with the task ID file. If --task-id-file is not provided, "
            "this field is ignored."
        ),
    ] = None,
    train_split_name: Annotated[
        str,
        typer.Option(help="name of the train split"),
    ] = "train",
    val_split_name: Annotated[
        str,
        typer.Option(help="name of the validation split"),
    ] = "val",
    task_id_file: Annotated[
        Optional[Path],
        typer.Option(help="path of a text file containing IDs of samples"),
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
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """Update the split field of tasks in a Label Studio project.

    The behavior of this command depends on the `--task-id-file` option.

    If `--task-id-file` is provided, it should contain a list of task IDs,
    one per line. The split field of these tasks will be updated to the value
    of `--split-name`.

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
def annotate_from_prediction(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    updated_by: Annotated[
        Optional[int], typer.Option(help="User ID to declare as annotator")
    ] = None,
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """Create annotations for all tasks from predictions.

    This command is useful if you imported tasks with predictions, and want to
    "validate" these predictions by creating annotations.
    """
    import tqdm
    from label_studio_sdk.client import LabelStudio
    from label_studio_sdk.types.task import Task

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    task: Task
    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, fields="all"), desc="tasks"
    ):
        task_id = task.id
        if task.total_annotations == 0 and task.total_predictions > 0:
            logger.info("Creating annotation for task: %s", task_id)
            ls.annotations.create(
                id=task_id,
                result=task.predictions[0]["result"],
                project=project_id,
                updated_by=updated_by,
            )


class PredictorBackend(enum.Enum):
    ultralytics = "ultralytics"
    robotoff = "robotoff"


@app.command()
def add_prediction(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    view_id: Annotated[
        Optional[int],
        typer.Option(
            help="Label Studio View ID to filter tasks. If not provided, all tasks in the "
            "project are processed."
        ),
    ] = None,
    model_name: Annotated[
        str,
        typer.Option(
            help="Name of the object detection model to run (for Robotoff server) or "
            "of the Ultralytics zero-shot model to run."
        ),
    ] = "yolov8x-worldv2.pt",
    server_url: Annotated[
        Optional[str],
        typer.Option(help="The Robotoff URL if the backend is robotoff"),
    ] = "https://robotoff.openfoodfacts.org",
    backend: Annotated[
        PredictorBackend,
        typer.Option(
            help="Prediction backend: either use Ultralytics to perform "
            "the prediction or Robotoff server."
        ),
    ] = PredictorBackend.ultralytics,
    labels: Annotated[
        Optional[list[str]],
        typer.Option(
            help="List of class labels to use for Yolo model. If you're using Yolo-World or other "
            "zero-shot models, this is the list of label names that are going to be provided to the "
            "model. In such case, you can use `label_mapping` to map the model's output to the "
            "actual class names expected by Label Studio."
        ),
    ] = None,
    label_mapping: Annotated[
        Optional[str],
        typer.Option(help="Mapping of model labels to class names, as a JSON string"),
    ] = None,
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
    threshold: Annotated[
        Optional[float],
        typer.Option(
            help="Confidence threshold for selecting bounding boxes. The default is 0.3 "
            "for robotoff backend and 0.1 for ultralytics backend."
        ),
    ] = None,
    max_det: Annotated[int, typer.Option(help="Maximum numbers of detections")] = 300,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Launch in dry run mode, without uploading annotations to Label Studio"
        ),
    ] = False,
    error_raise: Annotated[
        bool,
        typer.Option(help="Raise an error if image download fails"),
    ] = True,
    model_version: Annotated[
        Optional[str],
        typer.Option(help="Model version to use for the prediction"),
    ] = None,
):
    """Add predictions as pre-annotations to Label Studio tasks,
    for an object detection model running on Triton Inference Server."""

    import tqdm
    from label_studio_sdk.client import LabelStudio
    from openfoodfacts.utils import get_image_from_url, http_session
    from PIL import Image

    from ..annotate import (
        format_annotation_results_from_robotoff,
        format_annotation_results_from_ultralytics,
    )

    label_mapping_dict = None
    if label_mapping:
        label_mapping_dict = json.loads(label_mapping)

    if dry_run:
        logger.info("** Dry run mode enabled **")

    logger.info(
        "backend: %s, model_name: %s, labels: %s, threshold: %s, label mapping: %s",
        backend,
        model_name,
        labels,
        threshold,
        label_mapping,
    )
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    if backend == PredictorBackend.ultralytics:
        from ultralytics import YOLO

        if labels is None:
            raise typer.BadParameter("Labels are required for Ultralytics backend")

        if threshold is None:
            threshold = 0.1

        model = YOLO(model_name)
        if hasattr(model, "set_classes"):
            model.set_classes(labels)
        else:
            logger.warning("The model does not support setting classes directly.")
    elif backend == PredictorBackend.robotoff:
        if server_url is None:
            raise typer.BadParameter("--server-url is required for Robotoff backend")

        if threshold is None:
            threshold = 0.1
            server_url = server_url.rstrip("/")
    else:
        raise typer.BadParameter(f"Unsupported backend: {backend}")

    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, view=view_id), desc="tasks"
    ):
        if task.total_predictions == 0:
            image_url = task.data["image_url"]
            image = typing.cast(
                Image.Image,
                get_image_from_url(image_url, error_raise=error_raise),
            )
            if backend == PredictorBackend.ultralytics:
                results = model.predict(
                    image,
                    conf=threshold,
                    max_det=max_det,
                )[0]
                labels = typing.cast(list[str], labels)
                label_studio_result = format_annotation_results_from_ultralytics(
                    results, labels, label_mapping_dict
                )
            elif backend == PredictorBackend.robotoff:
                r = http_session.get(
                    f"{server_url}/api/v1/images/predict",
                    params={
                        "models": model_name,
                        "output_image": 0,
                        "image_url": image_url,
                    },
                )
                r.raise_for_status()
                response = r.json()
                label_studio_result = format_annotation_results_from_robotoff(
                    response["predictions"][model_name],
                    image.width,
                    image.height,
                    label_mapping_dict,
                )
            if dry_run:
                logger.info("image_url: %s", image_url)
                logger.info("result: %s", label_studio_result)
            else:
                ls.predictions.create(
                    task=task.id,
                    result=label_studio_result,
                    model_version=model_version,
                )


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
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    api_key: Annotated[
        Optional[str], typer.Option(envvar="LABEL_STUDIO_API_KEY")
    ] = None,
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """Check a dataset for duplicate images on Label Studio."""
    from label_studio_sdk.client import LabelStudio

    from ..check import check_ls_dataset

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    check_ls_dataset(ls, project_id)


@app.command()
def list_users(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """List all users in Label Studio."""
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    for user in ls.users.list():
        print(f"{user.id:02d}: {user.email}")


@app.command()
def delete_user(
    user_id: int,
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """Delete a user from Label Studio."""
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    ls.users.delete(user_id)


@app.command()
def dump_dataset(
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    output_file: Annotated[
        Path, typer.Option(help="Path of the output file", writable=True)
    ],
    view_id: Annotated[
        int | None,
        typer.Option(
            help="ID of the Label Studio view, if any. This option is useful "
            "to filter the tasks to dump."
        ),
    ] = None,
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """Dump all the tasks of a dataset in a JSONL file.

    All fields of the tasks are exported. A subset of the tasks can be
    selected by filtering tasks based on a view (=tab) using the `--view-id`
    option.
    """
    import orjson
    import tqdm
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    with output_file.open("wb") as f:
        for task in tqdm.tqdm(
            ls.tasks.list(project=project_id, view=view_id), desc="tasks"
        ):
            content = orjson.dumps(task.dict())
            f.write(content + b"\n")
