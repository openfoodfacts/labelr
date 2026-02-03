import enum
import json
import typing
from pathlib import Path
from typing import Annotated, Optional

import typer
from openfoodfacts.utils import get_logger

from . import typer_description
from ..config import config

app = typer.Typer()

logger = get_logger(__name__)


def check_label_studio_api_key(api_key: str | None):
    if not api_key:
        raise typer.BadParameter(
            "Label Studio API key not provided. Please provide it with the "
            "--api-key option or set the LABELR_LABEL_STUDIO_API_KEY environment variable."
        )


@app.command()
def create(
    title: Annotated[str, typer.Option(help="Project title")],
    config_file: Annotated[
        Path, typer.Option(help="Path to label config file", file_okay=True)
    ],
    api_key: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
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
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    dataset_path: Annotated[
        Path,
        typer.Option(
            help="Path to the Label Studio dataset JSONL file", file_okay=True
        ),
    ],
    api_key: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
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
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    with dataset_path.open("rt") as f:
        for batch in more_itertools.chunked(
            tqdm.tqdm(map(json.loads, f), desc="tasks"), batch_size
        ):
            ls.projects.import_tasks(id=project_id, request=batch)


@app.command()
def update_prediction(
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    api_key: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
    ] = config.label_studio_api_key,
    label_studio_url: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_URL)
    ] = config.label_studio_url,
):
    from label_studio_sdk.client import LabelStudio

    check_label_studio_api_key(api_key)
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
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
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
        str, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
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
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    api_key: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
    ] = config.label_studio_api_key,
    updated_by: Annotated[
        Optional[int], typer.Option(help="User ID to declare as annotator")
    ] = None,
    label_studio_url: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_URL)
    ] = config.label_studio_url,
):
    """Create annotations for all tasks from predictions.

    This command is useful if you imported tasks with predictions, and want to
    "validate" these predictions by creating annotations.
    """
    import tqdm
    from label_studio_sdk.client import LabelStudio
    from label_studio_sdk.types.task import Task

    check_label_studio_api_key(api_key)
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


class PredictorBackend(enum.StrEnum):
    ultralytics = enum.auto()
    ultralytics_yolo_world = enum.auto()
    ultralytics_sam3 = enum.auto()


YOLO_WORLD_MODELS = (
    "yolov8s-world.pt",
    "yolov8s-worldv2.pt",
    "yolov8m-world.pt",
    "yolov8m-worldv2.pt",
    "yolov8l-world.pt",
    "yolov8l-worldv2.pt",
    "yolov8x-world.pt",
    "yolov8x-worldv2.pt",
)


@app.command()
def add_prediction(
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    api_key: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
    ] = config.label_studio_api_key,
    view_id: Annotated[
        int | None,
        typer.Option(
            help="Label Studio View ID to filter tasks. If not provided, all tasks in the "
            "project are processed."
        ),
    ] = None,
    model_name: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Name or path of the object detection model to run. How this is used depends "
            "on the backend. If using `ultralytics` backend, the option is required and is the "
            "name of the model to download from the Ultralytics model zoo or the path to a local "
            "model. "
            "If using `ultralytics_yolo_world` backend, this is optional and is the name of the "
            "`yolo-world` model to download from the Ultralytics model zoo or the path to a local "
            "model (Defaults: `yolov8x-worldv2.pt`). "
            "If using `ultralytics_sam3` backend, this option is ignored, as there is a single model. "
            "The model is downloaded automatically from Hugging Face.",
        ),
    ] = None,
    skip_existing: Annotated[
        bool,
        typer.Option(
            help="Skip tasks that already have predictions",
        ),
    ] = True,
    backend: Annotated[
        PredictorBackend,
        typer.Option(
            help="The prediction backend, possible options are: `ultralytics` or `ultralytics_sam3`"
        ),
    ] = PredictorBackend.ultralytics,
    labels: Annotated[
        list[str] | None,
        typer.Option(
            help="List of class labels to use for Yolo model. If you're using Yolo-World or other "
            "zero-shot models, this is the list of label names that are going to be provided to the "
            "model. In such case, you can use `label_mapping` to map the model's output to the "
            "actual class names expected by Label Studio."
        ),
    ] = None,
    label_mapping: Annotated[
        str | None,
        typer.Option(
            help='Mapping of model labels to class names, as a JSON string. Example: \'{"price tag": "price-tag"}\''
        ),
    ] = None,
    label_studio_url: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_URL)
    ] = config.label_studio_url,
    threshold: Annotated[
        float | None,
        typer.Option(
            help="Confidence threshold for selecting bounding boxes. The default is 0.1 for "
            "ultralytics backend and 0.25 for ultralytics_sam3 backend."
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
        str | None,
        typer.Option(
            help="Set the model version field of the prediction sent to Label Studio. "
            "This is used to track which model generated the prediction."
        ),
    ] = None,
    imgsz: Annotated[
        int | None,
        typer.Option(
            help="Image size to use for Ultralytics models. If not provided, "
            "the default size of the model is used."
        ),
    ] = None,
):
    """Add predictions as pre-annotations to Label Studio tasks."""

    import tqdm
    from huggingface_hub import hf_hub_download
    from label_studio_sdk.client import LabelStudio
    from openfoodfacts.utils import get_image_from_url
    from PIL import Image

    from ..annotate import format_annotation_results_from_ultralytics

    check_label_studio_api_key(api_key)

    label_mapping_dict = None
    if label_mapping:
        label_mapping_dict = json.loads(label_mapping)

    if dry_run:
        logger.info("** Dry run mode enabled **")

    if backend == PredictorBackend.ultralytics and not Path(model_name).is_file():
        raise typer.BadParameter(
            f"Model file '{model_name}' not found. When the backend is `ultralytics` "
            "and the --model does not refer to a YOLO-World model, --model is expected "
            "to be a local Ultralytics model file (`.pt`)."
        )

    logger.info(
        "backend: %s, model_name: %s, labels: %s, threshold: %s, label mapping: %s",
        backend,
        model_name,
        labels,
        threshold,
        label_mapping,
    )
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    if backend in (
        PredictorBackend.ultralytics,
        PredictorBackend.ultralytics_yolo_world,
    ):
        from ultralytics import YOLO, YOLOWorld

        if labels is None:
            raise typer.BadParameter("Labels are required for `ultralytics` backend")

        if threshold is None:
            threshold = 0.1

        if backend == PredictorBackend.ultralytics:
            model = YOLO(model_name)
        elif backend == PredictorBackend.ultralytics_yolo_world:
            if model_name is None:
                model_name = "yolov8x-worldv2.pt"
            model = YOLOWorld(model_name)
            model.set_classes(labels)

    elif backend == PredictorBackend.ultralytics_sam3:
        from ultralytics.models.sam import SAM3SemanticPredictor

        if threshold is None:
            threshold = 0.25

        # SAM3 cannot be downloaded directly using to to a gated access. Use a
        # proxy repo.
        model_path = hf_hub_download(
            "1038lab/sam3",
            filename="sam3.pt",
            revision="f055b060a4de0a040891ba2ebac9c5cb3c1c0132",
        )
        overrides = dict(
            task="segment",
            mode="predict",
            model=model_path,
            save=False,
        )

        if imgsz is not None:
            overrides["imgsz"] = imgsz
        model = SAM3SemanticPredictor(overrides=overrides)
    else:
        raise typer.BadParameter(f"Unsupported backend: {backend}")

    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, view=view_id), desc="tasks"
    ):
        if not (skip_existing and task.total_predictions > 0):
            image_url = task.data["image_url"]
            image = typing.cast(
                Image.Image,
                get_image_from_url(image_url, error_raise=error_raise),
            )
            min_score = None
            if backend in (
                PredictorBackend.ultralytics,
                PredictorBackend.ultralytics_yolo_world,
            ):
                predict_kwargs = {
                    "conf": threshold,
                    "max_det": max_det,
                }
                if imgsz is not None:
                    predict_kwargs["imgsz"] = imgsz
                results = model.predict(image, **predict_kwargs)[0]
                labels = typing.cast(list[str], labels)
                label_studio_result = format_annotation_results_from_ultralytics(
                    results, labels, label_mapping_dict
                )
            elif backend == PredictorBackend.ultralytics_sam3:
                model.set_image(image)
                results = model(text=labels)[0]
                label_studio_result = format_annotation_results_from_ultralytics(
                    results, labels, label_mapping_dict
                )
                min_score = min(results.boxes.conf.tolist(), default=None)
            if dry_run:
                logger.info("image_url: %s", image_url)
                logger.info("result: %s", label_studio_result)
            else:
                ls.predictions.create(
                    task=task.id,
                    result=label_studio_result,
                    model_version=model_version,
                    score=min_score,
                )
                logger.info("Prediction added for task: %s", task.id)


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
    view_id: Annotated[int, typer.Option(help="Label Studio View ID, if any.")] = None,
    api_key: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
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
        str, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
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
        str, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
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
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    output_file: Annotated[
        Path, typer.Option(help="Path of the output file", writable=True)
    ],
    api_key: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
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
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    with output_file.open("wb") as f:
        for task in tqdm.tqdm(
            ls.tasks.list(project=project_id, view=view_id), desc="tasks"
        ):
            content = orjson.dumps(task.dict())
            f.write(content + b"\n")
