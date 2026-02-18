import enum
import json
import typing
from pathlib import Path
from typing import Annotated

import typer
from openfoodfacts.utils import get_logger

from . import typer_description
from ...config import config, check_required_field
from .common import check_label_studio_api_key

app = typer.Typer(no_args_is_help=True)

logger = get_logger(__name__)


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
def add(
    model_version: Annotated[
        str,
        typer.Option(
            help="Set the model version field of the prediction sent to Label Studio. "
            "This is used to track which model generated the prediction."
        ),
    ],
    project_id: Annotated[
        int | None, typer.Option(help=typer_description.LABEL_STUDIO_PROJECT_ID)
    ] = config.label_studio_project_id,
    api_key: Annotated[
        str | None, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
    ] = config.label_studio_api_key,
    view_id: Annotated[
        int | None,
        typer.Option(help=typer_description.LABEL_STUDIO_VIEW_ID),
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
    from PIL.ImageOps import exif_transpose
    from PIL import Image

    from labelr.annotate import format_annotation_results_from_ultralytics

    check_label_studio_api_key(api_key)
    check_required_field("--project-id", project_id)

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
        "backend: %s, model_name: %s, labels: %s, threshold: %s, label mapping: %s, view ID: %s",
        backend,
        model_name,
        labels,
        threshold,
        label_mapping,
        view_id,
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
                Image.Image | None,
                get_image_from_url(image_url, error_raise=error_raise),
            )
            if image is None:
                continue

            # Make sure that the image orientation is in accordance with the EXIF metadata
            exif_transpose(image, in_place=True)
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
                min_score = min(results.boxes.conf.tolist(), default=None)
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
def delete(
    model_version: Annotated[
        str,
        typer.Option(
            help="The model version associated with the prediction set to delete"
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
    dry_run: Annotated[
        bool,
        typer.Option(help="Launch in dry run mode, without deleting any prediction"),
    ] = False,
):
    """Delete predictions for a given model version in a Label Studio project."""
    from label_studio_sdk.client import LabelStudio
    import tqdm

    check_label_studio_api_key(api_key)
    check_required_field("--project-id", project_id)
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    query = {
        "conjunction": "and",
        "items": [
            {
                "filter": "filter:tasks:predictions_model_versions",
                "operator": "contains",
                "value": [model_version],
                "type": "List",
            }
        ],
    }

    for task in tqdm.tqdm(ls.tasks.list(project=project_id, query=query), desc="tasks"):
        for prediction in task.predictions:
            if prediction["model_version"] != model_version:
                continue
            if dry_run:
                logger.info("Dry run: prediction %s would be deleted", prediction["id"])
            else:
                ls.predictions.delete(prediction["id"])
                logger.info("Prediction deleted: %s", prediction["id"])


@app.command()
def show_uncertain(
    model_version: Annotated[
        str,
        typer.Option(help="The model version associated with the prediction"),
    ],
    threshold: Annotated[float, typer.Option(help="Score threshold")],
    add_score: Annotated[
        bool,
        typer.Option(
            help="Whether to add the score as a second column in output, making output a CSV file. "
            "This option is ignored if --output is not provided."
        ),
    ] = True,
    output: Annotated[
        str | None,
        typer.Option(help="Path to an file where the task IDs that match (optional)"),
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
    """Show all tasks that have a prediction named `model_version` with a minimum
    score below `threshold`."""
    from label_studio_sdk.client import LabelStudio
    import tqdm
    import sys

    check_label_studio_api_key(api_key)
    check_required_field("--project-id", project_id)
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    if output == "-":
        f = sys.stdout
    elif output is None:
        f = open("/dev/null", "w")
    else:
        f = open(output, "w")

    with f:
        for task in tqdm.tqdm(
            ls.tasks.list(project=project_id, fields="all"), desc="tasks"
        ):
            for prediction in task.predictions:
                if prediction["model_version"] != model_version:
                    continue
                elif (min_score := (prediction["score"] or 0.0)) <= threshold:
                    logger.info("Task ID: %s, score: %s", task.id, min_score)
                    line = f"{task.id},{min_score}\n" if add_score else f"{task.id}\n"
                    f.write(line)
                    f.flush()


@app.command()
def annotate_from(
    project_id: Annotated[
        int | None, typer.Option(help=typer_description.LABEL_STUDIO_PROJECT_ID)
    ] = config.label_studio_project_id,
    api_key: Annotated[
        str | None, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
    ] = config.label_studio_api_key,
    updated_by: Annotated[
        int | None, typer.Option(help="User ID to declare as annotator")
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
    from label_studio_sdk.types import LseTask

    check_label_studio_api_key(api_key)
    check_required_field("--project-id", project_id)
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    task: LseTask
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
