from pathlib import Path
from typing import Annotated
import typing

from PIL import Image, ImageOps
from openfoodfacts.types import JSONType
import typer
from openfoodfacts.utils import get_logger
from openfoodfacts.images import download_image
import orjson

from . import typer_description
from ...config import config, check_required_field
from .common import check_label_studio_api_key

app = typer.Typer(no_args_is_help=True)

logger = get_logger(__name__)


@app.command()
def export_objects(
    output_dir: Annotated[
        Path,
        typer.Option(
            help="Directory to save exported objects",
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ],
    project_id: Annotated[
        int | None, typer.Option(help=typer_description.LABEL_STUDIO_PROJECT_ID)
    ] = config.label_studio_project_id,
    api_key: Annotated[
        str | None, typer.Option(help=typer_description.LABEL_STUDIO_API_KEY)
    ] = config.label_studio_api_key,
    max_area: Annotated[
        int | None, typer.Option(help="Maximum area (in pixels) for exported objects")
    ] = None,
    min_area: Annotated[
        int | None, typer.Option(help="Minimum area (in pixels) for exported objects")
    ] = None,
    min_width: Annotated[
        int | None, typer.Option(help="Minimum width (in pixels) for exported objects")
    ] = None,
    max_width: Annotated[
        int | None, typer.Option(help="Maximum width (in pixels) for exported objects")
    ] = None,
    min_height: Annotated[
        int | None, typer.Option(help="Minimum height (in pixels) for exported objects")
    ] = None,
    max_height: Annotated[
        int | None, typer.Option(help="Maximum height (in pixels) for exported objects")
    ] = None,
    min_aspect_ratio: Annotated[
        float | None,
        typer.Option(help="Minimum aspect ratio (width/height) for exported objects"),
    ] = None,
    max_aspect_ratio: Annotated[
        float | None,
        typer.Option(help="Maximum aspect ratio (width/height) for exported objects"),
    ] = None,
    label_studio_url: Annotated[
        str, typer.Option(help=typer_description.LABEL_STUDIO_URL)
    ] = config.label_studio_url,
):
    """Export annotated objects of a Label Studio project to a specified directory.
    Optionally filters are available to export only objects that meet certain criteria
    based on their area, width, height, and aspect ratio.

    By default, all objects will be exported.
    In the ouput directory, each object will be saved as a separate image file,
    named with the format: `{task_id}_{object_id}.jpg`.

    The `task_id` corresponds to the ID of the task in Label Studio, and the
    `object_id` corresponds to the ID of the annotated object within that task.

    All metadata related to objects are also exported in a JSONL file named `objects.jsonl`
    in the output directory. Each line in the JSONL file corresponds to an exported object
    and contains the following fields:

    - `task_id`: The ID of the task in Label Studio.
    - `object_id`: The ID of the annotated object within the task.
    - `x`: The absolute x-coordinate of the top-left corner of the bounding box (in pixels).
    - `y`: The absolute y-coordinate of the top-left corner of the bounding box (in pixels).
    - `width`: The width of the bounding box (in pixels).
    - `height`: The height of the bounding box (in pixels).
    - `area`: The area of the bounding box (in pixels).
    - `aspect_ratio`: The aspect ratio of the bounding box (width divided by height).
    """
    import tqdm
    from label_studio_sdk.client import LabelStudio
    from label_studio_sdk.types import LseTask

    check_label_studio_api_key(api_key)
    check_required_field("--project-id", project_id)
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "objects.jsonl").open("wb") as f:
        for task in tqdm.tqdm(
            ls.tasks.list(project=project_id, fields="all"), desc="tasks"
        ):
            task = typing.cast(LseTask, task)
            task_id = task.id
            if task.total_annotations > 1:
                logger.info("No than one annotation for task: %s, skipping", task_id)
                continue
            elif task.total_annotations == 0:
                continue

            annotation_results: list[JSONType] = task.annotations[0]["result"]
            for annotation in annotation_results:
                if annotation["type"] != "rectanglelabels":
                    continue

                image_url = task.data.get("image_url")
                if not image_url:
                    logger.warning("No image URL found for task: %s, skipping", task_id)
                    continue

                try:
                    image = download_image(image_url, error_raise=True)
                except Exception as e:
                    logger.error("Failed to export object for task %s: %s", task_id, e)
                    continue

                image = typing.cast(Image.Image, image)
                if not image.mode == "RGB":
                    image = image.convert("RGB")  # Ensure the image is in RGB mode
                image = ImageOps.exif_transpose(
                    image
                )  # Correct orientation based on EXIF data

                object_id = annotation["id"]
                x = annotation["value"]["x"] * image.width / 100
                y = annotation["value"]["y"] * image.height / 100
                width = annotation["value"]["width"] * image.width / 100
                height = annotation["value"]["height"] * image.height / 100
                area = width * height
                aspect_ratio = width / height if height > 0 else 0

                if max_area is not None and area > max_area:
                    continue
                if min_area is not None and area < min_area:
                    continue
                if min_aspect_ratio is not None and aspect_ratio < min_aspect_ratio:
                    continue
                if max_aspect_ratio is not None and aspect_ratio > max_aspect_ratio:
                    continue
                if min_width is not None and width < min_width:
                    continue
                if max_width is not None and width > max_width:
                    continue
                if min_height is not None and height < min_height:
                    continue
                if max_height is not None and height > max_height:
                    continue

                f.write(
                    orjson.dumps(
                        {
                            "task_id": task_id,
                            "object_id": object_id,
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "area": area,
                            "aspect_ratio": aspect_ratio,
                        }
                    )
                    + b"\n"
                )
                # Crop the image to the bounding box of the annotated object
                left = int(x)
                top = int(y)
                right = int(x + width)
                bottom = int(y + height)

                if (
                    left < 0
                    or top < 0
                    or right > image.width
                    or bottom > image.height
                    or left >= right
                    or top >= bottom
                ):
                    logger.warning(
                        "Invalid bounding box for task %s, object %s: (%s, %s, %s, %s), skipping",
                        task_id,
                        object_id,
                        left,
                        top,
                        right,
                        bottom,
                    )
                    continue
                cropped_image = image.crop((left, top, right, bottom))
                output_path = output_dir / f"{task_id}_{object_id}.webp"
                cropped_image.save(output_path)
