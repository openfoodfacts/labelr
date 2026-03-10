import logging
import random
import re
import string
import typing

import datasets
from openfoodfacts.images import download_image
from openfoodfacts.types import JSONType
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def format_image_classification_sample_to_hf(
    task_id: int,
    task_data: dict,
    annotations: list[dict],
    label_names: list[str],
    merge_labels: bool = False,
    image_max_size: int | None = None,
    skip_labels: list[str] | None = None,
    meta_schema: JSONType | None = None,
) -> dict | None:
    """Format a Label Studio image classification sample to Hugging Face format.

    Args:
        task_id: The ID of the Label Studio task.
        task_data: The task data from Label Studio.
        annotations: The annotations from Label Studio.
        label_names: The list of label names.
        merge_labels: Whether to merge all labels into a single label (the
            first label in `label_names`).
        image_max_size: Maximum size (in pixels) for the images.
            If None, no resizing is performed. Defaults to None.
        skip_labels (list[str] | None): List of label names to skip. If the
            label of an annotation result is in this list, it will be skipped.
            Defaults to None.
        meta_schema (JSONType | None): If provided, the metadata of the samples will
            be formatted according to the given meta schema. The meta schema should
            be a dictionary that defines the structure of the metadata. The keys of
            the dictionary are the field names, and the values are the field types
            (e.g., "string", "integer", "float", or "sequence[string]").

    Returns:
        The formatted sample, or None in the following cases:
        - More than one annotation is found
        - No annotation is found
        - An error occurs when downloading the image
    """
    if len(annotations) > 1:
        logger.info("More than one annotation found, skipping")
        return None
    elif len(annotations) == 0:
        logger.info("No annotation found, skipping")
        return None

    skip_labels = skip_labels or []
    annotation = annotations[0]

    if not annotation["result"]:
        logger.info(
            "No annotation result found for task %s, skipping",
            task_id,
        )
        return None

    annotation_result = annotation["result"][0]
    if annotation_result["type"] != "choices":
        logger.warning(
            "Invalid annotation type: %s, expected 'choices', skipping sample",
            annotation_result["type"],
        )
        return None

    label_name = annotation_result["value"]["choices"][0]
    if merge_labels:
        label_name = "object"
        label_id = 0
    else:
        if label_name in skip_labels:
            return None

        label_id = label_names.index(label_name) if label_name in label_names else -1
        if label_id == -1:
            logger.warning(
                "Label name '%s' not found in label_names, skipping annotation result: '%s'",
                label_name,
                annotation_result,
            )
            return None

    image_url = task_data["image_url"]
    image = download_image(image_url, error_raise=False)
    if image is None:
        logger.error("Failed to download image: %s", image_url)
        return None

    image = typing.cast(Image.Image, image)
    # Correct image orientation using EXIF data
    # Label Studio provides bounding boxes based on the displayed image (after
    # eventual EXIF rotation), so we need to apply the same transformation to
    # the image.
    # Indeed, Hugging Face stores images without applying EXIF rotation, and
    # EXIF data is not preserved in the dataset.
    ImageOps.exif_transpose(image, in_place=True)
    if image.mode != "RGB":
        # Convert image to RGB if needed
        image = image.convert("RGB")
    # Resize image if larger than max size
    if image_max_size is not None and (
        image.width > image_max_size or image.height > image_max_size
    ):
        image.thumbnail((image_max_size, image_max_size), Image.Resampling.LANCZOS)

    width = image.width
    height = image.height
    meta = {
        "image_url": image_url,
    }

    if meta_schema:
        for field_name, _ in meta_schema.items():
            if field_name in task_data.get("meta", {}):
                meta[field_name] = task_data["meta"][field_name]
            else:
                logger.warning(
                    "Field '%s' defined in meta schema but not found in task data "
                    "(task %s), skipping this field",
                    field_name,
                    task_id,
                )

    return {
        "image_id": task_data["image_id"],
        "image": image,
        "width": width,
        "height": height,
        "meta": meta,
        "category_id": label_id,
        "category_name": label_name,
    }


def format_annotation_results(label: str):
    """Format annotation results for image classification tasks in Label Studio."""
    id_ = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    annotation_result = {
        "id": id_,
        "type": "choices",
        "from_name": "choice",
        "to_name": "image",
        "value": {
            "choices": [label],
        },
    }
    return [annotation_result]


def get_hf_image_classification_features(
    meta_schema: JSONType | None = None,
) -> datasets.Features:
    """Get the HuggingFace Dataset features for image classification.

    Args:
        is_openfoodfacts_dataset (bool): Whether the dataset is an Open Food
            Facts dataset. If True, the dataset will include additional
            metadata fields specific to Open Food Facts (`barcode` and
            `off_image_id`).
    """
    features_dict = {
        "image_id": datasets.Value("string"),
        "image": datasets.features.Image(),
        "width": datasets.Value("int64"),
        "height": datasets.Value("int64"),
        "meta": {
            "image_url": datasets.Value("string"),
        },
        "category_id": datasets.Value("int64"),
        "category_name": datasets.Value("string"),
    }

    if meta_schema:
        for field_name, field_type in meta_schema.items():
            if match := re.match(r"^sequence\[([^\]]+)\]$", field_type):
                inner_type = match.group(1)
                features_dict["meta"][field_name] = datasets.Sequence(
                    datasets.Value(inner_type)
                )
            else:
                features_dict["meta"][field_name] = datasets.Value(field_type)

    return datasets.Features(features_dict)
