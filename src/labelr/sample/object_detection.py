import logging
import random
import string
import typing

import datasets
from openfoodfacts import Flavor
from openfoodfacts.barcode import normalize_barcode
from openfoodfacts.images import download_image, generate_image_url
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def format_annotation_results_from_hf_to_ls(
    objects: dict, image_width: int, image_height: int
):
    """Format annotation results from a HF object detection dataset into Label
    Studio format."""
    annotation_results = []
    for i in range(len(objects["bbox"])):
        bbox = objects["bbox"][i]
        # category_id = objects["category_id"][i]
        category_name = objects["category_name"][i]
        # These are relative coordinates (between 0.0 and 1.0)
        y_min, x_min, y_max, x_max = bbox
        # Make sure the coordinates are within the image boundaries,
        # and convert them to percentages
        y_min = min(max(0, y_min), 1.0) * 100
        x_min = min(max(0, x_min), 1.0) * 100
        y_max = min(max(0, y_max), 1.0) * 100
        x_max = min(max(0, x_max), 1.0) * 100
        x = x_min
        y = y_min
        width = x_max - x_min
        height = y_max - y_min

        id_ = "".join(random.choices(string.ascii_letters + string.digits, k=10))
        annotation_results.append(
            {
                "id": id_,
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": image_width,
                "original_height": image_height,
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rectanglelabels": [category_name],
                },
            },
        )
    return annotation_results


def format_object_detection_sample_from_hf_to_ls(hf_sample: dict, split: str) -> dict:
    hf_meta = hf_sample["meta"]
    objects = hf_sample["objects"]
    image_width = hf_sample["width"]
    image_height = hf_sample["height"]
    annotation_results = format_annotation_results_from_hf_to_ls(
        objects, image_width, image_height
    )
    image_id = hf_sample["image_id"]
    image_url = hf_meta["image_url"]
    meta_kwargs = {}

    if "off_image_id" in hf_meta:
        # If `off_image_id` is present, we assume this is an Open Food Facts
        # dataset sample.
        # We normalize the barcode, and generate a new image URL
        # to make sure that:
        # - the image URL is valid with correct path
        # - we use the images subdomain everywhere
        off_image_id = hf_meta["off_image_id"]
        meta_kwargs["off_image_id"] = off_image_id
        barcode = normalize_barcode(hf_meta["barcode"])
        meta_kwargs["barcode"] = barcode
        image_id = f"{barcode}_{off_image_id}"

        if ".openfoodfacts." in image_url:
            flavor = Flavor.off
        elif ".openbeautyfacts." in image_url:
            flavor = Flavor.obf
        elif ".openpetfoodfacts." in image_url:
            flavor = Flavor.opf
        elif ".openproductsfacts." in image_url:
            flavor = Flavor.opf
        else:
            raise ValueError(
                f"Unknown Open Food Facts flavor for image URL: {image_url}"
            )
        image_url = generate_image_url(
            code=barcode, image_id=off_image_id, flavor=flavor
        )

    return {
        "data": {
            "image_id": image_id,
            "image_url": image_url,
            "batch": "null",
            "split": split,
            "meta": {
                "width": image_width,
                "height": image_height,
                **meta_kwargs,
            },
        },
        "predictions": [{"result": annotation_results}],
    }


def format_object_detection_sample_to_ls(
    image_id: str,
    image_url: str,
    width: int,
    height: int,
    extra_meta: dict | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Format an object detection sample in Label Studio format.

    Args:
        image_id: The image ID.
        image_url: The URL of the image.
        width: The width of the image.
        height: The height of the image.
        extra_meta: Extra metadata to include in the sample.
        tags: List of tags to include in the sample.
    """
    extra_meta = extra_meta or {}
    return {
        "data": {
            "image_id": image_id,
            "image_url": image_url,
            "batch": "null",
            "meta": {
                "width": width,
                "height": height,
                **extra_meta,
            },
            "tags": tags or [],
        },
    }


def format_object_detection_sample_to_hf(
    task_data: dict,
    annotations: list[dict],
    label_names: list[str],
    merge_labels: bool = False,
    use_aws_cache: bool = False,
    image_max_size: int | None = None,
    skip_labels: list[str] | None = None,
) -> dict | None:
    """Format a Label Studio object detection sample to Hugging Face format.

    Args:
        task_data: The task data from Label Studio.
        annotations: The annotations from Label Studio.
        label_names: The list of label names.
        merge_labels: Whether to merge all labels into a single label (the
            first label in `label_names`).
        use_aws_cache: Whether to use AWS cache when downloading images.
        image_max_size: Maximum size (in pixels) for the images.
            If None, no resizing is performed. Defaults to None.
        skip_labels (list[str] | None): List of label names to skip. If the
            label of an annotation result is in this list, it will be skipped.
            Defaults to None.

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
    bboxes = []
    bbox_label_ids = []
    bbox_label_names = []

    for annotation_result in annotation["result"]:
        if annotation_result["type"] != "rectanglelabels":
            continue
            # raise ValueError("Invalid annotation type: %s" % annotation_result["type"])

        value = annotation_result["value"]
        x_min = value["x"] / 100
        y_min = value["y"] / 100
        width = value["width"] / 100
        height = value["height"] / 100
        x_max = x_min + width
        y_max = y_min + height
        bboxes.append([y_min, x_min, y_max, x_max])

        label_name = label_names[0] if merge_labels else value["rectanglelabels"][0]
        if label_name in skip_labels:
            continue
        label_id = label_names.index(label_name) if label_name in label_names else -1
        if label_id == -1:
            logger.warning(
                "Label name '%s' not found in label_names, skipping annotation result: '%s'",
                label_name,
                annotation_result,
            )
            continue
        bbox_label_names.append(label_name)
        bbox_label_ids.append(label_id)

    image_url = task_data["image_url"]
    image = download_image(image_url, error_raise=False, use_cache=use_aws_cache)
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

    meta = task_data.get("meta", {})
    barcode = meta.get("barcode", None)
    off_image_id = meta.get("off_image_id", None)
    width = image.width
    height = image.height
    return {
        "image_id": task_data["image_id"],
        "image": image,
        "width": width,
        "height": height,
        "meta": {
            "barcode": barcode,
            "off_image_id": off_image_id,
            "image_url": image_url,
        },
        "objects": {
            "bbox": bboxes,
            "category_id": bbox_label_ids,
            "category_name": bbox_label_names,
        },
    }


def get_hf_object_detection_features(
    is_openfoodfacts_dataset: bool,
) -> datasets.Features:
    """Get the HuggingFace Dataset features for object detection.

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
        "objects": {
            "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
            "category_id": datasets.Sequence(datasets.Value("int64")),
            "category_name": datasets.Sequence(datasets.Value("string")),
        },
    }

    if is_openfoodfacts_dataset:
        features_dict["meta"]["barcode"] = datasets.Value("string")
        features_dict["meta"]["off_image_id"] = datasets.Value("string")

    return datasets.Features(features_dict)
