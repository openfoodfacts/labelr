import functools
import logging
import pickle
import random
import tempfile
from pathlib import Path

import datasets
import tqdm
from label_studio_sdk.client import LabelStudio
from openfoodfacts.images import download_image

from labelr.export.common import _pickle_sample_generator
from labelr.sample.object_detection import (
    HF_DS_OBJECT_DETECTION_FEATURES,
    format_object_detection_sample_to_hf,
)

logger = logging.getLogger(__name__)


def export_from_ls_to_hf_object_detection(
    ls: LabelStudio,
    repo_id: str,
    label_names: list[str],
    project_id: int,
    merge_labels: bool = False,
    use_aws_cache: bool = True,
    revision: str = "main",
) -> None:
    """Export annotations from a Label Studio project to a Hugging Face
    dataset.

    The Label Studio project should be an object detection project.

    Args:
        ls (LabelStudio): Label Studio client instance.
        repo_id (str): Hugging Face repository ID to push the dataset to.
        label_names (list[str]): List of label names in the project.
        project_id (int): Label Studio project ID to export from.
        merge_labels (bool): Whether to merge all labels into a single label
            named "object". Defaults to False.
        use_aws_cache (bool): Whether to use the AWS image cache when
            downloading images. Defaults to True.
        revision (str): The dataset revision to push to. Defaults to 'main'.
    """
    if merge_labels:
        label_names = ["object"]

    logger.info(
        "Project ID: %d, label names: %s, repo_id: %s, revision: %s",
        project_id,
        label_names,
        repo_id,
        revision,
    )

    for split in ["train", "val"]:
        logger.info("Processing split: %s", split)

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            logger.info("Saving samples to temporary directory: %s", tmp_dir)
            for i, task in tqdm.tqdm(
                enumerate(ls.tasks.list(project=project_id, fields="all")),
                desc="tasks",
            ):
                if task.data["split"] != split:
                    continue
                sample = format_object_detection_sample_to_hf(
                    task_data=task.data,
                    annotations=task.annotations,
                    label_names=label_names,
                    merge_labels=merge_labels,
                    use_aws_cache=use_aws_cache,
                )
                if sample is not None:
                    # Save output as pickle
                    with open(tmp_dir / f"{split}_{i:05}.pkl", "wb") as f:
                        pickle.dump(sample, f)

            hf_ds = datasets.Dataset.from_generator(
                functools.partial(_pickle_sample_generator, tmp_dir),
                features=HF_DS_OBJECT_DETECTION_FEATURES,
            )
            hf_ds.push_to_hub(repo_id, split=split, revision=revision)


def export_from_ls_to_ultralytics_object_detection(
    ls: LabelStudio,
    output_dir: Path,
    label_names: list[str],
    project_id: int,
    train_ratio: float = 0.8,
    error_raise: bool = True,
    merge_labels: bool = False,
    use_aws_cache: bool = True,
):
    """Export annotations from a Label Studio project to the Ultralytics
    format.

    The Label Studio project should be an object detection project with a
    single rectanglelabels annotation result per task.
    """
    if merge_labels:
        label_names = ["object"]
    logger.info("Project ID: %d, label names: %s", project_id, label_names)

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    split_warning_displayed = False

    # NOTE: before, all images were sent to val, the last split
    label_dir = data_dir / "labels"
    images_dir = data_dir / "images"
    for split in ["train", "val"]:
        (label_dir / split).mkdir(parents=True, exist_ok=True)
        (images_dir / split).mkdir(parents=True, exist_ok=True)

    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, fields="all"),
        desc="tasks",
    ):
        split = task.data.get("split")

        if split is None:
            if not split_warning_displayed:
                logger.warning(
                    "Split information not found, assigning randomly. "
                    "To avoid this, set the `split` field in the task data."
                )
                split_warning_displayed = True
            split = "train" if random.random() < train_ratio else "val"

        elif split not in ["train", "val"]:
            raise ValueError("Invalid split name: %s", split)

        if len(task.annotations) > 1:
            logger.warning("More than one annotation found, skipping")
            continue
        elif len(task.annotations) == 0:
            logger.debug("No annotation found, skipping")
            continue

        annotation = task.annotations[0]
        if annotation["was_cancelled"] is True:
            logger.debug("Annotation was cancelled, skipping")
            continue

        if "image_id" not in task.data:
            raise ValueError(
                "`image_id` field not found in task data. "
                "Make sure the task data contains the `image_id` "
                "field, which should be a unique identifier for the image."
            )
        if "image_url" not in task.data:
            raise ValueError(
                "`image_url` field not found in task data. "
                "Make sure the task data contains the `image_url` "
                "field, which should be the URL of the image."
            )
        image_id = task.data["image_id"]
        image_url = task.data["image_url"]

        has_valid_annotation = False
        with (label_dir / split / f"{image_id}.txt").open("w") as f:
            if not any(
                annotation_result["type"] == "rectanglelabels"
                for annotation_result in annotation["result"]
            ):
                continue

            for annotation_result in annotation["result"]:
                if annotation_result["type"] == "rectanglelabels":
                    value = annotation_result["value"]
                    x_min = value["x"] / 100
                    y_min = value["y"] / 100
                    width = value["width"] / 100
                    height = value["height"] / 100
                    label_name = (
                        label_names[0] if merge_labels else value["rectanglelabels"][0]
                    )
                    label_id = label_names.index(label_name)

                    # Save the labels in the Ultralytics format:
                    # - one label per line
                    # - each line is a list of 5 elements:
                    #   - label_id
                    #   - x_center
                    #   - y_center
                    #   - width
                    #   - height
                    x_center = x_min + width / 2
                    y_center = y_min + height / 2
                    f.write(f"{label_id} {x_center} {y_center} {width} {height}\n")
                    has_valid_annotation = True

        if has_valid_annotation:
            download_output = download_image(
                image_url,
                return_struct=True,
                error_raise=error_raise,
                use_cache=use_aws_cache,
            )
            if download_output is None:
                logger.error("Failed to download image: %s", image_url)
                continue

            with (images_dir / split / f"{image_id}.jpg").open("wb") as f:
                f.write(download_output.image_bytes)

    with (output_dir / "data.yaml").open("w") as f:
        f.write("path: data\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test:\n")
        f.write("names:\n")
        for i, label_name in enumerate(label_names):
            f.write(f"  {i}: {label_name}\n")


def export_from_hf_to_ultralytics_object_detection(
    repo_id: str,
    output_dir: Path,
    download_images: bool = True,
    error_raise: bool = True,
    use_aws_cache: bool = True,
    revision: str = "main",
):
    """Export annotations from a Hugging Face dataset project to the
    Ultralytics format.

    The Label Studio project should be an object detection project with a
    single rectanglelabels annotation result per task.

    Args:
        repo_id (str): Hugging Face repository ID to load the dataset from.
        output_dir (Path): Path to the output directory.
        download_images (bool): Whether to download images from URLs in the
            dataset. If False, the dataset is expected to contain an `image`
            field with the image data.
        error_raise (bool): Whether to raise an error if an image fails to
            download. If False, the image will be skipped. This option is only
            used if `download_images` is True. Defaults to True.
        use_aws_cache (bool): Whether to use the AWS image cache when
            downloading images. This option is only used if `download_images`
            is True. Defaults to True.
        revision (str): The dataset revision to load. Defaults to 'main'.
    """
    logger.info("Repo ID: %s, revision: %s", repo_id, revision)
    ds = datasets.load_dataset(repo_id, revision=revision)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    category_id_to_name = {}

    split_map = {
        "train": "train",
        "val": "val",
    }
    if "val" not in ds and "test" in ds:
        logger.info("val split not found, using test split instead as val")
        split_map["val"] = "test"

    for split in ["train", "val"]:
        split_target = split_map[split]
        split_labels_dir = data_dir / "labels" / split
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        split_images_dir = data_dir / "images" / split
        split_images_dir.mkdir(parents=True, exist_ok=True)

        for sample in tqdm.tqdm(ds[split_target], desc="samples"):
            image_id = sample["image_id"]

            if download_images:
                if "meta" not in sample or "image_url" not in sample["meta"]:
                    raise ValueError(
                        "`meta.image_url` field not found in sample. "
                        "Make sure the dataset contains the `meta.image_url` "
                        "field, which should be the URL of the image, or set "
                        "`download_images` to False."
                    )
                image_url = sample["meta"]["image_url"]
                download_output = download_image(
                    image_url,
                    return_struct=True,
                    error_raise=error_raise,
                    use_cache=use_aws_cache,
                )
                if download_output is None:
                    logger.error("Failed to download image: %s", image_url)
                    continue

                with (split_images_dir / f"{image_id}.jpg").open("wb") as f:
                    f.write(download_output.image_bytes)
            else:
                image = sample["image"]
                image.save(split_images_dir / f"{image_id}.jpg")

            objects = sample["objects"]
            bboxes = objects["bbox"]
            category_ids = objects["category_id"]
            category_names = objects["category_name"]

            with (split_labels_dir / f"{image_id}.txt").open("w") as f:
                for bbox, category_id, category_name in zip(
                    bboxes, category_ids, category_names
                ):
                    if category_id not in category_id_to_name:
                        category_id_to_name[category_id] = category_name
                    y_min, x_min, y_max, x_max = bbox
                    y_min = min(max(y_min, 0.0), 1.0)
                    x_min = min(max(x_min, 0.0), 1.0)
                    y_max = min(max(y_max, 0.0), 1.0)
                    x_max = min(max(x_max, 0.0), 1.0)
                    width = x_max - x_min
                    height = y_max - y_min
                    # Save the labels in the Ultralytics format:
                    # - one label per line
                    # - each line is a list of 5 elements:
                    #   - category_id
                    #   - x_center
                    #   - y_center
                    #   - width
                    #   - height
                    x_center = x_min + width / 2
                    y_center = y_min + height / 2
                    f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

    category_names = [
        x[1] for x in sorted(category_id_to_name.items(), key=lambda x: x[0])
    ]
    with (output_dir / "data.yaml").open("w") as f:
        f.write("path: data\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test:\n")
        f.write("names:\n")
        for i, category_name in enumerate(category_names):
            f.write(f"  {i}: {category_name}\n")
