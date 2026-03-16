import functools
import logging
import pickle
import tempfile
from pathlib import Path

import datasets
from huggingface_hub import HfApi
from label_studio_sdk import LabelStudio
from PIL import Image, ImageOps
import orjson
import tqdm

from labelr.export.common import _pickle_sample_generator
from labelr.sample.classification import get_hf_image_classification_features
from labelr.sample.classification import format_image_classification_sample_to_hf

logger = logging.getLogger(__name__)


def export_from_ultralytics_to_hf_classification(
    dataset_dir: Path,
    repo_id: str,
    label_names: list[str],
    merge_labels: bool = False,
) -> None:
    """Export an Ultralytics classification dataset to a Hugging Face dataset.

    The Ultralytics dataset directory should contain 'train', 'val' and/or
    'test' subdirectories, each containing subdirectories for each label.

    Args:
        dataset_dir (Path): Path to the Ultralytics dataset directory.
        repo_id (str): Hugging Face repository ID to push the dataset to.
        label_names (list[str]): List of label names.
        merge_labels (bool): Whether to merge all labels into a single label
            named 'object'.
    """
    logger.info("Repo ID: %s, dataset_dir: %s", repo_id, dataset_dir)

    if merge_labels:
        label_names = ["object"]

    ds_features = get_hf_image_classification_features(label_names=label_names)
    if not any((dataset_dir / split).is_dir() for split in ["train", "val", "test"]):
        raise ValueError(
            f"Dataset directory {dataset_dir} does not contain 'train', 'val' or 'test' subdirectories"
        )

    # Save output as pickle
    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split

        if not split_dir.is_dir():
            logger.info("Skipping missing split directory: %s", split_dir)
            continue

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            for label_dir in (d for d in split_dir.iterdir() if d.is_dir()):
                label_name = label_dir.name
                if merge_labels:
                    label_name = "object"
                if label_name not in label_names:
                    raise ValueError(
                        "Label name %s not in provided label names (label names: %s)"
                        % (label_name, label_names),
                    )
                label_id = label_names.index(label_name)

                for image_path in label_dir.glob("*"):
                    image_id = image_path.stem
                    image = Image.open(image_path)
                    image.load()

                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    # Rotate image according to exif orientation using Pillow
                    ImageOps.exif_transpose(image, in_place=True)
                    sample = {
                        "image_id": image_id,
                        "image": image,
                        "width": image.width,
                        "height": image.height,
                        "meta": {},
                        "label": label_id,
                    }
                    with open(tmp_dir / f"{split}_{image_id}.pkl", "wb") as f:
                        pickle.dump(sample, f)

            hf_ds = datasets.Dataset.from_generator(
                functools.partial(_pickle_sample_generator, tmp_dir),
                features=ds_features,
            )
            hf_ds.push_to_hub(repo_id, split=split)


def export_from_ls_to_hf_classification(
    ls: LabelStudio,
    repo_id: str,
    label_names: list[str],
    project_id: int,
    image_max_size: int | None = None,
    view_id: int | None = None,
    merge_labels: bool = False,
    revision: str | None = None,
    skip_labels: list[str] | None = None,
    meta_schema_path: Path | None = None,
    add_repo_tag: str | None = None,
) -> None:
    """Export annotations of an image classification project from a Label
    Studio project to a Hugging Face dataset.

    The Label Studio project should be an image classification project.

    Args:
        ls (LabelStudio): Label Studio instance.
        repo_id (str): Hugging Face repository ID to push the dataset to.
        label_names (list[str]): List of label names.
        project_id (int): Label Studio project ID to export.
        image_max_size (int | None): If provided, the images will be resized to
            have a maximum size of `image_max_size` while keeping the aspect ratio.
        view_id (int | None): If provided, only the annotations from the given
            view will be exported.
        merge_labels (bool): Whether to merge all labels into a single label
            named 'object'.
        revision (str | None): If provided, the dataset will be exported from
            the given revision of the Label Studio project.
        skip_labels (list[str] | None): If provided, the labels in this list
            will be skipped during the export.
        meta_schema_path (Path | None): If provided, the metadata of the samples will
        be formatted according to the given meta schema file. The meta schema file
        should be a JSON file that defines the structure of the metadata.
        add_repo_tag (str | None): If provided, the given git tag will be set to the
            Hugging Face Datasets repository after the export is complete.
    """
    if merge_labels:
        label_names = ["object"]

    logger.info(
        "Project ID: %d, label names: %s, skip labels: %s, "
        "repo_id: %s, revision: %s, view ID: %s, "
        "image_max_size: %s",
        project_id,
        label_names,
        skip_labels,
        repo_id,
        revision,
        view_id,
        image_max_size,
    )

    meta_schema = None
    if meta_schema_path:
        with open(meta_schema_path, "rb") as f:
            meta_schema = orjson.loads(f.read())

    ds_features = get_hf_image_classification_features(
        label_names, meta_schema=meta_schema
    )
    # Save output as pickle
    for split in ["train", "val", "test"]:
        logger.info("Processing split: %s", split)
        has_samples = False

        query = {
            "conjunction": "and",
            "items": [
                {
                    "filter": "filter:tasks:data.split",
                    "operator": "equal",
                    "value": split,
                    "type": "Unknown",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            logger.info("Saving samples to temporary directory: %s", tmp_dir)
            for i, task in tqdm.tqdm(
                enumerate(
                    ls.tasks.list(
                        project=project_id, fields="all", view=view_id, query=query
                    )
                ),
                desc="tasks",
            ):
                if task.data["split"] != split:
                    continue
                sample = format_image_classification_sample_to_hf(
                    task_id=task.id,
                    task_data=task.data,
                    annotations=task.annotations,
                    label_names=label_names,
                    merge_labels=merge_labels,
                    image_max_size=image_max_size,
                    skip_labels=skip_labels,
                    meta_schema=meta_schema,
                )
                if sample is None:
                    continue

                has_samples = True
                with open(tmp_dir / f"{split}_{i:05}.pkl", "wb") as f:
                    pickle.dump(sample, f)

            if not has_samples:
                logger.info("No samples for split: %s", split)
                continue

            hf_ds = datasets.Dataset.from_generator(
                functools.partial(_pickle_sample_generator, tmp_dir),
                features=ds_features,
            )
            hf_ds.push_to_hub(repo_id, split=split)

    if add_repo_tag is not None:
        api = HfApi()
        api.create_tag(
            repo_id, tag=add_repo_tag, revision=revision, repo_type="dataset"
        )
