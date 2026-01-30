import typing
from collections import defaultdict
from pathlib import Path

import imagehash
import tqdm
from label_studio_sdk.client import LabelStudio
from openfoodfacts.types import JSONType
from openfoodfacts.utils import ImageDownloadItem, get_image_from_url, get_logger
from PIL import Image

logger = get_logger(__name__)


def check_ls_dataset(
    ls: LabelStudio, project_id: int, delete_missing_images: bool = False
):
    """Perform sanity checks of a Label Studio dataset.

    This function checks for:
    - Tasks with missing images (404)
    - Duplicate images based on perceptual hash (pHash)
    - Tasks with multiple annotations

    This function doesn't perform any modifications to the dataset, except
    optionally deleting tasks with missing images if `delete_missing_images`
    is set to True.

    Args:
        ls (LabelStudio): Label Studio client instance.
        project_id (int): ID of the Label Studio project to check.
        delete_missing_images (bool): Whether to delete tasks with missing
            images.
    """
    skipped = 0
    not_annotated = 0
    annotated = 0
    deleted = 0
    multiple_annotations = 0
    hash_map = defaultdict(list)
    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, fields="all"), desc="tasks"
    ):
        annotations = typing.cast(list[JSONType], task.annotations)

        if len(annotations) == 0:
            not_annotated += 1
            continue
        elif len(annotations) > 1:
            logger.warning("Task has multiple annotations: %s", task.id)
            multiple_annotations += 1
            continue

        annotation = annotations[0]

        if annotation["was_cancelled"]:
            skipped += 1

        annotated += 1
        image_url = task.data["image_url"]
        image_struct = typing.cast(
            ImageDownloadItem,
            get_image_from_url(image_url, return_struct=True, error_raise=False),
        )

        if image_struct.response.status_code == 404:
            logger.warning("Image not found (404): %s", image_url)

            if delete_missing_images:
                ls.tasks.delete(task.id)
                deleted += 1
                logger.info("Deleted task with missing image: %s", task.id)
            continue

        if image_struct.image is None:
            logger.warning("Could not open image: %s", image_url)
            continue

        image_hash = str(imagehash.phash(image_struct.image))
        hash_map[image_hash].append(task.id)

    for image_hash, task_ids in hash_map.items():
        if len(task_ids) > 1:
            logger.warning("Duplicate images: %s", task_ids)

    logger.info(
        "Tasks - annotated: %d, skipped: %d, not annotated: %d, multiple annotations: %d",
        annotated,
        skipped,
        not_annotated,
        multiple_annotations,
    )
    logger.info("Deleted tasks with missing images: %d", deleted)


def check_local_dataset(dataset_dir: Path, remove: bool = False):
    hash_map = defaultdict(list)
    for path in tqdm.tqdm(dataset_dir.glob("**/*.jpg"), desc="images"):
        if path.is_file() and path.suffix in [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".tiff",
            ".gif",
        ]:
            image = Image.open(path)
            image_hash = str(imagehash.phash(image))
            logger.debug("Image hash: %s", image_hash)
            hash_map[image_hash].append(path)

    duplicated = 0
    to_remove = []
    for image_hash, image_paths in hash_map.items():
        if len(image_paths) > 1:
            logger.warning(
                "Duplicate images: %s",
                [str(x.relative_to(dataset_dir)) for x in image_paths],
            )
            duplicated += 1
            to_remove.append(image_paths[0])

    logger.info("Total duplicated groups: %d", duplicated)

    if remove and to_remove:
        for path in to_remove:
            logger.info("Removing: %s", str(path))
            path.unlink()
