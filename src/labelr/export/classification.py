import functools
import logging
import pickle
import tempfile
from pathlib import Path

import datasets
from openfoodfacts.images import generate_image_url
from openfoodfacts.types import Flavor
from PIL import Image, ImageOps

from labelr.export.common import _pickle_sample_generator
from labelr.sample.classification import HF_DS_CLASSIFICATION_FEATURES

logger = logging.getLogger(__name__)


def export_from_ultralytics_to_hf_classification(
    dataset_dir: Path,
    repo_id: str,
    label_names: list[str],
    merge_labels: bool = False,
    is_openfoodfacts_dataset: bool = False,
    openfoodfacts_flavor: Flavor = Flavor.off,
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
        is_openfoodfacts_dataset (bool): Whether the dataset is from
            Open Food Facts. If True, the `off_image_id` and `image_url` will
            be generated automatically. `off_image_id` is extracted from the
            image filename.
        openfoodfacts_flavor (Flavor): Flavor of Open Food Facts dataset. This
            is ignored if `is_openfoodfacts_dataset` is False.
    """
    logger.info("Repo ID: %s, dataset_dir: %s", repo_id, dataset_dir)

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
                    if is_openfoodfacts_dataset:
                        image_stem_parts = image_path.stem.split("_")
                        barcode = image_stem_parts[0]
                        off_image_id = image_stem_parts[1]
                        image_id = f"{barcode}_{off_image_id}"
                        image_url = generate_image_url(
                            barcode, off_image_id, flavor=openfoodfacts_flavor
                        )
                    else:
                        image_id = image_path.stem
                        barcode = ""
                        off_image_id = ""
                        image_url = ""
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
                        "meta": {
                            "barcode": barcode,
                            "off_image_id": off_image_id,
                            "image_url": image_url,
                        },
                        "category_id": label_id,
                        "category_name": label_name,
                    }
                    with open(tmp_dir / f"{split}_{image_id}.pkl", "wb") as f:
                        pickle.dump(sample, f)

            hf_ds = datasets.Dataset.from_generator(
                functools.partial(_pickle_sample_generator, tmp_dir),
                features=HF_DS_CLASSIFICATION_FEATURES,
            )
            hf_ds.push_to_hub(repo_id, split=split)
