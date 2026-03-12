import functools
import logging
from pathlib import Path
import pickle
import tempfile
import typing

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import datasets
from datasets import Dataset, Features, Value, ClassLabel
from datasets import Image as HFImage
from labelr.export.common import _pickle_sample_generator
import numpy as np
from openfoodfacts.images import download_image
from PIL import Image, ImageOps
import torch
import tqdm
import typer
import ultralytics
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import (
    ClassificationPredictor,
    ClassificationTrainer,
    ClassificationValidator,
)

logger = logging.getLogger(__name__)


DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)


def get_train_transform(
    max_size: int,
    square_symmetry_prob: float = 1.0,
    coarse_dropout_prob: float = 0.4,
    drop_color_prob: float = 0.1,
):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max_size, p=1.0),
            A.PadIfNeeded(min_height=max_size, min_width=max_size, p=1.0),
            A.SquareSymmetry(p=square_symmetry_prob),
            A.CoarseDropout(p=coarse_dropout_prob),
            A.OneOf(
                [
                    A.ToGray(p=1.0),  # p=1.0 inside OneOf
                    A.ChannelDropout(p=1.0),  # p=1.0 inside OneOf
                ],
                p=drop_color_prob,
            ),
            A.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD, p=1.0),
            ToTensorV2(p=1.0),
        ]
    )


def get_predict_transform(max_size: int):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max_size, p=1.0),
            A.PadIfNeeded(min_height=max_size, min_width=max_size, p=1.0),
            A.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD, p=1.0),
            ToTensorV2(p=1.0),
        ]
    )


class ImageClassificationPredictor(ClassificationPredictor):
    def setup_source(self, source):
        super().setup_source(source)
        self.transforms = get_predict_transform(self.args.imgsz)

    def preprocess(self, img):
        """Converts input image to model-compatible data type."""
        if not isinstance(img, torch.Tensor):
            img = torch.stack(
                [
                    self.transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))["image"]
                    for im in img
                ],
                dim=0,
            )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(
            self.model.device
        )
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32


class ImageClassificationValidator(ClassificationValidator):
    def build_dataset(self, img_path: str) -> ClassificationDataset:
        return ImageClassificationDataset(
            root=img_path, args=self.args, augment=False, prefix=self.args.split
        )


class ImageClassificationDataset(ClassificationDataset):
    """A customized dataset class for image classification with enhanced data
    augmentation transforms."""

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """Initialize a customized classification dataset with enhanced data
        augmentation transforms."""
        super().__init__(root, args, augment, prefix)
        train_transforms = get_train_transform(args.imgsz)
        val_transforms = get_predict_transform(args.imgsz)
        self.torch_transforms = train_transforms if augment else val_transforms

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[
            i
        ]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            # Warning: two separate if statements required here, do not combine this with previous line
            if im is None:
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert to RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        sample = self.torch_transforms(image=im)["image"]
        return {"img": sample, "cls": j}


class ImageClassificationTrainer(ClassificationTrainer):
    """A customized trainer class for YOLO classification models with enhanced
    dataset handling."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Build a customized dataset for classification training or
        validation."""
        return ImageClassificationDataset(
            root=img_path, args=self.args, augment=mode == "train", prefix=mode
        )


def export_from_hf_to_ultralytics_image_classification(
    repo_id: str,
    output_dir: Path,
    download_images: bool = True,
    error_raise: bool = True,
    image_max_size: int | None = None,
    revision: str = "main",
    skip_labels: list[str] | None = None,
):
    """Export annotations from a Hugging Face dataset project to the
    Ultralytics format for image classification tasks.

    Args:
        repo_id (str): Hugging Face repository ID to load the dataset from.
        output_dir (Path): Path to the output directory.
        download_images (bool): Whether to download images from URLs in the
            dataset. If False, the dataset is expected to contain an `image`
            field with the image data.
        error_raise (bool): Whether to raise an error if an image fails to
            download. If False, the image will be skipped. This option is only
            used if `download_images` is True. Defaults to True.
        image_max_size (int | None): Maximum size (in pixels) for the images.
            If None, no resizing is performed. Defaults to None.
        revision (str): The dataset revision to load. Defaults to 'main'.
        skip_labels (list[str] | None): List of label names to skip. If the
            label of an annotation result is in this list, it will be skipped.
            Defaults to None.
    """
    skip_labels = skip_labels or []
    logger.info(
        "Repo ID: %s, revision: %s, skip labels: %s", repo_id, revision, skip_labels
    )
    ds = datasets.load_dataset(repo_id, revision=revision)
    label_feature = ds.features["label"]
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": "train",
        "val": "val",
    }
    if "val" not in ds and "test" in ds:
        logger.info("val split not found, using test split instead as val")
        split_map["val"] = "test"

    for split in ["train", "val"]:
        split_target = split_map[split]
        split_dir = data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

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
                image = download_image(
                    image_url, return_struct=False, error_raise=error_raise
                )
                if image is None:
                    logger.error("Failed to download image: %s", image_url)
                    continue
            else:
                image = sample["image"]

            image = typing.cast(Image.Image, image)
            # Rotate image according to exif orientation using Pillow
            # If the image source is Hugging Face, EXIF data is not preserved,
            # so this step is only useful when downloading images.
            ImageOps.exif_transpose(image, in_place=True)
            if image.mode != "RGB":
                # Convert image to RGB if needed
                image = image.convert("RGB")
            # Resize image if larger than max size
            if image_max_size is not None and (
                image.width > image_max_size or image.height > image_max_size
            ):
                image.thumbnail(
                    (image_max_size, image_max_size), Image.Resampling.LANCZOS
                )

            label_id = sample["label"]
            label_name = label_feature.int2str(label_id)

            label_dir = split_dir / label_name
            label_dir.mkdir(parents=True, exist_ok=True)

            image.save(label_dir / f"{image_id}.jpg")


def generate_image_classification_prediction_features(
    label_names: list[str],
) -> Features:
    label_features = ClassLabel(num_classes=len(label_names), names=label_names)
    return Features(
        {
            "image": HFImage(),
            "image_id": Value("string"),
            "detected": {
                "label": label_features,
                "confidence": Value("float32"),
            },
            "split": Value("string"),
            "width": Value("int64"),
            "height": Value("int64"),
            "label": label_features,
        }
    )


def image_classification_create_predict_dataset(
    model: ultralytics.YOLO,
    ds: Dataset,
    output_path: Path,
    imgsz: int,
    label_names: list[str],
    conf: float = 0.25,
):
    """Create a Parquet dataset with model predictions."""
    # Run the model on the full dataset and save them as a Hugging Face dataset

    if output_path.exists():
        raise ValueError(f"Output parquet file already exists: {output_path}")

    ds_features = generate_image_classification_prediction_features(label_names)
    with tempfile.TemporaryDirectory() as tmpdirname_str:
        tmp_dir = Path(tmpdirname_str)
        for split_name in ds.keys():
            for i, sample in tqdm.tqdm(enumerate(ds[split_name])):
                image_id = sample["image_id"]
                image = sample["image"]
                res = model.predict(
                    predictor=ImageClassificationPredictor,
                    source=image,
                    imgsz=imgsz,
                    save=False,
                    verbose=False,
                    conf=conf,
                )[0]
                probs = res.probs.cpu().numpy()
                label_id = probs.argmax().item()
                confidence = probs[label_id].item()
                record = {
                    "image": image,
                    "image_id": image_id,
                    "detected": {
                        "label": label_id,
                        "confidence": confidence,
                    },
                    "split": split_name,
                    "label": sample["label"],
                }

                if "width" in sample:
                    record["width"] = sample["width"]
                if "height" in sample:
                    record["height"] = sample["height"]

                if "meta" in sample:
                    record["meta"] = sample["meta"]

                with open(tmp_dir / f"{i:06d}.pkl", "wb") as f:
                    pickle.dump(record, f)

        # Build a Hugging Face dataset where each example contains the plotted
        # image
        output_ds = typing.cast(
            Dataset,
            Dataset.from_generator(
                functools.partial(_pickle_sample_generator, tmp_dir),
                features=ds_features,
            ),
        )
        output_ds.to_parquet(output_path)
        typer.echo(f"Saved Hugging Face dataset as Parquet file to: {output_path}")
