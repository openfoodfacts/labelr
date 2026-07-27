import json
import os
import re
from pathlib import Path

import ultralytics
from datasets import Dataset

from labelr.export.object_detection import (
    export_from_hf_to_ultralytics_object_detection,
)
from train_yolo.image_classification import (
    ImageClassificationPredictor,
    export_from_hf_to_ultralytics_image_classification,
    image_classification_create_predict_dataset,
    image_classification_predict_from_webdataset,
)
from train_yolo.object_detection import (
    object_detection_create_predict_dataset,
    object_detection_predict_from_webdataset,
)
from train_yolo.types import TaskType


def download_dataset(
    task: TaskType,
    hf_repo_id: str,
    dataset_dir: Path,
    revision: str,
    skip_dataset_download: bool = False,
) -> None:
    """Export a Hugging Face dataset to the Ultralytics format.

    Args:
        task: The task type, either "detect" or "classify".
        hf_repo_id: The Hugging Face repository ID of the dataset.
        dataset_dir: The directory to save the dataset to.
        revision: The revision of the dataset to download.
        skip_dataset_download: Whether to skip dataset download.
    """
    # `skip_dataset_download` is an option to skip dataset download, useful
    # for debugging locally
    if not skip_dataset_download:
        if task == "detect":
            export_from_hf_to_ultralytics_object_detection(
                repo_id=hf_repo_id,
                output_dir=dataset_dir,
                revision=revision,
                download_images=False,
                error_raise=True,
            )
        else:
            export_from_hf_to_ultralytics_image_classification(
                repo_id=hf_repo_id,
                output_dir=dataset_dir,
                revision=revision,
                download_images=False,
                error_raise=True,
            )


def create_predict_dataset(
    output_path: Path,
    task: TaskType,
    model: ultralytics.YOLO,
    ds: Dataset,
    imgsz: int,
    validation_keep_aspect_ratio: bool,
    batch: int,
) -> None:
    """Run prediction on the full dataset and save results as a parquet file.

    Args:
        output_path: The path to save the predictions to (parquet file).
        task: The task type, either "detect" or "classify".
        model: The trained Ultralytics YOLO model.
        ds: The dataset to run prediction on.
        imgsz: The image size to use for prediction.
        validation_keep_aspect_ratio: Whether to keep aspect ratio during validation.
        batch: The batch size to use for prediction.
    """
    if task == "detect":
        object_detection_create_predict_dataset(
            model=model,
            ds=ds,
            output_path=output_path,
            imgsz=imgsz,
            batch=batch,
        )
    else:
        image_classification_create_predict_dataset(
            model=model,
            predictor_cls=(
                ImageClassificationPredictor if validation_keep_aspect_ratio else None
            ),
            ds=ds,
            output_path=output_path,
            imgsz=imgsz,
            batch=batch,
        )


def predict_from_webdataset(
    output_path: Path,
    task: TaskType,
    model: ultralytics.YOLO,
    webdataset_url: str,
    imgsz: int,
    batch: int,
    label_names: list[str],
    num_workers: int,
) -> None:
    """Run prediction on a webdataset URL.

    Args:
        output_path: The path to save the predictions to (JSONL file).
        task: The task type, either "detect" or "classify".
        model: The trained Ultralytics YOLO model.
        webdataset_url: The URL of the webdataset to run prediction on.
        imgsz: The image size to use for prediction.
        batch: The batch size to use for prediction.
        label_names: The list of label names to use for prediction.
    """
    if task == "detect":
        object_detection_predict_from_webdataset(
            model=model,
            webdataset_url=webdataset_url,
            output_path=output_path,
            imgsz=imgsz,
            batch=batch,
            label_names=label_names,
            num_workers=num_workers,
        )
    else:
        image_classification_predict_from_webdataset(
            model=model,
            webdataset_url=webdataset_url,
            output_path=output_path,
            imgsz=imgsz,
            batch=batch,
            label_names=label_names,
            num_workers=num_workers,
        )


def generate_ultralytics_settings(root_dir: Path) -> dict:
    return {
        "settings_version": "0.0.6",
        "datasets_dir": f"{root_dir}/datasets",
        "weights_dir": f"{root_dir}/weights",
        "runs_dir": f"{root_dir}/runs",
        "uuid": "08c1ccdf367db40afac4e8d21426192fc60fab1eb920743fcb7daaf744cf1752",
        "sync": True,
        "api_key": "",
        "openai_api_key": "",
        "clearml": False,
        "comet": False,
        "dvc": False,
        "hub": False,
        "mlflow": False,
        "neptune": False,
        "raytune": False,
        "tensorboard": False,
        "wandb": True,
        "vscode_msg": False,
        "openvino_msg": False,
    }


def save_ultralytics_settings(root_dir: Path) -> None:
    """Save the Ultralytics settings to a JSON file in the root directory,
    and set the YOLO_CONFIG_DIR environment variable to the directory
    containing the settings.json file."""
    ultralytics_settings = generate_ultralytics_settings(root_dir)
    settings_dir = root_dir / "Ultralytics"
    settings_dir.mkdir(exist_ok=True)
    (settings_dir / "settings.json").write_text(
        json.dumps(ultralytics_settings, indent=2)
    )
    # Setting the YOLO_CONFIG_DIR environment variable to the directory containing
    # the settings.json file, so that the ultralytics library can find it
    os.environ["YOLO_CONFIG_DIR"] = str(root_dir)


def check_envvar():
    if not os.getenv("HF_TOKEN"):
        raise ValueError(
            "HF_TOKEN environment variable not set. This is required to push the trained model to Hugging Face."
        )

    if not os.getenv("WANDB_API_KEY"):
        raise ValueError(
            "WANDB_API_KEY environment variable not set. This is required to log training runs to Weights & Biases."
        )


def get_webdataset_shard_count(url: str) -> int:
    """Return the number of shards in a webdataset URL.

    A webdataset URL has the format 'https://huggingface.co/datasets/timm/imagenet-12k-wds/resolve/main/imagenet12k-train-{{0000..1023}}.tar'
    with the shard indices specified by {{0000..1023}}.

    This function parses the URL and returns the number of shards.
    """
    pattern = re.compile(r"\{\{(\d+)\.\.(\d+)\}\}")
    match = pattern.search(url)
    if not match:
        return 1
    start_idx = int(match.group(1))
    end_idx = int(match.group(2))
    return end_idx - start_idx + 1
