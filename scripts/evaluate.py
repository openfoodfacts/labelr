# /// script
# dependencies = [
#     "datasets",
#     "fiftyone",
#     "huggingface_hub",
#     "typer",
# ]
# ///

from pathlib import Path
import tempfile
from typing import Annotated, Literal

import datasets
import fiftyone as fo
from huggingface_hub import hf_hub_download

import typer

app = typer.Typer(no_args_is_help=True)

OBJECT_DETECTION_DS_PREDICTION_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "image_with_prediction": datasets.Image(),
        "image_id": datasets.Value("string"),
        "detected": {
            "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
            "category_id": datasets.Sequence(datasets.Value("int64")),
            "category_name": datasets.Sequence(datasets.Value("string")),
            "confidence": datasets.Sequence(datasets.Value("float32")),
        },
        "split": datasets.Value("string"),
        "width": datasets.Value("int64"),
        "height": datasets.Value("int64"),
        "meta": {
            "barcode": datasets.Value("string"),
            "off_image_id": datasets.Value("string"),
            "image_url": datasets.Value("string"),
        },
        "objects": {
            "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
            "category_id": datasets.Sequence(datasets.Value("int64")),
            "category_name": datasets.Sequence(datasets.Value("string")),
        },
    }
)


def generate_image_classification_prediction_features(
    label_names: list[str],
) -> datasets.Features:
    label_features = datasets.ClassLabel(
        num_classes=len(label_names), names=label_names
    )
    return datasets.Features(
        {
            "image": datasets.Image(),
            "image_id": datasets.Value("string"),
            "detected": {
                "label": label_features,
                "confidence": datasets.Value("float32"),
                "probs": datasets.Sequence(datasets.Value("float32")),
            },
            "split": datasets.Value("string"),
            "width": datasets.Value("int64"),
            "height": datasets.Value("int64"),
            "label": label_features,
        }
    )


def parse_hf_repo_id(hf_repo_id: str) -> tuple[str, str]:
    """Parse the repo_id and the revision from a hf_repo_id in the format:
    `org/repo-name@revision`.

    Returns a tuple (repo_id, revision), with revision = 'main' if it
    was not provided.
    """
    if "@" in hf_repo_id:
        hf_repo_id, revision = hf_repo_id.split("@", 1)
    else:
        revision = "main"

    return hf_repo_id, revision


def convert_bbox_to_fo_format(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    # Bounding box coordinates should be relative values
    # in [0, 1] in the following format:
    # [top-left-x, top-left-y, width, height]
    y_min, x_min, y_max, x_max = bbox
    return (
        x_min,
        y_min,
        (x_max - x_min),
        (y_max - y_min),
    )


def _visualize(
    task: Literal["detect", "classify"],
    label_names: list[str] | None,
    hf_repo_id: str,
    dataset_name: str,
    persistent: bool,
):
    hf_repo_id, hf_revision = parse_hf_repo_id(hf_repo_id)

    file_path = hf_hub_download(
        hf_repo_id,
        filename="predictions.parquet",
        revision=hf_revision,
        repo_type="model",
    )
    file_path = Path(file_path).absolute()

    if task == "detect":
        features = OBJECT_DETECTION_DS_PREDICTION_FEATURES
    else:
        if label_names is None:
            raise ValueError("Label names must be provided for classification models.")
        features = generate_image_classification_prediction_features(label_names)

    prediction_dataset = datasets.load_dataset(
        "parquet",
        data_files=str(file_path),
        split="train",
        features=features,
    )
    fo_dataset = fo.Dataset(name=dataset_name, persistent=persistent)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmp_dir = Path(tmpdir_str)
        for i, hf_sample in enumerate(prediction_dataset):
            image = hf_sample["image"]
            image_path = tmp_dir / f"{i}.jpg"
            image.save(image_path)
            split = hf_sample["split"]
            sample = fo.Sample(
                filepath=image_path,
                split=split,
                tags=[split],
                image=hf_sample["image_id"],
            )
            if task == "detect":
                ground_truth_detections = [
                    fo.Detection(
                        label=hf_sample["objects"]["category_name"][i],
                        bounding_box=convert_bbox_to_fo_format(
                            bbox=hf_sample["objects"]["bbox"][i],
                        ),
                    )
                    for i in range(len(hf_sample["objects"]["category_name"]))
                ]
                sample["ground_truth"] = fo.Detections(
                    detections=ground_truth_detections
                )

                if hf_sample["detected"] is not None and hf_sample["detected"]["bbox"]:
                    model_detections = [
                        fo.Detection(
                            label=hf_sample["detected"]["category_name"][i],
                            bounding_box=convert_bbox_to_fo_format(
                                bbox=hf_sample["detected"]["bbox"][i]
                            ),
                            confidence=hf_sample["detected"]["confidence"][i],
                        )
                        for i in range(len(hf_sample["detected"]["bbox"]))
                    ]
                    sample["model"] = fo.Detections(detections=model_detections)
            else:
                sample["ground_truth"] = fo.Classification(
                    label=features["label"].int2str(hf_sample["label"]),
                )
                sample["model"] = fo.Classification(
                    label=features["label"].int2str(hf_sample["detected"]["label"]),
                    confidence=hf_sample["detected"]["confidence"],
                )

            fo_dataset.add_sample(sample)

        # View summary info about the dataset
        print(fo_dataset)

        # Print the first few samples in the dataset
        print(fo_dataset.head())

        # Visualize the dataset in the FiftyOne App
        session = fo.launch_app(fo_dataset)

        if task == "detect":
            fo_dataset.evaluate_detections(
                "model", gt_field="ground_truth", eval_key="eval", compute_mAP=True
            )
        session.wait()


@app.command()
def visualize(
    hf_repo_id: Annotated[
        str,
        typer.Argument(
            help="Hugging Face repository ID of the trained model. "
            "A `predictions.parquet` file is expected in the repo. Revision can be specified "
            "by appending `@<revision>` to the repo ID.",
        ),
    ],
    task: Annotated[
        Literal["detect", "classify"],
        typer.Option(help="The task type of the model predictions to visualize."),
    ],
    label_names: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated list of label names (required for classification models).",
        ),
    ] = None,
    dataset_name: Annotated[
        str | None, typer.Option(help="Name of the FiftyOne dataset to create.")
    ] = None,
    persistent: Annotated[
        bool,
        typer.Option(
            help="Whether to make the FiftyOne dataset persistent (i.e., saved to disk).",
        ),
    ] = False,
):
    """Visualize object detection model predictions stored in a Hugging Face
    repository using FiftyOne."""

    if task == "classify" and label_names is None:
        raise ValueError("Label names must be provided for classification models.")

    if dataset_name is None:
        dataset_name = hf_repo_id.replace("/", "-").replace("@", "-")

    _visualize(
        task=task,
        label_names=label_names.split(",") if label_names else None,
        hf_repo_id=hf_repo_id,
        dataset_name=dataset_name,
        persistent=persistent,
    )


if __name__ == "__main__":
    app()
