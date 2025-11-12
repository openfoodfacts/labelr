import tempfile
from pathlib import Path

import datasets
import fiftyone as fo
from huggingface_hub import hf_hub_download

from labelr.dataset_features import OBJECT_DETECTION_DS_PREDICTION_FEATURES
from labelr.utils import parse_hf_repo_id


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


def visualize(
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
        # local_dir="./predictions/",
    )
    file_path = Path(file_path).absolute()
    prediction_dataset = datasets.load_dataset(
        "parquet",
        data_files=str(file_path),
        split="train",
        features=OBJECT_DETECTION_DS_PREDICTION_FEATURES,
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
            ground_truth_detections = [
                fo.Detection(
                    label=hf_sample["objects"]["category_name"][i],
                    bounding_box=convert_bbox_to_fo_format(
                        bbox=hf_sample["objects"]["bbox"][i],
                    ),
                )
                for i in range(len(hf_sample["objects"]["bbox"]))
            ]
            sample["ground_truth"] = fo.Detections(detections=ground_truth_detections)

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

            fo_dataset.add_sample(sample)

        # View summary info about the dataset
        print(fo_dataset)

        # Print the first few samples in the dataset
        print(fo_dataset.head())

        # Visualize the dataset in the FiftyOne App
        session = fo.launch_app(fo_dataset)
        fo_dataset.evaluate_detections(
            "model", gt_field="ground_truth", eval_key="eval", compute_mAP=True
        )
        session.wait()
