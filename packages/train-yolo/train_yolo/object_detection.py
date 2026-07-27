import functools
import pickle
import tempfile
from pathlib import Path

import tqdm
import typer
import ultralytics
import webdataset as wds
from datasets import Dataset
from more_itertools import chunked
from PIL import Image

from labelr.dataset_features import OBJECT_DETECTION_DS_PREDICTION_FEATURES
from labelr.export.common import _pickle_sample_generator


def object_detection_create_predict_dataset(
    model: ultralytics.YOLO,
    ds: Dataset,
    output_path: Path,
    imgsz: int,
    conf: float = 0.1,
    batch: int = 16,
):
    """Create a Parquet dataset with model predictions."""
    # Run the model on the full dataset, draw bounding boxes on images, and
    # save them as a Hugging Face dataset

    if output_path.exists():
        raise ValueError(f"Output parquet file already exists: {output_path}")

    with tempfile.TemporaryDirectory() as tmpdirname_str:
        tmp_dir = Path(tmpdirname_str)
        for split_name in ds.keys():
            typer.echo(f"Processing split: {split_name}")
            offset = 0
            for samples in tqdm.tqdm(chunked(ds[split_name], batch), desc="batch"):
                images = [sample["image"] for sample in samples]
                results = model.predict(
                    source=images,
                    imgsz=imgsz,
                    save=False,
                    verbose=False,
                    conf=conf,
                )
                for i, (sample, result) in enumerate(
                    zip(samples, results, strict=True)
                ):
                    # res.plot() returns an image (numpy array) with boxes drawn
                    plotted = result.plot()
                    # convert BGR to RGB
                    plotted = plotted[:, :, ::-1]
                    pil_img = Image.fromarray(plotted)

                    boxes = result.boxes
                    # Convert ultralytics xyxyn format to
                    # (y_min, x_min, y_max, x_max)
                    xyxyn = [
                        (y_min, x_min, y_max, x_max)
                        for (x_min, y_min, x_max, y_max) in boxes.xyxyn.cpu()
                        .numpy()
                        .tolist()
                    ]
                    record = {
                        "image": sample["image"],
                        "image_with_predictions": pil_img,
                        "detected": {
                            "bbox": xyxyn,
                            "category_id": boxes.cls.cpu()
                            .numpy()
                            .astype("int64")
                            .tolist(),
                            "category_name": [
                                model.names[int(c)] for c in boxes.cls.cpu().numpy()
                            ],
                            "confidence": boxes.conf.cpu().numpy().tolist(),
                        },
                        "split": split_name,
                        "image_id": sample["image_id"],
                        "objects": sample["objects"],
                    }

                    if "width" in sample:
                        record["width"] = sample["width"]
                    if "height" in sample:
                        record["height"] = sample["height"]
                    if "meta" in sample:
                        record["meta"] = sample["meta"]

                    record_idx = offset + i
                    with open(tmp_dir / f"{record_idx:06d}.pkl", "wb") as f:
                        pickle.dump(record, f)

                    offset += len(samples)

        # Build a Hugging Face dataset where each example contains the plotted
        # image
        ds = Dataset.from_generator(
            functools.partial(_pickle_sample_generator, tmp_dir),
            features=OBJECT_DETECTION_DS_PREDICTION_FEATURES,
        )
        ds.to_parquet(output_path)
        typer.echo(f"Saved Hugging Face dataset as Parquet file to: {output_path}")


def object_detection_predict_from_webdataset(
    model: ultralytics.YOLO,
    webdataset_url: str,
    output_path: Path,
    label_names: list[str],
    imgsz: int,
    conf: float = 0.1,
    batch: int = 16,
    num_workers: int = 4,
):
    """Create a Parquet dataset with model predictions."""

    if output_path.exists():
        raise ValueError(f"Output JSONL file already exists: {output_path}")

    ds = wds.WebDataset(webdataset_url, shardshuffle=False).map(webdataset_map)
    data_loader = wds.WebLoader(
        ds.batched(batch), batch_size=None, num_workers=num_workers
    )

    with tempfile.TemporaryDirectory() as tmpdirname_str:
        tmp_dir = Path(tmpdirname_str)
        for split_name in ds.keys():
            typer.echo(f"Processing split: {split_name}")
            offset = 0
            for samples in tqdm.tqdm(chunked(ds[split_name], batch), desc="batch"):
                images = [sample["image"] for sample in samples]
                results = model.predict(
                    source=images,
                    imgsz=imgsz,
                    save=False,
                    verbose=False,
                    conf=conf,
                )
                for i, (sample, result) in enumerate(
                    zip(samples, results, strict=True)
                ):
                    # res.plot() returns an image (numpy array) with boxes drawn
                    plotted = result.plot()
                    # convert BGR to RGB
                    plotted = plotted[:, :, ::-1]
                    pil_img = Image.fromarray(plotted)

                    boxes = result.boxes
                    # Convert ultralytics xyxyn format to
                    # (y_min, x_min, y_max, x_max)
                    xyxyn = [
                        (y_min, x_min, y_max, x_max)
                        for (x_min, y_min, x_max, y_max) in boxes.xyxyn.cpu()
                        .numpy()
                        .tolist()
                    ]
                    record = {
                        "image": sample["image"],
                        "image_with_predictions": pil_img,
                        "detected": {
                            "bbox": xyxyn,
                            "category_id": boxes.cls.cpu()
                            .numpy()
                            .astype("int64")
                            .tolist(),
                            "category_name": [
                                model.names[int(c)] for c in boxes.cls.cpu().numpy()
                            ],
                            "confidence": boxes.conf.cpu().numpy().tolist(),
                        },
                        "split": split_name,
                        "image_id": sample["image_id"],
                        "objects": sample["objects"],
                    }

                    if "width" in sample:
                        record["width"] = sample["width"]
                    if "height" in sample:
                        record["height"] = sample["height"]
                    if "meta" in sample:
                        record["meta"] = sample["meta"]

                    record_idx = offset + i
                    with open(tmp_dir / f"{record_idx:06d}.pkl", "wb") as f:
                        pickle.dump(record, f)

                    offset += len(samples)
