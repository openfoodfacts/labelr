import datetime
import os
import tempfile
from pathlib import Path
from typing import Annotated

import datasets
import torch
import tqdm
import typer
import ultralytics
import wandb
from datasets import Dataset, Features
from datasets import Image as HFImage
from huggingface_hub import HfApi, ModelCard, ModelCardData
from PIL import Image

from labelr.export import export_from_hf_to_ultralytics_object_detection

FORMAT_TO_EXTENSION = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "GIF": ".gif",
    "BMP": ".bmp",
    "TIFF": ".tif",
    "ICO": ".ico",
    "WEBP": ".webp",
}


CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for {{ model_id | default("Model ID", true) }}

This object detection model was fine-tuned using the Ultralytics YOLO library.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** {{ developers | default("[More Information Needed]", true)}}
- **Model type:** {{ model_type | default("[More Information Needed]", true)}}
- **License:** {{ license | default("[More Information Needed]", true)}}
- **Finetuned from model [optional]:** {{ base_model | default("[More Information Needed]", true)}}

## Training Details

### Training Data

The model was fine-tuned using the following dataset: {{ repo | default("[More Information Needed]", true)}}

### Training Procedure

Dependency versions:

- ultralytics: {{ ultralytics_version | default("[More Information Needed]", true)}}
- pytorch: {{ pytorch_version | default("[More Information Needed]", true)}}

#### Training Hyperparameters

- **Epochs:** {{ training_epochs | default("[More Information Needed]", true)}}
- **Batch size:** {{ training_batch_size | default("[More Information Needed]", true)}}
- **Image size:** {{ training_imgsz | default("[More Information Needed]", true)}}

## Evaluation

The model was evaluated using the following metrics:

{% for metric_name, metric_value in metrics_results_dict.items() %}
- **{{ metric_name }}:** {{ metric_value }}
{% endfor %}

## Files

Most files stored on the repo are standard files created during training with the Ultralytics YOLO library.

What was added:

- an ONNX export of the trained model (best model), stored in `weights/model.onnx`.
- a Parquet file containing predictions on the full dataset, stored in `predictions.parquet`.
"""


def create_model_card(
    dataset_repo_id: str,
    model_id: str,
    base_model: str,
    training_epochs: int,
    training_imgsz: int,
    training_batch_size: int,
    metrics_results_dict: dict[str, float],
    licence: str = "agpl-3.0",
) -> ModelCard:
    card_data = ModelCardData(
        license=licence,
        library_name="ultralytics",
        pipeline_tag="object-detection",
        datasets=[dataset_repo_id],
        base_model=base_model,
    )
    return ModelCard.from_template(
        card_data,
        template_str=CARD_TEMPLATE,
        model_id=model_id,
        developers="Open Food Facts",
        model_type="object detection",
        repo=f"https://huggingface.co/datasets/{dataset_repo_id}",
        metrics_results_dict=metrics_results_dict,
        training_epochs=training_epochs,
        training_imgsz=training_imgsz,
        training_batch_size=training_batch_size,
        ultralytics_version=ultralytics.__version__,
        pytorch_version=torch.__version__,
    )


def create_predict_dataset(
    model: ultralytics.YOLO,
    ds: Dataset,
    output_image_dir: Path,
    output_path: Path,
    imgsz: int,
    batch: int,
    conf: float = 0.25,
):
    # Run the model on the full dataset, draw bounding boxes on images, and
    # save them as a Hugging Face dataset

    # Reset output directories
    if output_image_dir.exists():
        raise ValueError(f"Output images directory already exists: {output_image_dir}")

    if output_path.exists():
        raise ValueError(f"Output parquet file already exists: {output_path}")

    records = []
    for split_name in ds.keys():
        for sample in tqdm.tqdm(ds[split_name]):
            image_id = sample["image_id"]
            image = sample["image"]
            res = model.predict(
                source=image,
                imgsz=imgsz,
                batch=batch,
                save=False,
                verbose=False,
                conf=conf,
            )[0]
            # res.plot() returns an image (numpy array) with boxes drawn
            plotted = res.plot()
            # convert BGR to RGB
            plotted = plotted[:, :, ::-1]
            pil_img = Image.fromarray(plotted)
            extension = FORMAT_TO_EXTENSION[image.format]
            out_path = output_image_dir / split_name / f"{image_id}.{extension}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(out_path)

            boxes = res.boxes
            # Convert ultralytics xyxyn format to (y_min, x_min, y_max, x_max)
            xyxyn = [
                (y_min, x_min, y_max, x_max)
                for (x_min, y_min, x_max, y_max) in boxes.xyxyn.cpu().numpy().tolist()
            ]
            record = {
                "image": str(out_path),
                "detected": {
                    "bbox": xyxyn,
                    "category_id": boxes.cls.cpu().numpy().astype("int64").tolist(),
                    "category_name": [
                        model.names[int(c)] for c in boxes.cls.cpu().numpy()
                    ],
                    "confidence": boxes.conf.cpu().numpy().tolist(),
                },
                "split": split_name,
                "image_id": image_id,
                "width": sample["width"],
                "height": sample["height"],
                "objects": sample["objects"],
            }

            if "meta" in sample:
                record["meta"] = sample["meta"]

            records.append(record)

    # Build a Hugging Face dataset where each example contains the plotted
    # image
    ds = Dataset.from_list(
        records,
        features=Features(
            {
                "image": HFImage(),
                "image_id": datasets.Value("string"),
                "detected": {
                    "bbox": datasets.Sequence(
                        datasets.Sequence(datasets.Value("float32"))
                    ),
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
                    "bbox": datasets.Sequence(
                        datasets.Sequence(datasets.Value("float32"))
                    ),
                    "category_id": datasets.Sequence(datasets.Value("int64")),
                    "category_name": datasets.Sequence(datasets.Value("string")),
                },
            }
        ),
    )
    ds.to_parquet(output_path)
    typer.echo(f"Saved {len(records)} prediction images to: {output_image_dir}")
    typer.echo(f"Saved Hugging Face dataset as Parquet file to: {output_path}")


def main(
    hf_repo_id: Annotated[
        str,
        typer.Argument(
            envvar="HF_REPO_ID", help="Hugging Face repo ID of the dataset to train on"
        ),
    ],
    trained_model_repo_id: Annotated[
        str,
        typer.Option(
            envvar="HF_TRAINED_MODEL_REPO_ID",
            help="Hugging Face repo ID to upload the trained model to",
        ),
    ],
    wandb_api_key: Annotated[
        str, typer.Argument(envvar="WANDB_API_KEY", help="Wandb API key (required)")
    ],
    project: Annotated[
        str,
        typer.Option(
            envvar="WANDB_PROJECT",
            help="Name of the wandb project used to track training",
        ),
    ] = "train-yolo",
    run_name: Annotated[
        str | None,
        typer.Option(
            envvar="RUN_NAME",
            help="Name of the run, used to track training. "
            "This is also used as the branch name when uploading the "
            "trained model to Hugging Face.",
        ),
    ] = None,
    model_name: Annotated[
        str,
        typer.Argument(
            envvar="YOLO_MODEL_NAME",
            help="Name of the base YOLO model to use for training",
        ),
    ] = "yolov8n.pt",
    epochs: Annotated[
        int, typer.Argument(envvar="EPOCHS", help="Number of epochs")
    ] = 100,
    imgsz: Annotated[int, typer.Argument(envvar="IMGSZ")] = 640,
    batch: Annotated[int, typer.Argument(envvar="BATCH_SIZE")] = 64,
    skip_dataset_download: Annotated[
        bool, typer.Option(help="Skip dataset download step, only for debugging")
    ] = False,
):
    if not os.getenv("HF_TOKEN"):
        raise ValueError(
            "HF_TOKEN environment variable not set. This is required to push the trained model to Hugging Face."
        )

    if not os.getenv("WANDB_API_KEY"):
        raise ValueError(
            "WANDB_API_KEY environment variable not set. This is required to log training runs to Weights & Biases."
        )

    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if run_name is None:
        run_name = f"run-{datestamp}"

    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    hf_api = HfApi()
    # This is the same as the dataset directory set in the Ultralytics
    # settings (see `Ultralytics/settings.json`)
    root_dir = Path(__file__).parent.absolute()
    dataset_dir = root_dir / "datasets"
    run_dir = (Path(__file__).parent / project / run_name).absolute()

    # `skip_dataset_download` is an option to skip dataset download, useful
    # for debugging locally
    if not skip_dataset_download:
        export_from_hf_to_ultralytics_object_detection(
            repo_id=hf_repo_id,
            output_dir=dataset_dir,
            download_images=False,
            error_raise=True,
        )

    model = ultralytics.YOLO(model_name, task="detect")
    typer.echo(f"Starting training run: {run_name}")
    # After training, ultralytics re-loads the best model weights
    model.train(
        data=dataset_dir / "data.yaml",
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
        project=project,
        name=run_name,
    )

    # Export the trained model to ONNX format
    model.export(format="onnx")
    # Rename the exported model to a standard name
    (run_dir / "weights/best.onnx").rename(run_dir / "weights/model.onnx")

    metrics = model.metrics
    metrics_results_dict: dict[str, float] = metrics.results_dict

    ds = datasets.load_dataset(hf_repo_id)
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        # After training, run prediction on the full dataset and save results
        output_image_dir = tmp_dir / "predictions" / "images"
        output_parquet_path = run_dir / "predictions.parquet"
        create_predict_dataset(
            model=model,
            ds=ds,
            output_image_dir=output_image_dir,
            output_path=output_parquet_path,
            imgsz=imgsz,
            batch=batch,
        )

    typer.echo(f"Uploading trained model to Hugging Face repo: {trained_model_repo_id}")
    hf_api.create_repo(
        repo_id=trained_model_repo_id,
        repo_type="model",
        exist_ok=True,
    )
    # Commits are sorted by date (last commit first)
    initial_commit = hf_api.list_repo_commits(trained_model_repo_id)[-1]

    branch_exists = any(
        branch.name == run_name
        for branch in hf_api.list_repo_refs(
            repo_id=trained_model_repo_id, repo_type="model"
        ).branches
    )
    if branch_exists:
        typer.echo(
            f"Branch {run_name} already exists in repo {trained_model_repo_id}, adding datestamp to branch name"
        )
        run_name = f"{run_name}-{datestamp}"

    typer.echo(f"Creating branch {run_name} and uploading model files")
    hf_api.create_branch(
        repo_id=trained_model_repo_id,
        repo_type="model",
        # Name the branch after the name of the run
        branch=run_name,
        # Create branch from the initial commit
        revision=initial_commit.commit_id,
    )
    hf_api.upload_folder(
        folder_path=run_dir,
        repo_id=trained_model_repo_id,
        repo_type="model",
        revision=run_name,
    )

    model_id = hf_repo_id.split("/")[1] if "/" in hf_repo_id else hf_repo_id
    model_card = create_model_card(
        dataset_repo_id=hf_repo_id,
        model_id=model_id,
        base_model=model_name,
        training_epochs=epochs,
        training_imgsz=imgsz,
        training_batch_size=batch,
        metrics_results_dict=metrics_results_dict,
    )
    model_card.push_to_hub(
        repo_id=trained_model_repo_id,
        repo_type="model",
        revision=run_name,
    )
    typer.echo("Upload complete")


if __name__ == "__main__":
    app = typer.Typer(pretty_exceptions_show_locals=False)
    app.command()(main)
    app()
