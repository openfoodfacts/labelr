import datetime
import os
from pathlib import Path
from typing import Annotated, Literal

import typer

WANDB_RUN_URL = None


def register_wanb_run_url(trainer):
    """Register the Wandb run URL to a global variable for later use."""
    global WANDB_RUN_URL

    from ultralytics.utils.callbacks.wb import wb

    if wb is not None:
        WANDB_RUN_URL = wb.run.url
        print("Set WANDB_RUN_URL to:", WANDB_RUN_URL)


def main(
    hf_repo_id: Annotated[
        str,
        typer.Argument(
            envvar="HF_REPO_ID",
            help="Hugging Face repo ID of the dataset to train on. "
            "The revision can be specified with '@revision' suffix (ex: @main).",
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
        str | None,
        typer.Argument(
            envvar="YOLO_MODEL_NAME",
            help="Name of the base YOLO model to use for training. Defaults to yolov8n.",
        ),
    ] = None,
    epochs: Annotated[
        int, typer.Option(envvar="EPOCHS", help="Number of epochs")
    ] = 100,
    use_custom_augmentations: Annotated[
        bool,
        typer.Option(
            help="Whether to use custom augmentations for training (image "
            "classification only). The custom augmentation pipeline uses "
            "Albumentations to apply LetterBox transformation at the beginning of the "
            "pipeline, in order to keep the original aspect ratio of the image while "
            "preventing information loss. By default, the standard augmentations "
            "provided by the Ultralytics YOLO library are used.",
            envvar="USE_CUSTOM_AUGMENTATIONS",
        ),
    ] = False,
    validation_keep_aspect_ratio: Annotated[
        bool,
        typer.Option(
            help="Whether to keep the aspect ratio of images during validation "
            "(image classification only). Always True if `use_custom_augmentations` "
            "is True.",
            envvar="VALIDATION_KEEP_ASPECT_RATIO",
        ),
    ] = False,
    imgsz: Annotated[int, typer.Option(envvar="IMGSZ")] = 640,
    batch: Annotated[int, typer.Option(envvar="BATCH_SIZE")] = 64,
    workers: Annotated[
        int, typer.Option(help="Number of data loading workers", envvar="WORKERS")
    ] = 8,
    skip_dataset_download: Annotated[
        bool, typer.Option(help="Skip dataset download step, only for debugging")
    ] = False,
    root_dir: Annotated[
        Path | None,
        typer.Option(help="Root directory for the project", envvar="ROOT_DIR"),
    ] = None,
    task: Annotated[
        Literal["detect", "classify"],
        typer.Option(
            help="The task to perform, either 'detect' for object detection or 'classify' for image classification",
            envvar="TASK",
        ),
    ] = "detect",
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Whether to overwrite the output directory if it already exists. "
            "Use with caution, as this will delete existing files.",
            envvar="OVERWRITE",
        ),
    ] = False,
):
    from train_yolo.utils import check_envvar, save_ultralytics_settings

    check_envvar()
    if root_dir is None:
        root_dir = Path(os.getcwd())

    save_ultralytics_settings(root_dir)
    # Setting the YOLO_CONFIG_DIR environment variable to the directory containing
    # the settings.json file, so that the ultralytics library can find it
    os.environ["YOLO_CONFIG_DIR"] = str(root_dir)

    import shutil

    import datasets
    import wandb
    from huggingface_hub import HfApi
    from labelr.export.object_detection import (
        export_from_hf_to_ultralytics_object_detection,
    )
    from labelr.utils import parse_hf_repo_id
    import ultralytics

    from train_yolo.image_classification import (
        ImageClassificationPredictor,
        ImageClassificationTrainer,
        ImageClassificationValidator,
        export_from_hf_to_ultralytics_image_classification,
        image_classification_create_predict_dataset,
    )
    from train_yolo.model_card import create_model_card
    from train_yolo.object_detection import object_detection_create_predict_dataset

    if task == "detect":
        validation_keep_aspect_ratio = False

    elif task == "classify" and use_custom_augmentations:
        validation_keep_aspect_ratio = True

    dataset_dir = root_dir / "datasets"

    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if run_name is None:
        run_name = f"run-{datestamp}"

    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    run_dir = (root_dir / "runs" / task / project / run_name).absolute()

    if run_dir.exists():
        if overwrite:
            typer.echo(f"Run directory already exists, deleting: {run_dir}.")
            shutil.rmtree(run_dir)
        else:
            raise ValueError(
                f"Run directory already exists: {run_dir}. "
                "Pass --overwrite to overwrite existing run."
            )

    hf_repo_id, revision = parse_hf_repo_id(hf_repo_id)

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

    # Ultralytics expects the `data` parameter to be data.yml for object detection tasks
    # and the data directory for image classification tasks
    dataset_path = (
        dataset_dir / "data.yaml" if task == "detect" else dataset_dir / "data"
    )

    if model_name is None:
        model_name = "yolov8n.pt" if task == "detect" else "yolov8n-cls.pt"
    else:
        if task == "classify" and "-cls" not in model_name:
            raise ValueError(
                "For image classification task, the model name should contain '-cls' suffix. "
                "Either provide a valid model name or let it default to 'yolov8n-cls.pt'"
            )

    # After training, ultralytics re-loads the best model weights
    model = ultralytics.YOLO(model_name, task=task)
    model.add_callback("on_train_start", register_wanb_run_url)
    typer.echo(f"Starting training run: {run_name}")

    # Use a custom trainer for image classification to apply LetterBox
    # layout
    trainer = (
        ImageClassificationTrainer
        if task == "classify" and use_custom_augmentations
        else None
    )
    model.train(
        trainer=trainer,
        data=dataset_path,
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
        project=project,
        name=run_name,
        workers=workers,
    )

    if batch < 0:
        # If batch size is set to -1, ultralytics uses the maximum batch size
        # that fits in the GPU memory. We need to get the actual batch size
        # used, as it will be needed later during validation.
        # The actual batch size is stored in model.trainer.args.batch
        batch = model.trainer.args.batch

    # Export the trained model to ONNX format
    model.export(
        format="onnx",
        # Include NMS in the exported model
        # Ultralytics tweaks the ONNX opset when exporting for GPUs
        # to prevent compatibility issues
        device="gpu",
        # We use opset 20, as it's the latest supported by our version of
        # Triton (25.02)
        opset=20,
    )
    # Rename the exported model to a standard name
    (run_dir / "weights/best.onnx").rename(run_dir / "weights/model.onnx")

    metrics_results_dict: dict[str, dict[str, float]] = {}

    ds = datasets.load_dataset(hf_repo_id, revision=revision)
    # After training, run prediction on the full dataset and save results

    if task == "detect":
        object_detection_create_predict_dataset(
            model=model,
            ds=ds,
            output_path=run_dir / "predictions.parquet",
            imgsz=imgsz,
        )
    else:
        image_classification_create_predict_dataset(
            model=model,
            predictor_cls=(
                ImageClassificationPredictor if validation_keep_aspect_ratio else None
            ),
            ds=ds,
            output_path=run_dir / "predictions.parquet",
            imgsz=imgsz,
        )

    typer.echo("Running validation on exported models to get metrics")
    # Run validation to get metrics for exported models
    for exported_model_path, format_name in [
        (run_dir / "weights/best.pt", "pytorch"),
        (run_dir / "weights/model.onnx", "onnx"),
    ]:
        model = ultralytics.YOLO(exported_model_path, task=task)
        validator = (
            ImageClassificationValidator if validation_keep_aspect_ratio else None
        )
        metrics = model.val(
            validator=validator,
            data=dataset_path,
            imgsz=imgsz,
            batch=batch,
        )
        metrics_results_dict[format_name] = metrics.results_dict
        # Saving metrics as JSON file
        suffix = "" if format_name == "pytorch" else f"_{format_name}"
        (run_dir / f"metrics{suffix}.json").write_text(metrics.to_json())

    typer.echo(f"Uploading trained model to Hugging Face repo: {trained_model_repo_id}")
    hf_api = HfApi()
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
        dataset_revision=revision,
        model_id=model_id,
        base_model=model_name,
        training_epochs=epochs,
        training_imgsz=imgsz,
        training_batch_size=batch,
        metrics_results_dict=metrics_results_dict,
        task=task,
        wandb_run_url=WANDB_RUN_URL,
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
