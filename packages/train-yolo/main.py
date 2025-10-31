from pathlib import Path
from typing import Annotated

import typer
import ultralytics
import wandb

from labelr.export import export_from_hf_to_ultralytics_object_detection


def main(
    hf_repo_id: Annotated[str, typer.Argument(envvar="HF_REPO_ID")],
    model_name: Annotated[str, typer.Argument(envvar="YOLO_MODEL_NAME")] = "yolov8n.pt",
    epochs: Annotated[int, typer.Argument(envvar="EPOCHS")] = 100,
    imgsz: Annotated[int, typer.Argument(envvar="IMGSZ")] = 640,
    batch: Annotated[int, typer.Argument(envvar="BATCH_SIZE")] = 64,
    use_aws_image_cache: Annotated[
        bool, typer.Argument(envvar="USE_AWS_IMAGE_CACHE")
    ] = False,
    wandb_api_key: Annotated[str, typer.Argument(envvar="WANDB_API_KEY")] = None,
    project: Annotated[
        str | None,
        typer.Option(
            envvar="WANDB_PROJECT",
            help="Name of the wandb project used to track training",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            envvar="WANDB_NAME",
            help="Name of the wandb project used for to track training",
        ),
    ] = None,
    skip_dataset_download: Annotated[
        bool, typer.Option(help="Skip dataset download step, only for debugging")
    ] = False,
):
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    dataset_dir = (Path(__file__).parent / "dataset").absolute()

    updated_settings = {"datasets_dir": str(dataset_dir)}

    if wandb_api_key:
        updated_settings["wandb"] = True

    ultralytics.settings.update(updated_settings)

    if not skip_dataset_download:
        export_from_hf_to_ultralytics_object_detection(
            repo_id=hf_repo_id,
            output_dir=dataset_dir,
            download_images=False,
            use_aws_cache=use_aws_image_cache,
            error_raise=True,
        )

    model = ultralytics.YOLO(model_name, task="detect")
    model.train(
        data=dataset_dir / "data.yaml",
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
        project=project,
        name=name,
    )


if __name__ == "__main__":
    app = typer.Typer(pretty_exceptions_show_locals=False)
    app.command()(main)
    app()
