import json
import random
import shutil
import typing
from pathlib import Path
from typing import Annotated, Optional

import typer
from openfoodfacts import Flavor
from openfoodfacts.utils import get_logger

from labelr.export import export_from_ultralytics_to_hf
from labelr.utils import parse_hf_repo_id

from ..config import LABEL_STUDIO_DEFAULT_URL
from ..types import ExportDestination, ExportSource, TaskType

app = typer.Typer()

logger = get_logger(__name__)


@app.command()
def check(
    api_key: Annotated[
        Optional[str], typer.Option(envvar="LABEL_STUDIO_API_KEY")
    ] = None,
    project_id: Annotated[
        Optional[int], typer.Option(help="Label Studio Project ID")
    ] = None,
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
    dataset_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to the dataset directory", exists=True, file_okay=False
        ),
    ] = None,
    remove: Annotated[
        bool,
        typer.Option(
            help="Remove duplicate images from the dataset, only for local datasets"
        ),
    ] = False,
):
    """Check a dataset for duplicate images."""
    from label_studio_sdk.client import LabelStudio

    from ..check import check_local_dataset, check_ls_dataset

    if project_id is not None:
        ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
        check_ls_dataset(ls, project_id)
    elif dataset_dir is not None:
        check_local_dataset(dataset_dir, remove=remove)
    else:
        raise typer.BadParameter("Either project ID or dataset directory is required")


@app.command()
def split_train_test(
    task_type: TaskType, dataset_dir: Path, output_dir: Path, train_ratio: float = 0.8
):
    """Split a dataset into training and test sets.

    Only classification tasks are supported.
    """
    if task_type == TaskType.classification:
        class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        logger.info("Found classes: %s", [d.name for d in class_dirs])

        output_dir.mkdir(parents=True, exist_ok=True)
        train_dir = output_dir / "train"
        test_dir = output_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        for class_dir in class_dirs:
            input_paths = list(class_dir.glob("*"))
            random.shuffle(input_paths)

            test_count = int(len(input_paths) * (1 - train_ratio))
            if test_count == 0:
                logger.warning("Not enough samples, skipping class: %s", class_dir.name)
                continue

            test_paths = input_paths[:test_count]
            train_paths = input_paths[test_count:]

            for output_dir, input_paths in (
                (train_dir, train_paths),
                (test_dir, test_paths),
            ):
                output_cls_dir = output_dir / class_dir.name
                output_cls_dir.mkdir(parents=True, exist_ok=True)

                for path in input_paths:
                    logger.info("Copying: %s to %s", path, output_cls_dir)
                    shutil.copy(path, output_cls_dir / path.name)
    else:
        raise typer.BadParameter("Unsupported task type")


@app.command()
def convert_object_detection_dataset(
    repo_id: Annotated[
        str, typer.Option(help="Hugging Face Datasets repository ID to convert")
    ],
    output_file: Annotated[
        Path, typer.Option(help="Path to the output JSON file", exists=False)
    ],
):
    """Convert object detection dataset from Hugging Face Datasets to Label
    Studio format, and save it to a JSON file."""
    from datasets import load_dataset

    from labelr.sample import format_object_detection_sample_from_hf_to_ls

    logger.info("Loading dataset: %s", repo_id)
    ds = load_dataset(repo_id)
    logger.info("Dataset loaded: %s", tuple(ds.keys()))

    with output_file.open("wt") as f:
        for split in ds.keys():
            logger.info("Processing split: %s", split)
            for sample in ds[split]:
                label_studio_sample = format_object_detection_sample_from_hf_to_ls(
                    sample, split=split
                )
                f.write(json.dumps(label_studio_sample) + "\n")


@app.command()
def export(
    from_: Annotated[ExportSource, typer.Option("--from", help="Input source to use")],
    to: Annotated[ExportDestination, typer.Option(help="Where to export the data")],
    api_key: Annotated[Optional[str], typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    task_type: Annotated[
        TaskType, typer.Option(help="Type of task to export")
    ] = TaskType.object_detection,
    repo_id: Annotated[
        Optional[str],
        typer.Option(
            help="Hugging Face Datasets repository ID to convert (only if --from or --to is `hf`)"
        ),
    ] = None,
    label_names: Annotated[
        Optional[str],
        typer.Option(help="Label names to use, as a comma-separated list"),
    ] = None,
    project_id: Annotated[
        Optional[int], typer.Option(help="Label Studio Project ID")
    ] = None,
    label_studio_url: Optional[str] = LABEL_STUDIO_DEFAULT_URL,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to the output directory. Only used if the destintation (`to`) is `ultralytics`",
            file_okay=False,
        ),
    ] = None,
    dataset_dir: Annotated[
        Optional[Path],
        typer.Option(help="Path to the dataset directory, only for Ultralytics source"),
    ] = None,
    download_images: Annotated[
        bool,
        typer.Option(
            help="if True, don't use HF images and download images from the server"
        ),
    ] = False,
    is_openfoodfacts_dataset: Annotated[
        bool,
        typer.Option(
            help="Whether the Ultralytics dataset is an OpenFoodFacts dataset, only "
            "for Ultralytics source. This is used to generate the correct image URLs "
            "each image name."
        ),
    ] = True,
    openfoodfacts_flavor: Annotated[
        Flavor,
        typer.Option(
            help="Flavor of the Open Food Facts dataset to use for image URLs, only "
            "for Ultralytics source if is_openfoodfacts_dataset is True. This is used to "
            "generate the correct image URLs each image name. This option is ignored if "
            "is_openfoodfacts_dataset is False."
        ),
    ] = Flavor.off,
    train_ratio: Annotated[
        float,
        typer.Option(
            help="Train ratio for splitting the dataset, if the split name is not "
            "provided (typically, if the source is Label Studio)"
        ),
    ] = 0.8,
    error_raise: Annotated[
        bool,
        typer.Option(
            help="Raise an error if an image download fails, only for Ultralytics"
        ),
    ] = True,
    use_aws_cache: Annotated[
        bool,
        typer.Option(
            help="Use the AWS S3 cache for image downloads instead of images.openfoodfacts.org, "
            "it is ignored if the export format is not Ultralytics"
        ),
    ] = False,
    merge_labels: Annotated[
        bool,
        typer.Option(help="Merge multiple labels into a single label"),
    ] = False,
    revision: Annotated[
        str,
        typer.Option(
            help="Revision (branch, tag or commit) for the Hugging Face Datasets repository. "
            "Only used when importing from or exporting to Hugging Face Datasets."
        ),
    ] = "main",
):
    """Export Label Studio annotation, either to Hugging Face Datasets or
    local files (ultralytics format)."""
    from label_studio_sdk.client import LabelStudio

    from labelr.export import (
        export_from_hf_to_ultralytics_object_detection,
        export_from_ls_to_hf_object_detection,
        export_from_ls_to_ultralytics_object_detection,
    )

    if (to == ExportDestination.hf or from_ == ExportSource.hf) and repo_id is None:
        raise typer.BadParameter("Repository ID is required for export/import with HF")

    if from_ == ExportSource.ultralytics and dataset_dir is None:
        raise typer.BadParameter(
            "Dataset directory is required for export from Ultralytics source"
        )

    label_names_list: list[str] | None = None

    if label_names is None:
        if to == ExportDestination.hf:
            raise typer.BadParameter("Label names are required for HF export")
        if from_ == ExportSource.ls:
            raise typer.BadParameter(
                "Label names are required for export from LS source"
            )
    else:
        label_names = typing.cast(str, label_names)
        label_names_list = label_names.split(",")

    if from_ == ExportSource.ls:
        if project_id is None:
            raise typer.BadParameter("Project ID is required for LS export")
        if api_key is None:
            raise typer.BadParameter("API key is required for LS export")

    if to == ExportDestination.ultralytics and output_dir is None:
        raise typer.BadParameter("Output directory is required for Ultralytics export")

    if from_ == ExportSource.ls:
        if task_type != TaskType.object_detection:
            raise typer.BadParameter(
                "Only object detection task is currently supported with LS source"
            )
        ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
        if to == ExportDestination.hf:
            repo_id = typing.cast(str, repo_id)
            export_from_ls_to_hf_object_detection(
                ls,
                repo_id=repo_id,
                label_names=typing.cast(list[str], label_names_list),
                project_id=typing.cast(int, project_id),
                merge_labels=merge_labels,
                use_aws_cache=use_aws_cache,
                revision=revision,
            )
        elif to == ExportDestination.ultralytics:
            export_from_ls_to_ultralytics_object_detection(
                ls,
                typing.cast(Path, output_dir),
                typing.cast(list[str], label_names_list),
                typing.cast(int, project_id),
                train_ratio=train_ratio,
                error_raise=error_raise,
                merge_labels=merge_labels,
                use_aws_cache=use_aws_cache,
            )

    elif from_ == ExportSource.hf:
        if task_type != TaskType.object_detection:
            raise typer.BadParameter(
                "Only object detection task is currently supported with HF source"
            )
        if to == ExportDestination.ultralytics:
            export_from_hf_to_ultralytics_object_detection(
                typing.cast(str, repo_id),
                typing.cast(Path, output_dir),
                download_images=download_images,
                error_raise=error_raise,
                use_aws_cache=use_aws_cache,
                revision=revision,
            )
        else:
            raise typer.BadParameter("Unsupported export format")
    elif from_ == ExportSource.ultralytics:
        if task_type != TaskType.classification:
            raise typer.BadParameter(
                "Only classification task is currently supported with Ultralytics source"
            )
        if to == ExportDestination.hf:
            export_from_ultralytics_to_hf(
                task_type=task_type,
                dataset_dir=typing.cast(Path, dataset_dir),
                repo_id=typing.cast(str, repo_id),
                merge_labels=merge_labels,
                label_names=typing.cast(list[str], label_names_list),
                is_openfoodfacts_dataset=is_openfoodfacts_dataset,
                openfoodfacts_flavor=openfoodfacts_flavor,
            )


@app.command()
def show_hf_sample(
    repo_id: Annotated[
        str,
        typer.Argument(
            ...,
            help="Hugging Face Datasets repo ID. The revision can be specified by "
            "appending `@<revision>` to the repo ID.",
        ),
    ],
    image_id: Annotated[
        str,
        typer.Argument(
            ...,
            help="ID of the image associated with the sample to display (field: `image_id`)",
        ),
    ],
    output_image_path: Annotated[
        Path | None,
        typer.Option(help="Path to save the sample image (optional)", exists=False),
    ] = None,
):
    """Display a sample from a Hugging Face Datasets repository by image ID."""
    repo_id, revision = parse_hf_repo_id(repo_id)

    from datasets import load_dataset

    ds = load_dataset(repo_id, revision=revision)

    sample = None
    for split in ds.keys():
        samples = ds[split].filter(lambda x: x == image_id, input_columns="image_id")
        if len(samples) > 0:
            sample = samples[0]
            break
    if sample is None:
        typer.echo(f"Sample with image ID {image_id} not found in dataset {repo_id}")
        raise typer.Exit(code=1)

    else:
        for key, value in sample.items():
            typer.echo(f"{key}: {value}")

        if output_image_path is not None:
            image = sample["image"]
            image.save(output_image_path)
            typer.echo(f"Image saved to {output_image_path}")
