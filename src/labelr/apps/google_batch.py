import asyncio
import importlib
from pathlib import Path
from typing import Annotated, Any

import typer
from openfoodfacts import Flavor
from openfoodfacts.types import JSONType
from pydantic import BaseModel


app = typer.Typer(no_args_is_help=True)


def _check_json_schema(item: JSONType) -> None:
    if item.get("type") == "object":
        required_fields = item.get("required", [])
        all_fields = item.get("properties", [])
        diff = set(all_fields) - set(required_fields)
        if diff:
            raise ValueError(
                f"fields '{diff}' must be marked as required in the JSONSchema. "
                "All fields with type 'object' must be required."
            )
    return None


def check_json_schema(json_schema: JSONType) -> None:
    """Check that for all `object`s, all fields are marked as required.

    This is important to check, as otherwise the structured generation
    backend may prevent the model to generate these fields.
    This is the case as of vLLM 0.13 and xgrammars as backend.

    To prevent this, we ask all fields to be marked as required.
    """
    stack = [json_schema]

    for def_item in json_schema.get("$defs", {}).values():
        stack.append(def_item)

    while stack:
        item = stack.pop()
        _check_json_schema(item)
        for sub_item in item.get("properties", {}).values():
            stack.append(sub_item)


def convert_pydantic_model_to_google_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Google doesn't support natively OpenAPI schemas, so we convert them to
    Google `Schema` (a subset of OpenAPI)."""
    from google.genai.types import JSONSchema as GoogleJSONSchema
    from google.genai.types import Schema as GoogleSchema

    return GoogleSchema.from_json_schema(
        json_schema=GoogleJSONSchema.model_validate(schema.model_json_schema())
    ).model_dump(mode="json", exclude_none=True, exclude_unset=True)


@app.command()
def generate_dataset(
    data_path: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path to a JSONL file containing the raw batch samples.",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path where to write the generated dataset file.",
            exists=False,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    config_module: Annotated[
        str,
        typer.Option(
            ...,
            help="Python module path (e.g., 'myschema') containing two variables: "
            "OUTPUT_SCHEMA (a Pydantic class representing the output schema) and "
            "INSTRUCTIONS (a str containing instructions to add before each sample).",
        ),
    ],
    bucket_name: Annotated[
        str,
        typer.Option(
            ...,
            help="Name of the GCS bucket where the images are stored.",
        ),
    ] = "robotoff-batch",
    bucket_dir_name: Annotated[
        str,
        typer.Option(
            ...,
            help="Directory name in the GCS bucket where the images are stored.",
        ),
    ] = "gemini-batch-images",
    max_concurrent_uploads: Annotated[
        int,
        typer.Option(
            ...,
            help="Maximum number of concurrent uploads to GCS.",
        ),
    ] = 30,
    base_image_dir: Annotated[
        Path | None,
        typer.Option(
            ...,
            help="Base directory to resolve local image paths from.",
        ),
    ] = None,
    from_key: Annotated[
        str | None,
        typer.Option(
            ...,
            help="If specified, resume processing from this sample key.",
        ),
    ] = None,
    skip_upload: Annotated[
        bool, typer.Option(..., help="Skip uploading images to GCS")
    ] = False,
    thinking_level: Annotated[
        str | None,
        typer.Option(
            ...,
            help="Thinking level to use for the generation config.",
        ),
    ] = None,
):
    """Generate a dataset file in JSONL format to be used for batch
    processing, using Gemini Batch Inference."""
    from labelr.google_genai import generate_batch_dataset

    typer.echo(f"Uploading images from '{data_path}' to GCS bucket '{bucket_name}'...")
    typer.echo(f"Writing updated dataset to {output_path}...")
    typer.echo(f"Max concurrent uploads: {max_concurrent_uploads}...")
    typer.echo(f"Base image directory: {base_image_dir}...")
    typer.echo(f"From key: {from_key}...")
    typer.echo(f"Skip upload: {skip_upload}...")
    typer.echo(f"Thinking level: {thinking_level}...")

    module = importlib.import_module(config_module)
    base_cls = getattr(module, "OUTPUT_SCHEMA")

    if not issubclass(base_cls, BaseModel):
        typer.echo(
            f"Error: {config_module}.OUTPUT_SCHEMA is not a subclass of pydantic.BaseModel"
        )
        raise typer.Exit(code=1)

    instructions = getattr(module, "INSTRUCTIONS", None) or None

    if instructions:
        typer.echo(f"Using instructions: '{instructions}'...")
    else:
        typer.echo("No instructions provided.")

    # JSON Schema is supoorted natively by Vertex AI and Gemini APIs,
    # but not yet on Batch Inference...
    # So we convert the JSON schema to Google internal "Schema"
    # google_json_schema = base_cls.model_json_schema()
    google_json_schema = convert_pydantic_model_to_google_schema(base_cls)
    asyncio.run(
        generate_batch_dataset(
            data_path=data_path,
            output_path=output_path,
            google_json_schema=google_json_schema,
            instructions=instructions,
            bucket_name=bucket_name,
            bucket_dir_name=bucket_dir_name,
            max_concurrent_uploads=max_concurrent_uploads,
            base_image_dir=base_image_dir,
            from_key=from_key,
            skip_upload=skip_upload,
            thinking_level=thinking_level,
        )
    )


@app.command(name="launch-batch-job")
def launch_batch_job_command(
    run_name: Annotated[str, typer.Argument(..., help="Name of the batch job run")],
    dataset_path: Annotated[Path, typer.Option(..., help="Path to the dataset file")],
    model: Annotated[str, typer.Option(..., help="Model to use for the batch job")],
    location: Annotated[
        str,
        typer.Option(..., help="GCP location where to run the batch job"),
    ] = "europe-west4",
):
    """Launch a Gemini Batch Inference job."""
    from labelr.google_genai import launch_batch_job

    launch_batch_job(
        run_name=run_name,
        dataset_path=dataset_path,
        model=model,
        location=location,
    )


@app.command()
def upload_training_dataset_from_predictions(
    prediction_path: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path to the prediction JSONL file generated by Google Inference Batch",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    instructions_path: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path to the file with the instruction prompt for the model",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    json_schema_path: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path to the file with the JSON schema to follow",
            dir_okay=False,
            readable=True,
        ),
    ],
    repo_id: Annotated[
        str, typer.Option(help="Hugging Face Datasets repository ID to push to")
    ],
    revision: Annotated[
        str,
        typer.Option(
            help="Revision (branch, tag or commit) to use for the Hugging Face Datasets repository"
        ),
    ] = "main",
    is_openfoodfacts_dataset: Annotated[
        bool, typer.Option(..., help="Whether this is an Open Food Facts dataset")
    ] = False,
    openfoodfacts_flavor: Annotated[
        Flavor,
        typer.Option(
            ...,
            help="Open Food Facts flavor of the dataset (if applicable)",
        ),
    ] = Flavor.off,
    split: Annotated[str, typer.Option(..., help="Name of the split")] = "train",
    tmp_dir: Annotated[
        Path | None,
        typer.Option(
            ...,
            help="Temporary directory to use for intermediate files, default to a temporary directory "
            "generated automatically. This is useful to relaunch the command if it fails midway.",
        ),
    ] = None,
    skip: Annotated[int, typer.Option(..., help="Number of samples to skip")] = 0,
    limit: Annotated[
        int | None,
        typer.Option(
            ..., help="Limit number of samples to process, or None for no limit"
        ),
    ] = None,
    raise_on_invalid_sample: Annotated[
        bool,
        typer.Option(
            ...,
            help="Whether to raise an error on invalid samples instead of skipping them",
        ),
    ] = False,
    image_max_size: Annotated[
        int | None,
        typer.Option(
            help="Maximum size (in pixels) for the images. If None, no resizing is performed.",
        ),
    ] = None,
):
    """Upload a training dataset to a Hugging Face Datasets repository from a
    Gemini batch prediction file."""
    import tempfile

    import orjson
    from huggingface_hub import HfApi

    from labelr.export.llm import export_to_hf_llm_image_extraction
    from labelr.google_genai import generate_sample_iter

    instructions = instructions_path.read_text()
    print(f"Instructions: {instructions}")
    json_schema = orjson.loads(json_schema_path.read_text())

    # We check that all fields are marked as required
    check_json_schema(json_schema)

    api = HfApi()
    config = {
        "instructions": instructions,
        "json_schema": json_schema,
    }
    with tempfile.TemporaryDirectory() as config_tmp_dir_str:
        config_tmp_dir = Path(config_tmp_dir_str)
        config_path = config_tmp_dir / "config.json"
        config_path.write_text(
            orjson.dumps(config, option=orjson.OPT_INDENT_2).decode("utf-8")
        )
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
    sample_iter = generate_sample_iter(
        prediction_path=prediction_path,
        json_schema=json_schema,
        is_openfoodfacts_dataset=is_openfoodfacts_dataset,
        openfoodfacts_flavor=openfoodfacts_flavor,
        skip=skip,
        limit=limit,
        raise_on_invalid_sample=raise_on_invalid_sample,
    )
    export_to_hf_llm_image_extraction(
        sample_iter=sample_iter,
        split=split,
        repo_id=repo_id,
        revision=revision,
        tmp_dir=tmp_dir,
        image_max_size=image_max_size,
    )
