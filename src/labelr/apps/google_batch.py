import asyncio
import importlib
from pathlib import Path
from typing import Annotated, Any

import typer
from google.genai.types import JSONSchema as GoogleJSONSchema
from google.genai.types import Schema as GoogleSchema
from pydantic import BaseModel

from labelr.google_genai import generate_batch_dataset, launch_batch_job

app = typer.Typer()


def convert_pydantic_model_to_google_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Google doesn't support natively OpenAPI schemas, so we convert them to
    Google `Schema` (a subset of OpenAPI)."""
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
    launch_batch_job(
        run_name=run_name,
        dataset_path=dataset_path,
        model=model,
        location=location,
    )
