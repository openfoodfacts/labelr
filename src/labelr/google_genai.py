import asyncio
import mimetypes
from collections.abc import Iterator
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import aiofiles
import jsonschema
import orjson
import typer
from gcloud.aio.storage import Storage
from openfoodfacts import Flavor
from openfoodfacts.images import download_image, generate_image_url
from tqdm.asyncio import tqdm

from labelr.sample import LLMImageExtractionSample, SampleMeta
from labelr.utils import download_image_from_gcs

try:
    import google.genai  # noqa: F401
except ImportError:
    raise ImportError(
        "The 'google-genai' package is required to use this module. "
        "Please install labelr with the 'google' extra: "
        "`pip install labelr[google]`"
    )
import aiohttp
from google import genai
from google.cloud import storage
from google.genai.types import CreateBatchJobConfig, HttpOptions
from google.genai.types import JSONSchema as GoogleJSONSchema
from google.genai.types import Schema as GoogleSchema
from openfoodfacts.types import JSONType
from pydantic import BaseModel


class RawBatchSamplePart(BaseModel):
    type: Literal["text", "image"]
    data: str


class RawBatchSample(BaseModel):
    key: str
    parts: list[RawBatchSamplePart]
    meta: JSONType = {}


def convert_pydantic_model_to_google_schema(schema: type[BaseModel]) -> JSONType:
    """Google doesn't support natively OpenAPI schemas, so we convert them to
    Google `Schema` (a subset of OpenAPI)."""
    return GoogleSchema.from_json_schema(
        json_schema=GoogleJSONSchema.model_validate(schema.model_json_schema())
    ).model_dump(mode="json", exclude_none=True, exclude_unset=True)


async def download_image(url: str, session: aiohttp.ClientSession) -> bytes:
    """Download an image from a URL and return its content as bytes.

    Args:
        url (str): URL of the image to download.
    Returns:
        bytes: Content of the downloaded image.
    """
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.read()


async def download_image_from_filesystem(url: str, base_dir: Path) -> bytes:
    """Download an image from the filesystem and return its content as bytes.

    Args:
        url (str): URL of the image to download.
        base_dir (Path): Base directory where images are stored.
    Returns:
        bytes: Content of the downloaded image.
    """
    file_path = urlparse(url).path[1:]  # Remove leading '/'
    full_file_path = base_dir / file_path
    async with aiofiles.open(full_file_path, "rb") as f:
        return await f.read()


async def upload_to_gcs(
    image_url: str,
    bucket_name: str,
    blob_name: str,
    session: aiohttp.ClientSession,
    base_image_dir: Path | None = None,
) -> dict:
    """Upload data to Google Cloud Storage.
    Args:
        bucket_name (str): Name of the GCS bucket.
        blob_name (str): Name of the blob (object) in the bucket.
        data (bytes): Data to upload.
        session (aiohttp.ClientSession): HTTP session to use for downloading
            the image.
        base_image_dir (Path | None): If provided, images will be read from
            the filesystem under this base directory instead of downloading
            them from their URLs.
    Returns:
        dict: Status of the upload operation.
    """
    if base_image_dir is None:
        image_data = await download_image(image_url, session)
    else:
        image_data = await download_image_from_filesystem(image_url, base_image_dir)

    client = Storage(session=session)

    status = await client.upload(
        bucket_name,
        blob_name,
        image_data,
    )
    return status


async def upload_to_gcs_format_async(
    sample: RawBatchSample,
    google_json_schema: JSONType,
    instructions: str | None,
    bucket_name: str,
    bucket_dir_name: str,
    session: aiohttp.ClientSession,
    base_image_dir: Path | None = None,
    skip_upload: bool = False,
    thinking_level: str | None = None,
) -> JSONType | None:
    parts: list[JSONType] = []

    if instructions:
        parts.append({"text": instructions})

    for part in sample.parts:
        if part.type == "image":
            mime_type, _ = mimetypes.guess_type(part.data)
            if mime_type is None:
                raise ValueError(f"Cannot guess mimetype of file: {part.data}")

            file_uri = part.data
            image_blob_name = f"{bucket_dir_name}/{sample.key}/{Path(file_uri).name}"
            # Download the image from the URL
            if not skip_upload:
                try:
                    await upload_to_gcs(
                        image_url=file_uri,
                        bucket_name=bucket_name,
                        blob_name=image_blob_name,
                        session=session,
                        base_image_dir=base_image_dir,
                    )
                except FileNotFoundError:
                    return None

            parts.append(
                {
                    "file_data": {
                        "file_uri": f"gs://{bucket_name}/{image_blob_name}",
                        "mime_type": mime_type,
                    }
                }
            )
        else:
            parts.append({"text": part.data})

    generation_config = {
        "responseMimeType": "application/json",
        "response_json_schema": google_json_schema,
    }

    if thinking_level is not None:
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level}

    return {
        "key": f"key:{sample.key}",
        "request": {
            "contents": [
                {
                    "parts": parts,
                    "role": "user",
                }
            ],
            "generationConfig": generation_config,
        },
    }


async def generate_batch_dataset(
    data_path: Path,
    output_path: Path,
    google_json_schema: JSONType,
    instructions: str | None,
    bucket_name: str,
    bucket_dir_name: str,
    max_concurrent_uploads: int = 30,
    base_image_dir: Path | None = None,
    from_key: str | None = None,
    skip_upload: bool = False,
    thinking_level: str | None = None,
):
    limiter = asyncio.Semaphore(max_concurrent_uploads)
    ignore = True if from_key is None else False
    missing_files = 0
    async with aiohttp.ClientSession() as session:
        async with asyncio.TaskGroup() as tg:
            async with (
                aiofiles.open(data_path, "r") as input_file,
                aiofiles.open(output_path, "wb") as output_file,
            ):
                async with limiter:
                    tasks = set()
                    async for line in tqdm(input_file, desc="samples"):
                        # print(f"line: {line}")
                        sample = RawBatchSample.model_validate_json(line)
                        # print(f"sample: {sample}")
                        record_key = sample.key
                        if from_key is not None and ignore:
                            if record_key == from_key:
                                ignore = False
                            else:
                                continue
                        task = tg.create_task(
                            upload_to_gcs_format_async(
                                sample=sample,
                                google_json_schema=google_json_schema,
                                instructions=instructions,
                                bucket_name=bucket_name,
                                bucket_dir_name=bucket_dir_name,
                                session=session,
                                base_image_dir=base_image_dir,
                                skip_upload=skip_upload,
                                thinking_level=thinking_level,
                            )
                        )
                        tasks.add(task)

                        if len(tasks) >= max_concurrent_uploads:
                            for task in tasks:
                                await task
                                updated_record = task.result()
                                if updated_record is not None:
                                    await output_file.write(
                                        orjson.dumps(updated_record) + "\n".encode()
                                    )
                                else:
                                    missing_files += 1
                            tasks.clear()

                    for task in tasks:
                        await task
                        updated_record = task.result()
                        if updated_record is not None:
                            await output_file.write(
                                orjson.dumps(updated_record) + "\n".encode()
                            )
                        else:
                            missing_files += 1

    typer.echo(
        f"Upload and dataset update completed. Wrote updated dataset to {output_path}. "
        f"Missing files: {missing_files}."
    )


def launch_batch_job(
    run_name: str,
    dataset_path: Path,
    model: str,
    location: str,
):
    """Launch a Gemini Batch Inference job.

    Args:
        run_name (str): Name of the batch run.
        dataset_path (Path): Path to the dataset file in JSONL format.
        model (str): Model to use for the batch job. Example:
            'gemini-2.5-flash'.
        location (str): Location for the Vertex AI resources. Example:
            'europe-west4'.
    """
    # We upload the dataset to a GCS bucket using the Gcloud

    if model == "gemini-3-pro-preview" and location != "global":
        typer.echo(
            "Warning: only 'global' location is supported for 'gemini-3-pro-preview' model. Overriding location to 'global'."
        )
        location = "global"

    storage_client = storage.Client()
    bucket_name = "robotoff-batch"  # Replace with your bucket name
    run_dir = f"gemini-batch/{run_name}"
    input_file_blob_name = f"{run_dir}/inputs.jsonl"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(input_file_blob_name)
    blob.upload_from_filename(dataset_path)

    client = genai.Client(
        http_options=HttpOptions(api_version="v1"),
        vertexai=True,
        location=location,
    )
    output_uri = f"gs://{bucket_name}/{run_dir}"
    job = client.batches.create(
        model=model,
        src=f"gs://{bucket_name}/{input_file_blob_name}",
        config=CreateBatchJobConfig(dest=output_uri),
    )
    print(job)


def generate_sample_iter(
    prediction_path: Path,
    json_schema: JSONType,
    skip: int = 0,
    limit: int | None = None,
    is_openfoodfacts_dataset: bool = False,
    openfoodfacts_flavor: Flavor = Flavor.off,
    raise_on_invalid_sample: bool = False,
) -> Iterator[LLMImageExtractionSample]:
    """Generate training samples from a Gemini Batch Inference prediction
    JSONL file.

    Args:
        prediction_path (Path): Path to the prediction JSONL file.
        json_schema (JSONType): JSON schema to validate the predictions.
        skip (int): Number of initial samples to skip.
        limit (int | None): Maximum number of samples to generate.
        is_openfoodfacts_dataset (bool): Whether the dataset is from Open Food
            Facts.
        openfoodfacts_flavor (Flavor): Flavor of the Open Food Facts dataset.
    Yields:
        Iterator[LLMImageExtractionSample]: Generated samples.
    """
    skipped = 0
    invalid = 0
    storage_client = storage.Client()
    with prediction_path.open("r") as f_in:
        for i, sample_str in enumerate(f_in):
            if i < skip:
                skipped += 1
                continue
            if limit is not None and i >= skip + limit:
                break
            sample = orjson.loads(sample_str)
            try:
                yield generate_sample_from_prediction(
                    json_schema=json_schema,
                    sample=sample,
                    is_openfoodfacts_dataset=is_openfoodfacts_dataset,
                    openfoodfacts_flavor=openfoodfacts_flavor,
                    storage_client=storage_client,
                )
            except Exception as e:
                if raise_on_invalid_sample:
                    raise
                else:
                    typer.echo(
                        f"Skipping invalid sample at line {i + 1} in {prediction_path}: {e}"
                    )
                    invalid += 1
                    continue
    if skipped > 0:
        typer.echo(f"Skipped {skipped} samples.")
    if invalid > 0:
        typer.echo(f"Skipped {invalid} invalid samples.")


def generate_sample_from_prediction(
    json_schema: JSONType,
    sample: JSONType,
    is_openfoodfacts_dataset: bool = False,
    openfoodfacts_flavor: Flavor = Flavor.off,
    storage_client: storage.Client | None = None,
) -> LLMImageExtractionSample:
    """Generate a LLMImageExtractionSample from a prediction sample.
    Args:
        json_schema (JSONType): JSON schema to validate the predictions.
        sample (JSONType): Prediction sample.
        is_openfoodfacts_dataset (bool): Whether the dataset is from Open Food
            Facts.
        openfoodfacts_flavor (Flavor): Flavor of the Open Food Facts dataset.
        storage_client (storage.Client | None): Optional Google Cloud Storage
            client. If not provided, a new client will be created.
    Returns:
        LLMImageExtractionSample: Generated sample.
    """
    image_id = sample["key"][len("key:") :]
    response_str = sample["response"]["candidates"][0]["content"]["parts"][0]["text"]
    image_uri = sample["request"]["contents"][0]["parts"][1]["file_data"]["file_uri"]
    image = download_image_from_gcs(image_uri=image_uri, client=storage_client)
    response = orjson.loads(response_str)
    jsonschema.validate(response, json_schema)

    if is_openfoodfacts_dataset:
        image_stem_parts = image_id.split("_")
        barcode = image_stem_parts[0]
        off_image_id = image_stem_parts[1]
        image_id = f"{barcode}_{off_image_id}"
        image_url = generate_image_url(
            barcode, off_image_id, flavor=openfoodfacts_flavor
        )
    else:
        image_id = image_id
        barcode = ""
        off_image_id = ""
        image_url = ""

    sample_meta = SampleMeta(
        barcode=barcode,
        off_image_id=off_image_id,
        image_url=image_url,
    )
    return LLMImageExtractionSample(
        image_id=image_id,
        image=image,
        output=orjson.dumps(response).decode("utf-8"),
        meta=sample_meta,
    )
