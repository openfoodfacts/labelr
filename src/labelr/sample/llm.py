import typing
from collections.abc import Iterator
from pathlib import Path

import datasets
import orjson
from PIL import Image
from pydantic import BaseModel, Field

from labelr.sample.common import SampleMeta
from labelr.utils import download_image


class LLMImageExtractionSample(BaseModel):
    class Config:
        # required to allow PIL Image type
        arbitrary_types_allowed = True

    image_id: str = Field(
        ...,
        description="unique ID for the image. For Open Food Facts images, it follows the "
        "format `barcode:imgid`",
    )
    image: Image.Image = Field(..., description="Image to extract information from")
    output: str | None = Field(..., description="Expected response of the LLM")
    meta: SampleMeta = Field(..., description="Metadata associated with the sample")


HF_DS_LLM_IMAGE_EXTRACTION_FEATURES = datasets.Features(
    {
        "image_id": datasets.Value("string"),
        "image": datasets.features.Image(),
        "output": datasets.features.Value("string"),
        "meta": {
            "barcode": datasets.Value("string"),
            "off_image_id": datasets.Value("string"),
            "image_url": datasets.Value("string"),
        },
    }
)


def load_llm_image_extraction_dataset_from_jsonl(
    dataset_path: Path, **kwargs
) -> Iterator[LLMImageExtractionSample]:
    """Load a Hugging Face dataset for LLM image extraction from a JSONL file.

    Args:
        dataset_path (Path): Path to the JSONL dataset file.
        **kwargs: Additional keyword arguments to pass to the image downloader.
    Yields:
        Iterator[LLMImageExtractionSample]: Iterator of LLM image extraction
            samples.
    """
    with dataset_path.open("r") as f:
        for line in f:
            item = orjson.loads(line)
            image_id = item["image_id"]
            image_url = item["image_url"]
            image = typing.cast(Image.Image, download_image(image_url, **kwargs))
            barcode = item.pop("barcode", None)
            off_image_id = item.pop("off_image_id", None)
            output = item.pop("output", None)
            meta = SampleMeta(
                barcode=barcode,
                off_image_id=off_image_id,
                image_url=image_url,
            )
            sample = LLMImageExtractionSample(
                image_id=image_id,
                image=image,
                output=output,
                meta=meta,
            )
            yield sample
