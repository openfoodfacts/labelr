import datasets
from PIL import Image
from pydantic import BaseModel, Field

from labelr.sample.common import SampleMeta


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
    output: str = Field(..., description="Expected response of the LLM")
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
