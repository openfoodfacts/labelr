from pydantic import BaseModel, Field


class SampleMeta(BaseModel):
    barcode: str | None = Field(
        ..., description="The barcode of the product, if applicable"
    )
    off_image_id: str | None = Field(
        ...,
        description="The Open Food Facts image ID associated with the image, if applicable",
    )
    image_url: str | None = Field(
        ..., description="The URL of the image, if applicable"
    )
