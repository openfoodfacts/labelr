import io
from pathlib import Path

from google.cloud import storage
from openfoodfacts.images import download_image as _download_image
from openfoodfacts.utils import ImageDownloadItem
from PIL import Image


def parse_hf_repo_id(hf_repo_id: str) -> tuple[str, str]:
    """Parse the repo_id and the revision from a hf_repo_id in the format:
    `org/repo-name@revision`.

    Returns a tuple (repo_id, revision), with revision = 'main' if it
    was not provided.
    """
    if "@" in hf_repo_id:
        hf_repo_id, revision = hf_repo_id.split("@", 1)
    else:
        revision = "main"

    return hf_repo_id, revision


def download_image(
    image: str | tuple[str, str],
    *,
    error_raise: bool = True,
    return_struct: bool = False,
    **kwargs,
) -> Image.Image | ImageDownloadItem | None:
    """Download an image from a URL or GCS URI and return it as a PIL Image.
    Args:
        image (str | tuple[str, str]): The URL or GCS URI of the image.
        error_raise (bool): Whether to raise an error if the image cannot be
            downloaded.
        return_struct (bool): Whether to return an ImageDownloadItem struct
            instead of a PIL Image.
        **kwargs: Additional arguments to pass to the download function.
    Returns:
        Image.Image | ImageDownloadItem: The downloaded image as a PIL Image
            or an ImageDownloadItem struct.
    """
    if isinstance(image, str) and image.startswith("gs://"):
        return download_image_from_gcs(image, return_struct=return_struct, **kwargs)
    return _download_image(
        image,
        error_raise=error_raise,
        return_struct=return_struct,
        **kwargs,
    )


def download_image_from_gcs(
    image_uri: str, client: storage.Client | None = None, return_struct: bool = False
) -> Image.Image | ImageDownloadItem:
    """Download an image from a Google Cloud Storage URI and return it as a
    PIL Image.

    Args:
        image_uri (str): The GCS URI of the image
            (e.g., gs://bucket_name/path/to/image.jpg).
        client (storage.Client | None): An optional Google Cloud Storage
            client. If not provided, a new client will be created.
    """
    if client is None:
        client = storage.Client()

    bucket_name, blob_name = image_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    image_data = blob.download_as_bytes()
    pil_image = Image.open(io.BytesIO(image_data))

    if return_struct:
        return ImageDownloadItem(
            url=image_uri,
            image=pil_image,
            error=None,
        )
    return pil_image


class PathWithContext:
    """A context manager that yields a Path object.

    This is useful to have a common interface with tempfile.TemporaryDirectory
    without actually creating a temporary directory.
    """

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass
