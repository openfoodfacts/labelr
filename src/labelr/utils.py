import io
from pathlib import Path

from google.cloud import storage
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


def download_image_from_gcs(image_uri: str) -> Image.Image:
    """Download an image from a Google Cloud Storage URI and return it as a
    PIL Image."""
    storage_client = storage.Client()
    bucket_name, blob_name = image_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    image_data = blob.download_as_bytes()
    return Image.open(io.BytesIO(image_data))


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
