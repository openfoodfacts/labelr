import functools
import logging
import pickle
import tempfile
import typing
from collections.abc import Iterator
from pathlib import Path

import datasets
import tqdm
from PIL import Image, ImageOps

from labelr.export.common import _pickle_sample_generator
from labelr.sample.llm import (
    HF_DS_LLM_IMAGE_EXTRACTION_FEATURES,
    LLMImageExtractionSample,
)
from labelr.utils import PathWithContext

logger = logging.getLogger(__name__)


def export_to_hf_llm_image_extraction(
    sample_iter: Iterator[LLMImageExtractionSample],
    split: str,
    repo_id: str,
    revision: str = "main",
    tmp_dir: Path | None = None,
    image_max_size: int | None = None,
) -> None:
    """Export LLM image extraction samples to a Hugging Face dataset.

    Args:
        sample_iter (Iterator[LLMImageExtractionSample]): Iterator of samples
            to export.
        split (str): Name of the dataset split (e.g., 'train', 'val').
        repo_id (str): Hugging Face repository ID to push the dataset to.
        revision (str): Revision (branch, tag or commit) to use for the
            Hugging Face Datasets repository.
        tmp_dir (Path | None): Temporary directory to use for intermediate
            files. If None, a temporary directory will be created
            automatically.
        image_max_size (int | None): Maximum size (in pixels) for the images.
    """
    logger.info(
        "Repo ID: %s, revision: %s, split: %s, tmp_dir: %s, image_max_size: %s",
        repo_id,
        revision,
        split,
        tmp_dir,
        image_max_size,
    )

    tmp_dir_with_context: PathWithContext | tempfile.TemporaryDirectory
    if tmp_dir:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir_with_context = PathWithContext(tmp_dir)
    else:
        tmp_dir_with_context = tempfile.TemporaryDirectory()

    with tmp_dir_with_context as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for sample in tqdm.tqdm(sample_iter, desc="samples"):
            image = sample.image
            # Rotate image according to exif orientation using Pillow
            image = typing.cast(Image.Image, ImageOps.exif_transpose(image))

            if image_max_size is not None:
                if image.height > image_max_size or image.width > image_max_size:
                    image.thumbnail(
                        (image_max_size, image_max_size),
                        Image.Resampling.LANCZOS,
                    )
            image_id = sample.image_id
            json_sample = {
                "image_id": image_id,
                "image": image,
                "meta": {
                    k: v for k, v in sample.meta.model_dump().items() if v is not None
                },
                "output": sample.output,
            }
            # Save output as pickle
            with open(tmp_dir / f"{split}_{image_id}.pkl", "wb") as f:
                pickle.dump(json_sample, f)

        hf_ds = datasets.Dataset.from_generator(
            functools.partial(_pickle_sample_generator, tmp_dir),
            features=HF_DS_LLM_IMAGE_EXTRACTION_FEATURES,
        )
        hf_ds.push_to_hub(repo_id, split=split, revision=revision)
