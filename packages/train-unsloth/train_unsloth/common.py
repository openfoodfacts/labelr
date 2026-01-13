from pathlib import Path
from typing import Any

import orjson
from huggingface_hub import hf_hub_download
from PIL import Image

JSONType = dict[str, Any]


def get_config(ds_repo_id: str):
    config_path = hf_hub_download(ds_repo_id, "config.json", repo_type="dataset")
    return orjson.loads(Path(config_path).read_bytes())


def get_full_instructions(instructions: str, json_schema: JSONType):
    json_schema_str = orjson.dumps(json_schema).decode("utf-8")
    return f"{instructions}\n\nResponse must be formatted as JSON, and follow this JSON schema:\n{json_schema_str}"


def convert_to_conversation(
    sample: JSONType,
    instructions: str,
    train: bool = True,
    image_max_size: int | None = None,
):
    """Convert a dataset sample to a conversation format.

    Args:
        sample: A dataset sample containing "image" and "output" fields.
        instructions: The instructions to include in the conversation.
        train: Whether the conversion is for training (includes output) or
            validation (excludes output).
        image_max_size: The maximum size (height or width) of the images after
            resizing. If None, no resizing is performed.
    Returns:
        A dictionary with a "messages" field containing the conversation.
    """
    image = sample["image"]
    if image_max_size is not None:
        # Resize the image while maintaining aspect ratio
        image.thumbnail((image_max_size, image_max_size), Image.Resampling.LANCZOS)

    if train:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["output"]}],
            },
        ]
    else:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions},
                    {"type": "image_pil", "image_pil": image},
                ],
            }
        ]
    return {"messages": conversation}
