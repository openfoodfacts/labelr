from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel, Field
import os

CONFIG_PATH_STR = os.getenv("LABELR_CONFIG_PATH", None)

CONFIG_PATH = (
    Path(CONFIG_PATH_STR).expanduser()
    if CONFIG_PATH_STR
    else Path("~").expanduser() / ".config/labelr/config.json"
)


def check_required_field(name: str, value: Any):
    """Check if a field if not null or empty, raise a BadParameter exception if it is."""
    if not value:
        raise typer.BadParameter(f"'{name}' is required.")


# validate_assignment allows to validate the model everytime it is updated
class LabelrConfig(BaseModel, validate_assignment=True):
    label_studio_url: str = Field(
        default="http://127.0.0.1:8080",
        description="URL of the Label Studio instance to use. Defaults to http://127.0.0.1:8080.",
    )
    label_studio_api_key: str | None = Field(
        default=None,
        description="API key for Label Studio.",
    )
    label_studio_project_id: int | None = Field(
        default=None, description="ID of the Label Studio project to use"
    )


def get_config() -> LabelrConfig:
    """Get labelr configuration.

    The configuration can come from (by order of precedence):
    - Environment variables
    - JSON file (see below)

    The configuration is stored in a JSON file at ~/.config/labelr/config.json.

    The following environment variables are supported:
    - LABELR_LABEL_STUDIO_URL
    - LABELR_LABEL_STUDIO_API_KEY
    """
    if CONFIG_PATH.exists():
        config = LabelrConfig.model_validate_json(CONFIG_PATH.read_bytes())

        if "LABELR_LABEL_STUDIO_URL" in os.environ:
            config.label_studio_url = os.environ["LABELR_LABEL_STUDIO_URL"]
        if "LABELR_LABEL_STUDIO_API_KEY" in os.environ:
            config.label_studio_api_key = os.environ["LABELR_LABEL_STUDIO_API_KEY"]
        if "LABELR_LABEL_STUDIO_PROJECT_ID" in os.environ:
            config.label_studio_project_id = int(
                os.environ["LABELR_LABEL_STUDIO_PROJECT_ID"]
            )
        return config
    else:
        return LabelrConfig()


def set_file_config(key: str, value: str):
    """Update the labelr configuration.

    The configuration is stored in a JSON file at ~/.config/labelr/config.json.
    """
    config = get_config()
    setattr(config, key, value)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(config.model_dump_json(indent=2))


config = get_config()
