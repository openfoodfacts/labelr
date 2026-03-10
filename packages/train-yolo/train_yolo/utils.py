import json
import os
from pathlib import Path


def generate_ultralytics_settings(root_dir: Path) -> dict:
    return {
        "settings_version": "0.0.6",
        "datasets_dir": f"{root_dir}/datasets",
        "weights_dir": f"{root_dir}/weights",
        "runs_dir": f"{root_dir}/runs",
        "uuid": "08c1ccdf367db40afac4e8d21426192fc60fab1eb920743fcb7daaf744cf1752",
        "sync": True,
        "api_key": "",
        "openai_api_key": "",
        "clearml": False,
        "comet": False,
        "dvc": False,
        "hub": False,
        "mlflow": False,
        "neptune": False,
        "raytune": False,
        "tensorboard": False,
        "wandb": True,
        "vscode_msg": False,
        "openvino_msg": False,
    }


def save_ultralytics_settings(root_dir: Path) -> None:
    ultralytics_settings = generate_ultralytics_settings(root_dir)
    settings_dir = root_dir / "Ultralytics"
    settings_dir.mkdir(exist_ok=True)
    (settings_dir / "settings.json").write_text(
        json.dumps(ultralytics_settings, indent=2)
    )


def check_envvar():
    if not os.getenv("HF_TOKEN"):
        raise ValueError(
            "HF_TOKEN environment variable not set. This is required to push the trained model to Hugging Face."
        )

    if not os.getenv("WANDB_API_KEY"):
        raise ValueError(
            "WANDB_API_KEY environment variable not set. This is required to log training runs to Weights & Biases."
        )
