from typing import Literal

import torch
import ultralytics
from huggingface_hub import ModelCard, ModelCardData

CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for {{ model_id | default("Model ID", true) }}

{% if wandb_run_url %}[Wandb tracking run]({{ wandb_run_url }}){% endif %}

This {{ model_type }} model was fine-tuned using the Ultralytics YOLO library.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** {{ developers | default("[More Information Needed]", true)}}
- **Model type:** {{ model_type }}
- **License:** {{ license | default("[More Information Needed]", true)}}
- **Finetuned from model [optional]:** {{ base_model | default("[More Information Needed]", true)}}

## Training Details

### Training Data

The model was fine-tuned using the following dataset: [{{ dataset_repo_id }}](https://huggingface.co/datasets/{{ dataset_repo_id }}) (revision: `{{ dataset_revision }}`).

### Training Procedure

Dependency versions:

- ultralytics: {{ ultralytics_version | default("[More Information Needed]", true)}}
- pytorch: {{ pytorch_version | default("[More Information Needed]", true)}}

#### Training Hyperparameters

- **Epochs:** {{ training_epochs | default("[More Information Needed]", true)}}
- **Batch size:** {{ training_batch_size | default("[More Information Needed]", true)}}
- **Image size:** {{ training_imgsz | default("[More Information Needed]", true)}}

## Evaluation

The following evaluation metrics were obtained after training the model:

{% for metric_name, metric_value in metrics_results_dict["pytorch"].items() %}
- **{{ metric_name }}:** {{ metric_value }}
{% endfor %}

### Evaluation on exported models

The model was also evaluated after exporting to ONNX and TensorRT formats. The following metrics were obtained:

{% for format_name in ["onnx"] %}
#### {{ format_name | upper }} export
{% for metric_name, metric_value in metrics_results_dict[format_name].items() %}
- **{{ metric_name }}:** {{ metric_value }}
{% endfor %}
{% endfor %}


## Files

Most files stored on the repo are standard files created during training with the Ultralytics YOLO library.

What was added:

- an ONNX export of the trained model (best model), stored in `weights/model.onnx`.
- a Parquet file containing predictions on the full dataset, stored in `predictions.parquet`.
- metrics JSON files for each exported model format, stored in `metrics_*.json`:
    - `metrics.json`: metrics for the original PyTorch model
    - `metrics_onnx.json`: metrics for the ONNX exported model
"""


def create_model_card(
    dataset_repo_id: str,
    dataset_revision: str,
    model_id: str,
    base_model: str,
    training_epochs: int,
    training_imgsz: int,
    training_batch_size: int,
    metrics_results_dict: dict[str, dict[str, float]],
    task: Literal["classify", "detect"],
    license: str = "agpl-3.0",
    wandb_run_url: str | None = None,
) -> ModelCard:
    card_data = ModelCardData(
        license=license,
        library_name="ultralytics",
        pipeline_tag="object-detection" if task == "detect" else "image-classification",
        datasets=[dataset_repo_id],
        base_model=base_model,
    )
    return ModelCard.from_template(
        card_data,
        template_str=CARD_TEMPLATE,
        model_id=model_id,
        developers="Open Food Facts",
        model_type="object detection" if task == "detect" else "image classification",
        dataset_repo_id=dataset_repo_id,
        dataset_revision=dataset_revision,
        metrics_results_dict=metrics_results_dict,
        training_epochs=training_epochs,
        training_imgsz=training_imgsz,
        training_batch_size=training_batch_size,
        ultralytics_version=ultralytics.__version__,
        pytorch_version=torch.__version__,
        wandb_run_url=wandb_run_url,
    )
