from io import BytesIO
from pathlib import Path
from typing import Any

import orjson
import tqdm
from datasets import Dataset
from more_itertools import chunked
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams

JSONType = dict[str, Any]


def get_adapter_config(model_checkpoint_path: Path) -> JSONType:
    """Load the adapter configuration from the model checkpoint directory.

    Args:
        model_checkpoint_path (Path): The path to the model checkpoint
        directory.
    Returns:
        JSONType: The adapter configuration.
    """
    adapter_config_path = model_checkpoint_path / "adapter_config.json"
    with adapter_config_path.open("r") as f:
        adapter_config = orjson.loads(f.read())
    return adapter_config


def run_on_validation_set(
    base_model: str,
    val_ds: "Dataset",
    lora_checkpoint_dir: Path,
    output_path: Path,
    json_schema: JSONType,
    max_seq_length: int,
    batch_size: int = 4,
    enforce_schema: bool = True,
    max_lora_rank: int = 16,
) -> None:
    """Run the model on the validation set and save the outputs to a JSONL
    file.

    We use vLLM for inference. We currently assume the base model is a
    Qwen3-VL model.

    Args:
        base_model (str): The base model to use for inference. The LoRA
            weights will be applied on top of this model.
        val_ds (Dataset): The validation dataset, already formatted with the
            chat template.
        lora_checkpoint_dir (Path): The path to the LoRA checkpoint directory.
        output_path (Path): The path to the output JSONL file.
        json_schema (JSONType): The JSON schema to use for structured outputs.
        max_seq_length (int): The maximum sequence length for the model.
        batch_size (int, optional): The batch size to use for inference.
            Defaults to 4.
        enforce_schema (bool, optional): Whether to enforce the JSON schema
            during inference. Defaults to True.
        max_lora_rank (int, optional): The maximum LoRA rank to use. Defaults
            to 16.
    """
    llm = LLM(
        model=base_model,
        enable_lora=True,
        # Applying configuration tips for Qwen-VL models:
        # https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html#configuration-tips
        limit_mm_per_prompt={"video": 0},
        # The validation dataset is comprised of unique images, disable
        # multimodal caching
        mm_processor_cache_gb=0,
        max_model_len=max_seq_length,
        max_lora_rank=max_lora_rank,
    )

    if enforce_schema:
        # We guide the model to produce structured JSON outputs using the
        # provided JSON schema.
        structured_outputs_params = StructuredOutputsParams(json=json_schema)
    else:
        structured_outputs_params = None

    sampling_params = SamplingParams(
        structured_outputs=structured_outputs_params, max_tokens=8192
    )

    def reformat_conversation(conversation: JSONType) -> JSONType:
        image_bytes = conversation[-1]["content"][-1]["image_pil"]["bytes"]
        conversation[-1]["content"][-1] = {
            "image_pil": Image.open(BytesIO(image_bytes)),
            "type": "image_pil",
        }
        return conversation

    with output_path.open("w") as f:
        # Process the validation dataset
        for samples in tqdm.tqdm(
            chunked(val_ds, batch_size), desc="Running inference on validation set"
        ):
            # Extract the image and prompt from the sample
            conversations = [
                reformat_conversation(sample["messages"]) for sample in samples
            ]

            outputs = llm.chat(
                conversations,
                sampling_params=sampling_params,
                lora_request=LoRARequest(
                    lora_name="train-unsloth",
                    lora_int_id=1,
                    lora_path=str(lora_checkpoint_dir),
                ),
            )

            for sample, output in zip(samples, outputs):
                generated_text = output.outputs[0].text
                # Store the output
                line = {"image_id": sample["image_id"], "output": generated_text}
                f.write(orjson.dumps(line).decode("utf-8") + "\n")
