import functools
import tempfile
from pathlib import Path
from typing import Annotated, Any

# trl should be imported after unsloth
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
# Rest of the imports
import orjson
import tqdm
import typer
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, upload_file
from more_itertools import chunked
from trl import SFTConfig, SFTTrainer

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams

JSONType = dict[str, Any]

app = typer.Typer(pretty_exceptions_enable=False)


def get_config(ds_repo_id: str):
    config_path = hf_hub_download(ds_repo_id, "config.json", repo_type="dataset")
    return orjson.loads(Path(config_path).read_bytes())


def get_full_instructions(instructions: str, json_schema: JSONType):
    json_schema_str = orjson.dumps(json_schema).decode("utf-8")
    return f"{instructions}\n\nResponse must be formatted as JSON, and follow this JSON schema:\n{json_schema_str}"


def run_on_validation_set(
    base_model: str,
    val_ds: Dataset,
    lora_checkpoint_dir: Path,
    output_path: Path,
    json_schema: JSONType,
    batch_size: int = 4,
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
        batch_size (int, optional): The batch size to use for inference.
            Defaults to 4.
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
    )

    # We guide the model to produce structured JSON outputs using the provided
    # JSON schema.
    structured_outputs_params = StructuredOutputsParams(json=json_schema)
    sampling_params = SamplingParams(structured_outputs=structured_outputs_params)

    # Process the validation dataset
    outputs = []
    for samples in tqdm.tqdm(
        chunked(val_ds, batch_size), desc="Running inference on validation set"
    ):
        # Extract the image and prompt from the sample
        conversations = [sample["messages"] for sample in samples]

        # Run inference using vLLM
        try:
            outputs = llm.chat(
                conversations,
                sampling_params=sampling_params,
                lora_request=LoRARequest(
                    lora_name="train-unsloth",
                    lora_int_id=1,
                    lora_path=str(lora_checkpoint_dir),
                ),
            )
            generated_texts = [output.outputs[0].text for output in outputs]
        except Exception as e:
            print(f"Error generating output for prompt: {e}")
        else:
            for sample, generated_text in zip(samples, generated_texts):
                # Store the output
                outputs.append(
                    {"image_id": sample["image_id"], "output": generated_text}
                )

    # Save the outputs to a JSONL file
    with open(output_path, "w") as f:
        for output in outputs:
            f.write(orjson.dumps(output).decode("utf-8") + "\n")


@app.command()
def train(
    ds_repo_id: Annotated[str, typer.Option(..., help="The HF dataset repo ID")],
    output_repo_id: Annotated[
        str, typer.Option(..., help="The HF repo ID to push the trained model to")
    ],
    base_model: Annotated[
        str,
        typer.Option(
            ...,
            help="The base model to fine-tune. This must be a Unsloth 4-bit model.",
        ),
    ] = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    finetune_vision_layers: Annotated[
        bool,
        typer.Option(..., help="Whether to finetune the vision layers of the model"),
    ] = True,
    finetune_language_layers: Annotated[
        bool,
        typer.Option(..., help="Whether to finetune the language layers of the model"),
    ] = True,
    finetune_attention_layers: Annotated[
        bool,
        typer.Option(..., help="Whether to finetune the attention layers of the model"),
    ] = True,
    finetune_mlp_layers: Annotated[
        bool,
        typer.Option(..., help="Whether to finetune the MLP layers of the model"),
    ] = True,
    lora_r: Annotated[
        int,
        typer.Option(..., help="The LoRA rank to use for fine-tuning"),
    ] = 16,
    lora_alpha: Annotated[
        int,
        typer.Option(..., help="The LoRA alpha to use for fine-tuning"),
    ] = 16,
    lora_dropout: Annotated[
        float,
        typer.Option(..., help="The LoRA dropout to use for fine-tuning"),
    ] = 0.0,
    use_rslora: Annotated[
        bool,
        typer.Option(..., help="Whether to use Rank Stabilized LoRA"),
    ] = False,
    per_device_train_batch_size: Annotated[
        int,
        typer.Option(..., help="The per-device training batch size"),
    ] = 8,
    gradient_accumulation_steps: Annotated[
        int,
        typer.Option(..., help="The number of gradient accumulation steps"),
    ] = 4,
    warmup_steps: Annotated[
        int,
        typer.Option(..., help="The number of warmup steps"),
    ] = 5,
    learning_rate: Annotated[
        float,
        typer.Option(..., help="The learning rate"),
    ] = 2e-4,
    num_train_epochs: Annotated[
        int,
        typer.Option(..., help="The number of training epochs"),
    ] = 1,
    max_steps: Annotated[
        int,
        typer.Option(..., help="The maximum number of training steps. Ignored if -1"),
    ] = -1,
    max_samples: Annotated[
        int | None,
        typer.Option(
            ...,
            help="The maximum number of samples to use from the dataset. If None, use all samples",
        ),
    ] = None,
    shuffle_dataset: Annotated[
        bool,
        typer.Option(..., help="Whether to shuffle the dataset"),
    ] = True,
    optim: Annotated[
        str,
        typer.Option(..., help="The optimizer to use"),
    ] = "adamw_8bit",
    weight_decay: Annotated[
        float,
        typer.Option(..., help="The weight decay to use"),
    ] = 0.001,
    logging_steps: Annotated[
        int,
        typer.Option(..., help="The number of logging steps"),
    ] = 1,
    push_to_hub: Annotated[
        bool,
        typer.Option(..., help="Whether to push the trained model to the HF Hub"),
    ] = True,
    train: Annotated[
        bool,
        typer.Option(..., help="Whether to run training"),
    ] = True,
    image_max_size: Annotated[
        int,
        typer.Option(
            ...,
            help="The maximum size (height or width) of the images after resizing",
        ),
    ] = 1024,
    max_seq_length: Annotated[
        int,
        typer.Option(..., help="The maximum sequence length for the model"),
    ] = 8192,
):
    model, processor = FastVisionModel.from_pretrained(
        base_model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=finetune_vision_layers,
        finetune_language_layers=finetune_language_layers,
        finetune_attention_modules=finetune_attention_layers,
        finetune_mlp_modules=finetune_mlp_layers,
        r=lora_r,
        # Recommended alpha == r at least
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        # rank stabilized LoRA
        use_rslora=use_rslora,
        # LoftQ
        loftq_config=None,
        # target_modules = "all-linear", # Optional now! Can specify a list if
        # needed
    )
    ds = load_dataset(
        ds_repo_id,
        columns=["image", "output"],
    )

    train_ds = ds["train"]
    val_ds = ds["val"]

    if shuffle_dataset:
        train_ds = train_ds.shuffle()
    if max_samples is not None:
        train_ds = train_ds.select(list(range(max_samples)))

    ds_config = get_config(ds_repo_id=ds_repo_id)
    instructions = ds_config["instructions"]
    json_schema = ds_config["json_schema"]
    full_instructions = get_full_instructions(instructions, json_schema)

    def convert_to_conversation(sample, train: bool = True):
        if train:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_instructions},
                        {"type": "image", "image": sample["image"]},
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
                        {"type": "text", "text": full_instructions},
                        {"type": "image_pil", "image_pil": sample["image"]},
                    ],
                }
            ]
        return {"messages": conversation}

    converted_train_dataset = train_ds.map(
        convert_to_conversation, remove_columns=["image", "output"]
    )

    # Enable training mode
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=UnslothVisionDataCollator(
            model,
            processor,
            max_seq_length=max_seq_length,
            # Resize images to limit VRAM consumption
            resize_dimension="max",
            resize=image_max_size,
        ),
        train_dataset=converted_train_dataset,
        args=SFTConfig(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            optim=optim,
            weight_decay=weight_decay,
            lr_scheduler_type="linear",
            output_dir="outputs",
            report_to="wandb",
            # The following fields are required for vision finetuning
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=None,
        ),
    )

    if train:
        trainer.train()

    if push_to_hub:
        model.push_to_hub(output_repo_id)
        processor.push_to_hub(output_repo_id)

    converted_val_dataset = val_ds.map(
        functools.partial(convert_to_conversation, train=False),
        remove_columns=["image", "output"],
    )

    with tempfile.TemporaryDirectory(suffix="-lora-weights-val") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        checkpoint_dir = tmp_dir / "model"
        typer.echo(f"Saving LoRA adapters weights to {checkpoint_dir}")
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        del model  # free up memory
        del processor
        torch.cuda.empty_cache()
        output_file = tmp_dir / "validation_output.jsonl"
        typer.echo("Running model on validation set...")
        run_on_validation_set(
            base_model=base_model,
            val_ds=converted_val_dataset,
            lora_checkpoint_dir=checkpoint_dir,
            output_path=output_file,
            json_schema=json_schema,
            batch_size=per_device_train_batch_size,
        )
        typer.echo("Uploading validation outputs to the Hub...")
        # Upload the validation outputs to the Hub
        upload_file(
            output_file,
            path_in_repo="validation_output.jsonl",
            repo_id=output_repo_id,
            repo_type="model",
        )


if __name__ == "__main__":
    app()
