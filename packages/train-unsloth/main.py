import functools
from pathlib import Path
from typing import Annotated, Any

import orjson
import tqdm
import typer

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
# trl should be imported after unsloth
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from trl import SFTConfig, SFTTrainer

JSONType = dict[str, Any]

app = typer.Typer(pretty_exceptions_enable=False)


def get_config(ds_repo_id: str):
    config_path = hf_hub_download(ds_repo_id, "config.json", repo_type="dataset")
    return orjson.loads(Path(config_path).read_bytes())


def get_full_instructions(instructions: str, json_schema: JSONType):
    json_schema_str = orjson.dumps(json_schema).decode("utf-8")
    return f"{instructions}\n\nResponse must be formatted as JSON, and follow this JSON schema:\n{json_schema_str}"


def run_on_validation_set(val_ds: Dataset, model: FastVisionModel, tokenizer: Any):
    FastVisionModel.for_inference(model)  # Enable for inference!

    print("Running on validation set to verify model...")
    for sample in tqdm.tqdm(val_ds, desc="validation samples"):
        input_text = tokenizer.apply_chat_template(
            sample["messages"], add_generation_prompt=True
        )
        inputs = tokenizer(
            sample["image"],
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)
        _ = model.generate(inputs, max_new_tokens=4096, use_cache=True)


@app.command()
def main(
    ds_repo_id: Annotated[str, typer.Option(..., help="The HF dataset repo ID")],
    output_repo_id: Annotated[
        str, typer.Option(..., help="The HF repo ID to push the trained model to")
    ],
    hf_token: Annotated[str, typer.Option(..., help="The HF token")],
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
    ] = 10,
):
    model, tokenizer = FastVisionModel.from_pretrained(
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
                        {"type": "image"},
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
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer, max_seq_length=8192),
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
            report_to="none",  # For Weights and Biases
            # The following fields are required for vision finetuning
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=None,
        ),
    )
    trainer.train()
    model.push_to_hub(output_repo_id, token=hf_token)
    tokenizer.push_to_hub(output_repo_id, token=hf_token)
    converted_val_dataset = val_ds.map(
        functools.partial(convert_to_conversation, train=False),
        remove_columns=["output"],
    )
    run_on_validation_set(converted_val_dataset, model, tokenizer)


if __name__ == "__main__":
    app()
