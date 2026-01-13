import functools
import typing
from pathlib import Path
from typing import Annotated, Any

import typer

JSONType = dict[str, Any]

app = typer.Typer(pretty_exceptions_enable=False)


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
        typer.Option(
            ...,
            help="Whether to finetune the vision layers of the model. Defaults to False, "
            "as vLLM currently doesn't support LoRA adapters added to vision layers (as of v0.13).",
        ),
    ] = False,
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
    from unsloth import FastVisionModel  # isort:skip
    from unsloth.trainer import UnslothVisionDataCollator  # isort:skip
    from datasets import Dataset, load_dataset
    from train_unsloth.common import (
        convert_to_conversation,
        get_config,
        get_full_instructions,
    )
    from trl import SFTConfig, SFTTrainer

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

    train_ds = typing.cast(Dataset, ds["train"])

    if shuffle_dataset:
        train_ds = train_ds.shuffle()
    if max_samples is not None:
        train_ds = train_ds.select(list(range(max_samples)))

    ds_config = get_config(ds_repo_id=ds_repo_id)
    instructions = ds_config["instructions"]
    json_schema = ds_config["json_schema"]
    full_instructions = get_full_instructions(instructions, json_schema)

    converted_train_dataset = typing.cast(
        Dataset,
        train_ds.map(
            functools.partial(
                convert_to_conversation, instructions=full_instructions, train=True
            ),
            remove_columns=["image", "output"],
        ),
    )

    # Enable training mode
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,  # type: ignore
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


@app.command()
def validate(
    ds_repo_id: Annotated[str, typer.Option(..., help="The HF dataset repo ID")],
    lora_repo_id: Annotated[
        str, typer.Option(..., help="The HF repo ID where the LoRA adapters are stored")
    ],
    output_path: Annotated[
        Path,
        typer.Option(..., help="The path to the output JSONL file"),
    ],
    base_model: Annotated[
        str,
        typer.Option(
            ...,
            help="The base model associated with the LoRA adapters.",
        ),
    ] = "unsloth/Qwen3-VL-8B-Instruct",
    batch_size: Annotated[
        int,
        typer.Option(..., help="The per-device batch size to use during validation"),
    ] = 8,
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
    from datasets import Dataset, load_dataset
    from huggingface_hub import snapshot_download, upload_file
    from train_unsloth.common import (
        convert_to_conversation,
        get_config,
        get_full_instructions,
    )
    from train_unsloth.validate import run_on_validation_set

    typer.echo("Running model on validation set...")

    val_ds = typing.cast(
        Dataset,
        load_dataset(
            ds_repo_id,
            columns=["image", "output", "image_id"],
        )["val"],
    )
    ds_config = get_config(ds_repo_id=ds_repo_id)
    instructions = ds_config["instructions"]
    json_schema = ds_config["json_schema"]
    full_instructions = get_full_instructions(instructions, json_schema)

    converted_val_dataset = typing.cast(
        Dataset,
        val_ds.map(
            functools.partial(
                convert_to_conversation,
                instructions=full_instructions,
                train=False,
                image_max_size=image_max_size,
            ),
            remove_columns=["image", "output"],
        ),
    )

    typer.echo("Downloading LoRA weights...")
    lora_checkpoint_path = snapshot_download(repo_id=lora_repo_id, repo_type="model")
    run_on_validation_set(
        base_model=base_model,
        val_ds=converted_val_dataset,
        lora_checkpoint_dir=Path(lora_checkpoint_path),
        output_path=output_path,
        json_schema=json_schema,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )

    typer.echo("Uploading validation outputs to the Hub...")
    # Upload the validation outputs to the Hub
    upload_file(
        path_or_fileobj=output_path,
        path_in_repo="validation_output.jsonl",
        repo_id=lora_repo_id,
        repo_type="model",
    )


if __name__ == "__main__":
    app()
