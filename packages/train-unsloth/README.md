# Train Unsloth

This CLI tool allows you to train large visual language models (LVLM) using datasets hosted on Hugging Face.
It relies on [Unsloth](https://github.com/unslothai/unsloth) for training, and supports integration with Weights & Biases for experiment tracking.

The fine-tune model takes a single image as input, along with instructions and a JSON schema, and returns a structured response containing the extracted information. The training dataset on the Hub is expected to follow [this specific format](https://openfoodfacts.github.io/robotoff/references/datasets/llm-image-extraction/). The `train` command fetches a dataset from a Hugging Face dataset repo, fine-tunes a base model using this dataset and pushes the trained model to the Hub.

Currently, this CLI only supports fine-tuning the 4bits-quant version of Qwen3 VL models: some configuration options are specific to this architecture.

## Run training

To launch a training run, use the following command:

```bash
uv run main.py train --ds-repo-id DS_REPO_ID --output-repo-id MODEL_REPO_ID
```

with `DS_REPO_ID` being the Hugging Face dataset repo ID, and `MODEL_REPO_ID` being the Hugging Face model repo ID where the trained model will be pushed.

We only push the weights of the LoRA adapters to Hugging Face, as the base model is already hosted on Hugging Face.

Many options can be passed to the `train` command to customize the training process. Run `uv run main.py train --help` to see all available options.

The most important ones are:

- `--base-model`: the base model to fine-tune. Defaults to `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`.
- `--finetune-vision-layers`: whether to fine-tune the vision layers of the model. Defaults to `False`, as vLLM currently doesn't support LoRA adapters added to vision layers (as of v0.13).
- `--lora-r` and `--lora-alpha`: the LoRA rank and alpha to use for fine-tuning. Defaults to `16` and `16` respectively.
- `--per-device-train-batch-size` and `--gradient-accumulation-steps`: the per-device training batch size and the number of gradient accumulation steps. Defaults to `8` and `4` respectively (effective batch size of `32`).
- `--learning-rate`: the learning rate to use. Defaults to `2e-4`.
- `--num-train-epochs`: the number of training epochs. Defaults to `1`.


### Environment variables

The following environment variables can be set to customize the training process. Some are required with the default option values.

- `HF_TOKEN`: if `--push-to-hub` is provided (default), this token is used for authentication when pushing the model to the Hugging Face Hub.
- `WANDB_API_KEY`: Your WandB API key. Required if `--report-to` is set to `wandb` (default). 
- `WANDB_PROJECT`: the Weights & Biases project to log the training to.Optional, but recommended).
- `WANDB_NAME`: The name of your run. Recommended to distinguish your training runs. Optional, but recommended.
- `WANDB_TAGS=TAG1,TAG2`: Tags to add to your run. Useful to group run. Optional.

## Run validation

Once the model is fully trained, you can run the model on the validation split of the dataset.

```bash
uv run main.py validate --ds-repo-id DS_REPO_ID --lora-repo-id MODEL_REPO_ID --output-path val.jsonl --base-model BASE_MODEL_REPO_ID
```

with DS_REPO_ID being the Hugging Face dataset repo ID, MODEL_REPO_ID the Hugging Face model repo ID, containing the LoRA weights, and BASE_MODEL_REPO_ID the base model repo ID (defaults to `unsloth/Qwen3-VL-8B-Instruct`).

When running the `validate` command, we do the following:
- fetch the `val` split from the dataset
- load the base model using vLLM
- load the instructions and the JSON schema from the `config.json` file stored in MODEL_REPO_ID (the repo containing the LoRA weights and the generation config)
- run the model on the validation split, by providing as input (in this order):

    - the textual instructions
    - the full JSON schema
    - the image to extract data from

  By default, we use structured output mode to ensure that the model output respects the JSON schema, but you can disable this with `--no-enforce-schema`.
- save the results to `val.jsonl`, and push it to the Hugging Face model repo ID (if `--upload-to-hub` is provided, which is the default). The results are stored in the `predictions/val.json` folder on the Hub repo. This can be configured with the `output-path-in-repo` option.

If your model is on a specific branch of the HF repo, you can provide the `--lora-repo-revision` option to specify the revision. This will be used when fetching the model (weights + generation config) and when pushing the validation results.

## Future work

Being able to fully automate the process by submitting a job on a Cloud platform would be really useful (as what's done for `train-yolo` package).

A draft `Dockerfile` is provided in this repo, but it's not yet ready for production.