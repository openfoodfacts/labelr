# train-yolo

This CLI tool allows you to train YOLO object detection models using datasets hosted on Hugging Face. It supports integration with Weights & Biases for experiment tracking.

## Run training

To launch a training, the easiest way is to use labelr CLI:

```bash
labelr train train-object-detection \
--hf-repo-id openfoodfacts/nutrition-table-detection \
--hf-trained-model-repo-id openfoodfacts/nutrition-test \
--wandb-project nutrition-table-detection \
--run-name yolov8 \
--hf-token {HF_TOKEN} \
--wandb-api-key {WANDB_API_KEY} \
--epochs 100 \
--imgsz 640 \
--batch 64
```

The command above will launch a training job on Google Cloud Batch using the docker image `europe-west9-docker.pkg.dev/robotoff/gcf-artifacts/train-yolo:latest`. The training code inside the docker image will use the provided environment variables to perform the training.

The following arguments are available:

- `--hf-repo-id` (**required**): The Hugging Face repository ID of the dataset to use for training.
- `--hf-trained-model-repo-id` (**required**): The Hugging Face repository ID where the trained model will be uploaded.
- `--hf-token` (**required**): Your Hugging Face API token, used to upload trained models.
- `--wandb-api-key` (**required**): Your Weights & Biases API key.
- `--wandb-project`: The Weights & Biases project name (default: "train-yolo").
- `--run-name`: Name of the run (default: `run-` suffixed with current datetime, e.g. "run-20250101-123456"). This will be used in Weights & Biases to track the training, and also as the name of the branch created in the Hugging Face repository to store the trained model.
- `--epochs`: Number of training epochs (default: 100).
- `--imgsz`: Image size for training (default: 640).
- `--batch`: Batch size for training (default: 64).


## Environment variables

The following environment variables can be set when running the training. Some are required:
 
WANDB_PROJECT=nutrition-table-detection -e WANDB_NAME=yolov8 -e HF_REPO_ID=openfoodfacts/nutrition-table-detection train-yolo:latest python3 main.py

- `HF_TOKEN` (**required**): Your Hugging Face API token, used to upload trained models.
- `HF_TRAINED_MODEL_REPO_ID` (**required**): The Hugging Face repository ID where the trained model will be uploaded.
- `HF_REPO_ID` (**required**): The Hugging Face repository ID of the dataset to use for training.
- `WANDB_API_KEY` (**required**): Your Weights & Biases API key.
- `RUN_NAME`: Name of the run (default: `run-` suffixed with current datetime, e.g. "run-20250101-123456"). This will be used in Weights & Biases to track the training, and also as the name of the branch created in the Hugging Face repository to store the trained model.
- `WANDB_PROJECT`: The Weights & Biases project name (default: "train-yolo").
- `EPOCHS`: Number of training epochs (default: 100).
- `IMGSZ`: Image size for training (default: 640).


## Publishing a new docker image

We use Google Cloud Batch for training: a docker container is launched on a VM with GPU to perform the training. Before being able to launch a training job, you need to publish a docker image containing the training code and its dependencies. This should be done only when there are changes to the training code or its dependencies.

To publish a new docker image for this package, run the following commands from the root of the package:

```bash
docker build . -t europe-west9-docker.pkg.dev/robotoff/gcf-artifacts/train-yolo:latest
docker push europe-west9-docker.pkg.dev/robotoff/gcf-artifacts/train-yolo:latest
```

## Testing locally

You can test the docker image locally, once the image is built:

```bash
docker run -e WANDB_API_KEY={YOUR_API_KEY} -e WANDB_PROJECT={PROJECT} -e WANDB_NAME={RUN_NAME} -e HF_REPO_ID={REPO_ID} train-yolo:latest python3 main.py
```