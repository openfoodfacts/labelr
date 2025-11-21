# Training an object detection model with YOLO and Labelr

In this tutorial, we will walk through the steps to create a dataset for object detection using Labelr and Label Studio, export the annotated data to a Hugging Face dataset, and launch a training job for a YOLO object detection model using the `labelr` CLI.

## Create the Label Studio project

The first step is to create a Label Studio project for object detection. We need to generate a project configuration file, which can be done automatically for object detection tasks.

```bash
labelr ls create-config --labels 'brand' --labels 'label' --output-file label_config.xml
```

`ls` stands for Label Studio in the CLI.
Here, we have two labels: `brand` and `label`. The output configuration file will be saved as `label_config.xml`.

Then, you can create a project on Label Studio with the following command:

```bash
labelr ls create --title universal-logo-detector --api-key ${LABEL_STUDIO_API_KEY} --config-file label_config.xml
```

You need a valid Label Studio API key, which can be found on your Account page.

The project should be visible in your [Label Studio instance](https://annotate.openfoodfacts.org/projects/), and you can note the project ID (displayed in the URL when clicking on the project) for later use. We will assume from now that the project ID is stored in the `PROJECT_ID` environment variable.


#### Create a dataset file

If you have a list of images, for an object detection task, you can quickly create a dataset file with the following command:

```bash
labelr ls create-dataset-file --input-file image_urls.txt --output-file dataset.json
```

where `image_urls.txt` is a file containing the URLs of the images, one per line, and `dataset.json` is the output file.


#### Import the data

Next, import the generated data to a project with the following command:

```bash
labelr ls import-data --project-id ${PROJECT_ID} --dataset-path dataset.json
```

All tasks should now be visible in your Label Studio project.

#### Pre-annotate the data

To accelerate annotation, you can pre-annotate the images with an object detection model. We support two pre-annotation backends:

- Triton: you need to have a Triton server running with a model that supports object detection. The object detection model is expected to be a YOLO model. You can set the URL of the Triton server with the `--triton-url` CLI option.

- Ultralytics: you can use the [Yolo-World model from Ultralytics](https://github.com/ultralytics/ultralytics), Ultralytics should be installed in the same virtualenv.

To pre-annotate the data with Triton, use the following command:

```bash
labelr ls add-prediction --project-id ${PROJECT_ID} --backend ultralytics --labels 'brand' --labels 'label' --label-mapping '{"price tag": "price-tag"}'
```

where `labels` is the list of labels to use for the object detection task (you can add as many labels as you want).

If your label name contains spaces or special characters, you can use the `--label-mapping` option to map the labels from the model to the labels of the project (ex: `--label-mapping '{"my label": "my-tag"}'`).

By default, for Ultralytics, the `yolov8x-worldv2.pt` model is used. You can change the model by setting the `--model-name` CLI option.

#### Annotate the data

You need to validate manually all pre-annotations and correct them if necessary. Once the data is fully annotated, you can proceed to export the data.

## Export annotated data

To export the annotated data from Label Studio to a Hugging Face dataset, you can use the following command:

```bash
labelr datasets export \
--from ls \
--to hf \
--repo-id openfoodfacts/universal-logo-detector \
--merge-labels \
--project-id ${PROJECT_ID} \
--label-names brand,label
```

The required arguments are:
- `--from ls`: specifies that the source of the data is Label Studio.
- `--to hf`: specifies that the destination is a Hugging Face dataset.
- `--repo-id`: the Hugging Face repository ID where the dataset will be uploaded.
- `--project-id`: the ID of the Label Studio project containing the annotated data.
- `--label-names`: a comma-separated list of label names to include in the dataset

Optionally, you can use the `--merge-labels` flag to merge all labels into a single label called "object".

Then, we create a tag for the dataset:

```bash
hf repo tag create --repo-type dataset --revision main openfoodfacts/universal-logo-detector v1.3
```

## Launch training job

To launch a training job for the YOLO object detection model using the exported dataset, you can use the following command:

```bash
labelr train train-object-detection \
--hf-repo-id openfoodfacts/universal-logo-detector@v1.3 \
--wandb-project universal-logo-detector \
--run-name yolov8n-100-epochs-imgsz-640 \
--hf-trained-model-repo-id openfoodfacts/universal-logo-detector \
--hf-token ${HF_TOKEN} \
--wandb-api-key ${WANDB_API_KEY} \
--model-name yolov8n.pt \
--epochs 100 \
--imgsz 640 \
--batch 64
```

You need to provide a valid [wandb](https://wandb.ai/) API key (for model tracking) and Hugging Face Hub token (for pushing the model to the Hub).

The following arguments are available:

- `--hf-repo-id` (**required**): The Hugging Face repository ID of the dataset to use for training. Here, we specify the dataset tag `v1.3` to use a specific version of the dataset.
- `--hf-trained-model-repo-id` (**required**): The Hugging Face repository ID where the trained model will be uploaded.
- `--hf-token` (**required**): Your Hugging Face API token, used to upload trained models.
- `--wandb-api-key` (**required**): Your Weights & Biases API key.
- `--wandb-project`: The Weights & Biases project name (default: "train-yolo").
- `--run-name`: Name of the run (default: `run-` suffixed with current datetime, e.g. "run-20250101-123456"). This will be used in Weights & Biases to track the training, and also as the name of the branch created in the Hugging Face repository to store the trained model.
- `--epochs`: Number of training epochs (default: 100).
- `--imgsz`: Image size for training (default: 640).
- `--batch`: Batch size for training (default: 64).