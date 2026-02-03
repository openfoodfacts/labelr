# Labelr

Labelr a command line interface that aims to provide a set of tools to help data scientists and machine learning engineers to deal with ML data annotation, data preprocessing and format conversion.

This project started as a way to automate some of the tasks we do at Open Food Facts to manage data at different stages of the machine learning pipeline.

The CLI currently is integrated with Label Studio (for data annotation), Ultralytics (for object detection) and Hugging Face (for model and dataset storage). It only works with some specific tasks (object detection and image classification only for now), but it's meant to be extended to other tasks in the future.

For object detection and image classification models, it currently allows to:

- create Label Studio projects
- upload images to Label Studio
- pre-annotate the tasks either with an existing object detection model, or with a zero-shot model (Yolo-World or SAM), using Ultralytics
- perform data quality checks on Label Studio datasets
- export the data to Hugging Face or to local disk
- train the model on Google Batch (for object detection only)
- visualize the model predictions and compare them with the ground truth, using [Fiftyone](https://docs.voxel51.com/user_guide/index.html).

Labelr also support managing datasets for fine-tuning large visual language models. It currently only support a single task: structured extraction (JSON) from a single image.
The following features are supported:

- creating training datasets using Google Gemini Batch, from a list of images, textual instructions and a JSON schema
- uploading the dataset to Hugging Face
- fixing manually or automatically the model output using [Directus](https://directus.io/), a headless CMS used to manage the structured output
- export the dataset to Hugging Face

In addition, Labelr comes with two scripts that can be used to train ML models:

- in `packages/train-yolo`: the `main.py` script can be used to train an object detection model using Ultralytics. The training can be fully automatized on Google Batch, and Labelr provides a CLI to launch Google Batch jobs.
- in `packages/train-unsloth`: the `main.py` script can be used to train a visual language model using Unsloth. The training is not yet automatized on Google Batch, but the script can be used to train the model locally.

## Installation

Python 3.10 or higher is required to run this CLI.

To install the CLI, simply run:

```bash
pip install labelr
```
We recommend to install the CLI in a virtual environment. You can either use pip or conda for that.

There are two optional dependencies that you can install to use the CLI:
- `ultralytics`: pre-annotate object detection datasets with an ultralytics model (yolo, yolo-world)
- `fiftyone`: visualize the model predictions and compare them with the ground truth, using FiftyOne.

To install the ultralytics optional dependency, you can run:

```bash
pip install labelr[ultralytics]
```

## Usage

### Label Studio integration

To create a Label Studio project, you need to have a Label Studio instance running. Launching a Label Studio instance is out of the scope of this project, but you can follow the instructions on the [Label Studio documentation](https://labelstud.io/guide/install.html).

By default, the CLI will assume you're running Label Studio locally (url: http://127.0.0.1:8080). You can change the URL by setting the `--label-studio-url` CLI option or by providing the URL in the `LABELR_LABEL_STUDIO_URL` environment variable.

Labelr also support configuring settings globally using a global config file. To set the Label Studio URL, you can run:

```bash
labelr config label_studio_url http://127.0.0.1:8080
```

For all the commands that interact with Label Studio, you need to provide an API key using the `--api-key` CLI option (or through environment variable or the config file). You can get an API key by logging in to the Label Studio instance and going to the Account & Settings page.

#### Create a project

Once you have a Label Studio instance running, you can create a project easily. First, you need to create a configuration file for the project. The configuration file is an XML file that defines the labeling interface and the labels to use for the project. You can find an example of a configuration file in the [Label Studio documentation](https://labelstud.io/guide/setup).

For an object detection task, a command allows you to create the configuration file automatically:

```bash
labelr ls create-config --labels 'label1' --labels 'label2' --output-file label_config.xml
```

where `label1` and `label2` are the labels you want to use for the object detection task, and `label_config.xml` is the output file that will contain the configuration.

Then, you can create a project on Label Studio with the following command:

```bash
labelr ls create --title my_project --api-key API_KEY --config-file label_config.xml
```

where `API_KEY` is the API key of the Label Studio instance (API key is available at Account page), and `label_config.xml` is the configuration file of the project.

`ls` stands for Label Studio in the CLI.

#### Create a dataset file

If you have a list of images, for an object detection task, you can quickly create a dataset file with the following command:

```bash
labelr ls create-dataset-file --input-file image_urls.txt --output-file dataset.json
```

where `image_urls.txt` is a file containing the URLs of the images, one per line, and `dataset.json` is the output file.

#### Import data

Next, import the generated data to a project with the following command:

```bash
labelr ls import-data --project-id PROJECT_ID --dataset-path dataset.json
```

where `PROJECT_ID` is the ID of the project you created.

#### Pre-annotate the data

To accelerate annotation, you can pre-annotate the images with an object detection model. We support three pre-annotation backends:

- `ultralytics`: use your own model or [Yolo-World](https://docs.ultralytics.com/models/yolo-world/), a zero-shot model that can detect any object using a text description of the object. You can specify the path or the name of the model with the `--model-name` option. If no model name is provided, the `yolov8x-worldv2.pt` model (Yolo-World) is used.
- `ultralytics_sam3`: use [SAM3](https://docs.ultralytics.com/models/sam-3/), another zero-shot model. We advice to use this backend, as it's the most accurate. The `--model-name` option is ignored when this backend is used.
- `robotoff`: the ML backend of Open Food Facts (specific to Open Food Facts projects).

When using `ultralytics` or `ultralytics_sam3`, make sure you installed the labelr package with the `ultralytics` extra.

To pre-annotate the data with Ultralytics, use the following command:

```bash
labelr ls add-prediction --project-id PROJECT_ID --backend ultralytics_sam3 --labels 'product' --labels 'price tag' --label-mapping '{"price tag": "price-tag"}'
```

The SAM3 model will be automatically downloaded from Hugging Face. [SAM3](https://huggingface.co/facebook/sam3) is a gated model, it requires a permission before getting access to the model.Make sure you were granted the access before launching the command.

In the command above, `labels` is the list of labels to use for the object detection task (you can add as many labels as you want). You can also provide a `--label-mapping` option in case the names of the label of the model you use for pre-annotation is different from the names configured on your Label Studio project.


#### Export the data

Once the data is annotated, you can export it to a Hugging Face dataset or to local disk (Ultralytics format). To export it to disk, use the following command:

```bash
labelr datasets export --project-id PROJECT_ID --from ls --to ultralytics --output-dir output --label-names 'product,price-tag'
```

where `output` is the directory where the data will be exported. Currently, label names must be provided, as the CLI does not support exporting label names from Label Studio yet.

To export the data to a Hugging Face dataset, use the following command:

```bash
labelr datasets export --project-id PROJECT_ID --from ls --to huggingface --repo-id REPO_ID --label-names 'product,price-tag'
```

where `REPO_ID` is the ID of the Hugging Face repository where the dataset will be uploaded (ex: `openfoodfacts/food-detection`).

### Lauch training jobs

You can also launch training jobs for YOLO object detection models using datasets hosted on Hugging Face. Please refer to the [train-yolo package README](packages/train-yolo/README.md) for more details on how to use this feature.

## Configuration

Some Labelr settings can be configured using a configuration file or through environment variables. The configuration file is located at `~/.config/labelr/config.json`.

By order of precedence, the configuration is loaded from:

- CLI command option
- environment variable
- file configuration 

The following variables are currently supported:

- `label_studio_url`: URL of the Label Studio server. Can also be set with the `LABELR_LABEL_STUDIO_URL` environment variable.
- `label_studio_api_key`: API key for Label Studio. Can also be set with the `LABELR_LABEL_STUDIO_API_KEY` environment variable.
