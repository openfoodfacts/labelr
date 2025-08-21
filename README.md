# Labelr

Labelr a command line interface that aims to provide a set of tools to help data scientists and machine learning engineers to deal with ML data annotation, data preprocessing and format conversion.

This project started as a way to automate some of the tasks we do at Open Food Facts to manage data at different stages of the machine learning pipeline.

The CLI currently is integrated with Label Studio (for data annotation), Ultralytics (for object detection) and Hugging Face (for model and dataset storage). It only works with some specific tasks (object detection only currently), but it's meant to be extended to other tasks in the future.

It currently allows to:

- create Label Studio projects
- upload images to Label Studio
- pre-annotate the tasks either with an existing object detection model run by Triton, or with Yolo-World (through Ultralytics)
- perform data quality checks on Label Studio
- export the data to Hugging Face Dataset or to local disk

## Installation

Python 3.10 or higher is required to run this CLI.

To install the CLI, simply run:

```bash
pip install labelr
```
We recommend to install the CLI in a virtual environment. You can either use pip or conda for that.

There are two optional dependencies that you can install to use the CLI:
- `ultralytics`: pre-annotate object detection datasets with an ultralytics model (yolo, yolo-world)
- `triton`: pre-annotate object detection datasets using a model served by a Triton inference server

To install the optional dependencies, you can run:

```bash
pip install labelr[ultralytics,triton]
```

## Usage

### Label Studio integration

To create a Label Studio project, you need to have a Label Studio instance running. Launching a Label Studio instance is out of the scope of this project, but you can follow the instructions on the [Label Studio documentation](https://labelstud.io/guide/install.html).

By default, the CLI will use Open Food Facts Label Studio instance, but you can change the URL by setting the `--label-studio-url` CLI option.

For all the commands that interact with Label Studio, you need to provide an API key using the `--api-key` CLI option. You can get an API key by logging in to the Label Studio instance and going to the Account & Settings page.

#### Create a project

Once you have a Label Studio instance running, you can create a project easily. First, you need to create a configuration file for the project. The configuration file is an XML file that defines the labeling interface and the labels to use for the project. You can find an example of a configuration file in the [Label Studio documentation](https://labelstud.io/guide/setup).

For an object detection task, a command allows you to create the configuration file automatically:

```bash
labelr projects create-config --labels 'label1' --labels 'label2' --output-file label_config.xml
```

where `label1` and `label2` are the labels you want to use for the object detection task, and `label_config.xml` is the output file that will contain the configuration.

Then, you can create a project on Label Studio with the following command:

```bash
labelr projects create --title my_project --api-key API_KEY --config-file label_config.xml
```

where `API_KEY` is the API key of the Label Studio instance (API key is available at Account page), and `label_config.xml` is the configuration file of the project.

#### Create a dataset file

If you have a list of images, for an object detection task, you can quickly create a dataset file with the following command:

```bash
labelr projects create-dataset-file --input-file image_urls.txt --output-file dataset.json
```

where `image_urls.txt` is a file containing the URLs of the images, one per line, and `dataset.json` is the output file.

#### Import data

Next, import the generated data to a project with the following command:

```bash
labelr projects import-data --project-id PROJECT_ID --dataset-path dataset.json
```

where `PROJECT_ID` is the ID of the project you created.

#### Pre-annotate the data

To accelerate annotation, you can pre-annotate the images with an object detection model. We support two pre-annotation backends:

- Triton: you need to have a Triton server running with a model that supports object detection. The object detection model is expected to be a yolo-v8 model. You can set the URL of the Triton server with the `--triton-url` CLI option.

- Ultralytics: you can use the [Yolo-World model from Ultralytics](https://github.com/ultralytics/ultralytics), Ultralytics should be installed in the same virtualenv.

To pre-annotate the data with Triton, use the following command:

```bash
labelr projects add-prediction --project-id PROJECT_ID --backend ultralytics --labels 'product' --labels 'price tag' --label-mapping '{"price tag": "price-tag"}'
```

where `labels` is the list of labels to use for the object detection task (you can add as many labels as you want).
For Ultralytics, you can also provide a `--label-mapping` option to map the labels from the model to the labels of the project.

By default, for Ultralytics, the `yolov8x-worldv2.pt` model is used. You can change the model by setting the `--model-name` CLI option.

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