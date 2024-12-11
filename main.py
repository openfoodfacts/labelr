from typing import Annotated, Optional

import typer
from cli.apps import datasets as dataset_app
from cli.apps import projects as project_app
from cli.apps import users as user_app
from cli.config import LABEL_STUDIO_DEFAULT_URL

from openfoodfacts.utils import get_logger

app = typer.Typer(pretty_exceptions_show_locals=False)

logger = get_logger()


@app.command()
def predict_object(
    model_name: Annotated[
        str, typer.Option(help="Name of the object detection model to run")
    ],
    image_url: Annotated[str, typer.Option(help="URL of the image to process")],
    triton_uri: Annotated[
        str, typer.Option(help="URI (host+port) of the Triton Inference Server")
    ],
    threshold: float = 0.5,
):
    from cli.triton.object_detection import ObjectDetectionModelRegistry

    from openfoodfacts.utils import get_image_from_url

    model = ObjectDetectionModelRegistry.get(model_name)
    image = get_image_from_url(image_url)
    output = model.detect_from_image(image, triton_uri=triton_uri)
    results = output.select(threshold=threshold)

    for result in results:
        typer.echo(result)


# Temporary scripts


@app.command()
def skip_rotated_images(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    updated_by: Annotated[
        Optional[int], typer.Option(help="User ID to declare as annotator")
    ] = None,
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    import requests
    import tqdm
    from label_studio_sdk.client import LabelStudio
    from label_studio_sdk.types.task import Task

    from openfoodfacts.ocr import OCRResult

    session = requests.Session()
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    task: Task
    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, fields="all"), desc="tasks"
    ):
        if any(annotation["was_cancelled"] for annotation in task.annotations):
            continue

        assert task.total_annotations == 1, (
            "Task has multiple annotations (%s)" % task.id
        )
        task_id = task.id

        annotation = task.annotations[0]
        annotation_id = annotation["id"]

        ocr_url = task.data["image_url"].replace(".jpg", ".json")
        ocr_result = OCRResult.from_url(ocr_url, session=session, error_raise=False)

        if ocr_result is None:
            logger.warning("No OCR result for task: %s", task_id)
            continue

        orientation_result = ocr_result.get_orientation()

        if orientation_result is None:
            # logger.info("No orientation for task: %s", task_id)
            continue

        orientation = orientation_result.orientation.name
        if orientation != "up":
            logger.info(
                "Skipping rotated image for task: %s (orientation: %s)",
                task_id,
                orientation,
            )
            ls.annotations.update(
                id=annotation_id,
                was_cancelled=True,
                updated_by=updated_by,
            )
        elif orientation == "up":
            logger.debug("Keeping annotation for task: %s", task_id)


@app.command()
def fix_label(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    import tqdm
    from label_studio_sdk.client import LabelStudio
    from label_studio_sdk.types.task import Task

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    task: Task
    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, fields="all"), desc="tasks"
    ):
        for prediction in task.predictions:
            updated = False
            if "result" in prediction:
                for result in prediction["result"]:
                    value = result["value"]
                    if "rectanglelabels" in value and value["rectanglelabels"] != [
                        "price-tag"
                    ]:
                        value["rectanglelabels"] = ["price-tag"]
                        updated = True

            if updated:
                print(f"Updating prediction {prediction['id']}, task {task.id}")
                ls.predictions.update(prediction["id"], result=prediction["result"])

        for annotation in task.annotations:
            updated = False
            if "result" in annotation:
                for result in annotation["result"]:
                    value = result["value"]
                    if "rectanglelabels" in value and value["rectanglelabels"] != [
                        "price-tag"
                    ]:
                        value["rectanglelabels"] = ["price-tag"]
                        updated = True

            if updated:
                print(f"Updating annotation {annotation['id']}, task {task.id}")
                ls.annotations.update(annotation["id"], result=annotation["result"])


@app.command()
def select_price_tag_images(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    import typing
    from pathlib import Path
    from typing import Any
    from urllib.parse import urlparse

    import requests
    import tqdm
    from label_studio_sdk.client import LabelStudio
    from label_studio_sdk.types.task import Task

    session = requests.Session()
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    proof_paths = (Path(__file__).parent / "proof.txt").read_text().splitlines()
    task: Task
    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, include="data,id"), desc="tasks"
    ):
        data = typing.cast(dict[str, Any], task.data)

        if "is_raw_product_shelf" in data:
            continue
        image_url = data["image_url"]
        file_path = urlparse(image_url).path.replace("/img/", "")
        r = session.get(
            f"https://robotoff.openfoodfacts.org/api/v1/images/predict?image_url={image_url}&models=price_proof_classification",
        )

        if r.status_code != 200:
            print(
                f"Failed to get prediction for {image_url}, error: {r.text} (status: {r.status_code})"
            )
            continue

        prediction = r.json()["predictions"]["price_proof_classification"][0]["label"]

        is_raw_preduct_shelf = False
        if prediction in ("PRICE_TAG", "SHELF"):
            is_raw_preduct_shelf = file_path in proof_paths

        ls.tasks.update(
            task.id,
            data={
                **data,
                "is_raw_product_shelf": "true" if is_raw_preduct_shelf else "false",
            },
        )


@app.command()
def add_predicted_category(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    import typing
    from typing import Any

    import requests
    import tqdm
    from label_studio_sdk.client import LabelStudio
    from label_studio_sdk.types.task import Task

    session = requests.Session()
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    task: Task
    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, include="data,id"), desc="tasks"
    ):
        data = typing.cast(dict[str, Any], task.data)

        if "predicted_category" in data:
            continue
        image_url = data["image_url"]
        r = session.get(
            f"https://robotoff.openfoodfacts.org/api/v1/images/predict?image_url={image_url}&models=price_proof_classification",
        )

        if r.status_code != 200:
            print(
                f"Failed to get prediction for {image_url}, error: {r.text} (status: {r.status_code})"
            )
            continue

        predicted_category = r.json()["predictions"]["price_proof_classification"][0][
            "label"
        ]

        ls.tasks.update(
            task.id,
            data={
                **data,
                "predicted_category": predicted_category,
            },
        )


app.add_typer(user_app.app, name="users", help="Manage Label Studio users")
app.add_typer(
    project_app.app,
    name="projects",
    help="Manage Label Studio projects (create, import data, etc.)",
)
app.add_typer(
    dataset_app.app,
    name="datasets",
    help="Manage datasets (convert, export, check, etc.)",
)

if __name__ == "__main__":
    app()
