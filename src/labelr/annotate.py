import random
import string

from openfoodfacts.utils import get_logger

from ultralytics.engine.results import Results

logger = get_logger(__name__)


def format_annotation_results_from_ultralytics(
    results: Results,
    labels: list[str],
    label_mapping: dict[str, str] | None = None,
) -> list[dict]:
    annotation_results = []
    orig_height, orig_width = results.orig_shape
    boxes = results.boxes
    classes = boxes.cls.tolist()
    scores = boxes.conf.tolist()
    for i, xyxyn in enumerate(boxes.xyxyn):
        # Boxes found.
        if len(xyxyn) > 0:
            xyxyn = xyxyn.tolist()
            x1 = xyxyn[0] * 100
            y1 = xyxyn[1] * 100
            x2 = xyxyn[2] * 100
            y2 = xyxyn[3] * 100
            width = x2 - x1
            height = y2 - y1
            label_id = int(classes[i])
            label_name = labels[label_id]
            # Undocumented Label Studio feature, but we can add the individual bounding box score:
            # https://github.com/HumanSignal/label-studio/issues/2944#issuecomment-1249943187
            # This allows to:
            # - display each individual bounding score
            # - find tasks for which the model is not-confident using Labelr (to be implemented)
            score = scores[i]
            if label_mapping:
                label_name = label_mapping.get(label_name, label_name)
            annotation_results.append(
                {
                    "id": generate_id(),
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": orig_width,
                    "original_height": orig_height,
                    "image_rotation": 0,
                    "score": score,
                    "value": {
                        "rotation": 0,
                        "x": x1,
                        "y": y1,
                        "width": width,
                        "height": height,
                        "rectanglelabels": [label_name],
                    },
                },
            )
    return annotation_results


def generate_id(length: int = 10) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))
