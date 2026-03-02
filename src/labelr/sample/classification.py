import random
import string

import datasets


def format_annotation_results(label: str):
    """Format annotation results for image classification tasks in Label Studio."""
    id_ = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    annotation_result = {
        "id": id_,
        "type": "choices",
        "from_name": "choice",
        "to_name": "image",
        "value": {
            "choices": [label],
        },
    }
    return [annotation_result]


HF_DS_CLASSIFICATION_FEATURES = datasets.Features(
    {
        "image_id": datasets.Value("string"),
        "image": datasets.features.Image(),
        "width": datasets.Value("int64"),
        "height": datasets.Value("int64"),
        "meta": {
            "barcode": datasets.Value("string"),
            "off_image_id": datasets.Value("string"),
            "image_url": datasets.Value("string"),
        },
        "category_id": datasets.Value("int64"),
        "category_name": datasets.Value("string"),
    }
)
