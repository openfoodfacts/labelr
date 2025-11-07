import datasets
from datasets import Features
from datasets import Image as HFImage

# dataset features for predictions.parquet used in evaluation and visualization
OBJECT_DETECTION_DS_PREDICTION_FEATURES = Features(
    {
        "image": HFImage(),
        "image_with_prediction": HFImage(),
        "image_id": datasets.Value("string"),
        "detected": {
            "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
            "category_id": datasets.Sequence(datasets.Value("int64")),
            "category_name": datasets.Sequence(datasets.Value("string")),
            "confidence": datasets.Sequence(datasets.Value("float32")),
        },
        "split": datasets.Value("string"),
        "width": datasets.Value("int64"),
        "height": datasets.Value("int64"),
        "meta": {
            "barcode": datasets.Value("string"),
            "off_image_id": datasets.Value("string"),
            "image_url": datasets.Value("string"),
        },
        "objects": {
            "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
            "category_id": datasets.Sequence(datasets.Value("int64")),
            "category_name": datasets.Sequence(datasets.Value("string")),
        },
    }
)
