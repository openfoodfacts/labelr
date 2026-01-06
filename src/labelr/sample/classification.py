import datasets

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
