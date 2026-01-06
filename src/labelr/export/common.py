import pickle
from pathlib import Path

from openfoodfacts.types import Flavor

from labelr.export.classification import export_from_ultralytics_to_hf_classification
from labelr.types import TaskType


def _pickle_sample_generator(dir: Path):
    """Generator that yields samples from pickles in a directory."""
    for pkl in dir.glob("*.pkl"):
        with open(pkl, "rb") as f:
            yield pickle.load(f)


def export_from_ultralytics_to_hf(
    task_type: TaskType,
    dataset_dir: Path,
    repo_id: str,
    label_names: list[str],
    merge_labels: bool = False,
    is_openfoodfacts_dataset: bool = False,
    openfoodfacts_flavor: Flavor = Flavor.off,
) -> None:
    if task_type != TaskType.classification:
        raise NotImplementedError(
            "Only classification task is currently supported for Ultralytics to HF export"
        )

    if task_type == TaskType.classification:
        export_from_ultralytics_to_hf_classification(
            dataset_dir=dataset_dir,
            repo_id=repo_id,
            label_names=label_names,
            merge_labels=merge_labels,
            is_openfoodfacts_dataset=is_openfoodfacts_dataset,
            openfoodfacts_flavor=openfoodfacts_flavor,
        )
