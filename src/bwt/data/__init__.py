"""Dataset discovery, protocol definitions, and epoching."""

from bwt.data.physionet import (
    EXCLUDED_SUBJECTS,
    TASKS,
    Execution,
    Movement,
    RunSpec,
    TaskSpec,
    available_subjects,
    edf_path,
    get_task,
    run_spec,
)
from bwt.data.epochs import EpochBundle, concat_bundles, load_subject_epochs
from bwt.data.datasets import (
    DEFAULT_DATASET,
    BNCI2a,
    Dataset,
    EEGMMIDB,
    available_tasks,
    get_dataset,
    list_datasets,
    load_bundle,
)

__all__ = [
    "DEFAULT_DATASET",
    "EXCLUDED_SUBJECTS",
    "TASKS",
    "BNCI2a",
    "Dataset",
    "EEGMMIDB",
    "EpochBundle",
    "Execution",
    "Movement",
    "RunSpec",
    "TaskSpec",
    "available_subjects",
    "available_tasks",
    "concat_bundles",
    "edf_path",
    "get_dataset",
    "get_task",
    "list_datasets",
    "load_bundle",
    "load_subject_epochs",
    "run_spec",
]
