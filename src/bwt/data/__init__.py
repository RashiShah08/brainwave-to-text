"""Dataset discovery, protocol definitions, and epoching."""

from bwt.data.datasets import (
    DEFAULT_DATASET,
    EEGMMIDB,
    BNCI2a,
    Dataset,
    available_tasks,
    get_dataset,
    list_datasets,
    load_bundle,
)
from bwt.data.epochs import EpochBundle, concat_bundles, load_subject_epochs
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

__all__ = [
    "DEFAULT_DATASET",
    "EEGMMIDB",
    "EXCLUDED_SUBJECTS",
    "TASKS",
    "BNCI2a",
    "Dataset",
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
