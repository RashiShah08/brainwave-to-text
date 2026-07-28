"""Dataset registry.

Everything above this layer -- pipelines, evaluation, calibration, serving --
works on :class:`~bwt.data.epochs.EpochBundle` and never needs to know which
corpus produced it. Adding a dataset means implementing one class here, not
touching the models.

That indirection is what lets the project claim its methods generalise. Results
from a single recording rig are always suspect: montage, amplifier, reference
scheme and cue timing all differ between labs, and a pipeline tuned to one can
fail completely on another. Two independent datasets is the minimum honest
demonstration.

Registered datasets
-------------------
``eegmmidb``
    PhysioNet EEG Motor Movement/Imagery. 109 subjects (105 usable), 64
    channels, 160 Hz. Local EDF files.
``bnci2a``
    BCI Competition IV dataset 2a. 9 subjects, 22 channels, 250 Hz, 4 classes,
    two sessions recorded on different days. Fetched via MOABB.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np

from bwt.data.epochs import EpochBundle, concat_bundles
from bwt.logging_utils import get_logger

log = get_logger(__name__)


class Dataset(ABC):
    """A source of labelled motor-imagery trials."""

    #: Short registry key.
    name: str
    #: Human-readable description, surfaced by `bwt datasets`.
    description: str
    #: Native sampling rate in Hz.
    sfreq: float
    #: Task name -> ordered class names.
    tasks: dict[str, tuple[str, ...]]
    #: Default epoch window relative to cue onset, in seconds.
    default_window: tuple[float, float]

    @abstractmethod
    def subjects(self) -> list[int]:
        """Usable subject identifiers, ascending."""

    @abstractmethod
    def load_subject(
        self, subject: int, task: str, tmin: float, tmax: float
    ) -> EpochBundle | None:
        """Load one subject's trials, or ``None`` if they contribute none."""

    def task_classes(self, task: str) -> tuple[str, ...]:
        try:
            return self.tasks[task]
        except KeyError:
            raise ValueError(
                f"dataset {self.name!r} has no task {task!r}; "
                f"available: {', '.join(sorted(self.tasks))}"
            ) from None

    def load(
        self,
        task: str,
        *,
        subjects: Iterable[int] | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        n_jobs: int = 1,
    ) -> EpochBundle:
        """Load every requested subject and concatenate."""
        self.task_classes(task)
        tmin = self.default_window[0] if tmin is None else tmin
        tmax = self.default_window[1] if tmax is None else tmax

        chosen = sorted(int(s) for s in (subjects if subjects is not None
                                         else self.subjects()))
        if not chosen:
            raise RuntimeError(f"dataset {self.name!r} has no available subjects")

        log.info("loading %s task=%s subjects=%d window=%.2f-%.2fs",
                 self.name, task, len(chosen), tmin, tmax)

        def _one(subject: int):
            try:
                return self.load_subject(subject, task, tmin, tmax)
            except Exception as exc:
                log.warning("%s subject %s failed: %s", self.name, subject, exc)
                return None

        if n_jobs == 1:
            results = []
            for index, subject in enumerate(chosen, start=1):
                results.append(_one(subject))
                if index % 10 == 0 or index == len(chosen):
                    log.info("  %d/%d subjects", index, len(chosen))
        else:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs)(delayed(_one)(s) for s in chosen)

        usable = [r for r in results if r is not None and r.n_trials > 0]
        if not usable:
            raise RuntimeError(f"no usable trials for {self.name}/{task}")
        return concat_bundles(usable)


# --------------------------------------------------------------------------- #
# PhysioNet EEGMMIDB
# --------------------------------------------------------------------------- #


class EEGMMIDB(Dataset):
    """PhysioNet EEG Motor Movement/Imagery, read from local EDF files."""

    name = "eegmmidb"
    description = (
        "PhysioNet EEG Motor Movement/Imagery Database: 105 usable subjects, "
        "64 channels, 160 Hz"
    )
    sfreq = 160.0
    default_window = (0.5, 3.5)

    def __init__(self, root: Path | None = None):
        from bwt.data.physionet import TASKS

        self.root = root
        self.tasks = {name: spec.classes for name, spec in TASKS.items()}

    def subjects(self) -> list[int]:
        from bwt.data.physionet import available_subjects

        return available_subjects(self.root)

    def load_subject(self, subject, task, tmin, tmax):
        from bwt.data.epochs import load_subject_epochs

        return load_subject_epochs(
            subject, task, tmin=tmin, tmax=tmax, root=self.root
        )


# --------------------------------------------------------------------------- #
# BCI Competition IV dataset 2a
# --------------------------------------------------------------------------- #


class BNCI2a(Dataset):
    """BCI Competition IV dataset 2a, fetched through MOABB.

    A useful contrast with EEGMMIDB in every dimension that matters: a different
    laboratory and amplifier, 22 electrodes instead of 64, 250 Hz instead of
    160, four classes including tongue imagery, and two sessions per subject
    recorded on different days. Trials are also longer (4 s of cued imagery
    after a 2 s fixation), and the dataset is far smaller -- 9 subjects rather
    than 105 -- which makes it the harder test of whether a method transfers.

    Session identity is preserved in :attr:`EpochBundle.runs` so that a
    session-wise evaluation (train on day one, test on day two) is possible.
    """

    name = "bnci2a"
    description = (
        "BCI Competition IV-2a: 9 subjects, 22 channels, 250 Hz, 4 classes, "
        "2 sessions"
    )
    sfreq = 250.0
    # MOABB reports trials on the interval [2, 6] s where 2 s is cue onset, and
    # its loader already zeroes time at the cue. Skip the first half-second for
    # the same reason as EEGMMIDB: it is the visual response to the cue.
    default_window = (0.5, 3.5)

    tasks: ClassVar[dict[str, tuple[str, ...]]] = {
        "mi_left_right": ("left_hand", "right_hand"),
        "mi_four_class": ("left_hand", "right_hand", "feet", "tongue"),
    }

    _MOABB_EVENTS = ("left_hand", "right_hand", "feet", "tongue")

    def __init__(self):
        self._dataset = None

    def _moabb(self):
        if self._dataset is None:
            from moabb.datasets import BNCI2014_001

            self._dataset = BNCI2014_001()
        return self._dataset

    def subjects(self) -> list[int]:
        return list(self._moabb().subject_list)

    def load_subject(self, subject, task, tmin, tmax):
        import mne

        classes = self.task_classes(task)
        wanted = {name: index for index, name in enumerate(classes)}

        data = self._moabb().get_data(subjects=[subject])[subject]

        per_run_X, per_run_y, per_run_r = [], [], []
        ch_names: tuple[str, ...] | None = None

        for session_index, (_, runs) in enumerate(sorted(data.items())):
            for _, raw in sorted(runs.items()):
                raw = raw.copy().pick("eeg")
                sfreq = float(raw.info["sfreq"])
                if abs(sfreq - self.sfreq) > 1e-6:
                    raise ValueError(
                        f"expected {self.sfreq} Hz, got {sfreq} Hz"
                    )

                events, event_id = mne.events_from_annotations(
                    raw, verbose="ERROR"
                )
                keep = {
                    name: code for name, code in event_id.items()
                    if name in wanted
                }
                if not keep:
                    continue

                epochs = mne.Epochs(
                    raw, events, event_id=keep, tmin=tmin, tmax=tmax,
                    baseline=None, preload=True, picks="eeg", reject=None,
                    on_missing="ignore", event_repeated="drop", verbose="ERROR",
                )
                if len(epochs) == 0:
                    continue

                code_to_label = {
                    code: wanted[name] for name, code in keep.items()
                }
                labels = np.array(
                    [code_to_label[c] for c in epochs.events[:, -1]],
                    dtype=np.int64,
                )

                per_run_X.append(
                    (epochs.get_data(copy=True) * 1e6).astype(np.float32)
                )
                per_run_y.append(labels)
                per_run_r.append(
                    np.full(len(epochs), session_index, dtype=np.int64)
                )

                names = tuple(epochs.ch_names)
                if ch_names is None:
                    ch_names = names
                elif names != ch_names:
                    raise ValueError(
                        f"subject {subject}: channel names differ between runs"
                    )

        if not per_run_X:
            return None

        n_times = min(a.shape[2] for a in per_run_X)
        X = np.concatenate([a[:, :, :n_times] for a in per_run_X], axis=0)

        return EpochBundle(
            X=X,
            y=np.concatenate(per_run_y),
            groups=np.full(len(X), subject, dtype=np.int64),
            runs=np.concatenate(per_run_r),
            classes=classes,
            ch_names=ch_names,  # type: ignore[arg-type]
            sfreq=self.sfreq,
            tmin=tmin,
            tmax=tmax,
            task=task,
        )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, type[Dataset]] = {
    EEGMMIDB.name: EEGMMIDB,
    BNCI2a.name: BNCI2a,
}

DEFAULT_DATASET = EEGMMIDB.name


def get_dataset(name: str = DEFAULT_DATASET, **kwargs) -> Dataset:
    try:
        return _REGISTRY[name](**kwargs)
    except KeyError:
        raise ValueError(
            f"unknown dataset {name!r}; available: "
            f"{', '.join(sorted(_REGISTRY))}"
        ) from None


def list_datasets() -> list[tuple[str, str]]:
    return sorted((cls.name, cls.description) for cls in _REGISTRY.values())


def available_tasks(dataset: str = DEFAULT_DATASET) -> dict[str, tuple[str, ...]]:
    return dict(get_dataset(dataset).tasks)


# --------------------------------------------------------------------------- #
# Cached loading
# --------------------------------------------------------------------------- #


def _cache_key(dataset: str, task: str, subjects: Sequence[int],
               tmin: float, tmax: float) -> str:
    import hashlib
    import json

    from bwt.data.epochs import CACHE_VERSION

    payload = json.dumps(
        {
            "v": CACHE_VERSION,
            "dataset": dataset,
            "task": task,
            "subjects": list(subjects),
            "tmin": tmin,
            "tmax": tmax,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{dataset}_{task}_{len(subjects)}subj_{digest}.npz"


def load_bundle(
    task: str,
    *,
    dataset: str = DEFAULT_DATASET,
    subjects: Iterable[int] | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    root: Path | None = None,
    n_jobs: int = 1,
    use_cache: bool = True,
) -> EpochBundle:
    """Load (and cache) the epoch tensor for one dataset/task combination.

    The cache key covers the dataset, task, subject list, and epoch window, so
    changing any of them yields a different file rather than a stale hit.
    """
    from bwt.paths import cache_dir

    kwargs = {"root": root} if (root is not None and dataset == "eegmmidb") else {}
    source = get_dataset(dataset, **kwargs)

    tmin = source.default_window[0] if tmin is None else tmin
    tmax = source.default_window[1] if tmax is None else tmax

    chosen = sorted(int(s) for s in (subjects if subjects is not None
                                     else source.subjects()))
    if not chosen:
        raise RuntimeError(
            f"no subjects available for dataset {dataset!r}. For eegmmidb, "
            "expected directories S001..S109 under raw_data/ (override with "
            "$BWT_RAW_DATA); for bnci2a, MOABB downloads on first use."
        )

    cache_path = cache_dir() / _cache_key(dataset, task, chosen, tmin, tmax)
    if use_cache and cache_path.is_file():
        try:
            bundle = EpochBundle.load(cache_path)
            log.info("loaded cached epochs: %s", bundle.summary())
            return bundle
        except Exception as exc:
            log.warning("ignoring unreadable cache %s (%s)", cache_path.name, exc)

    bundle = source.load(task, subjects=chosen, tmin=tmin, tmax=tmax,
                         n_jobs=n_jobs)
    log.info("epoched %s", bundle.summary())

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        bundle.save(cache_path)
        log.info("cached to %s", cache_path.name)
    return bundle


__all__ = [
    "DEFAULT_DATASET",
    "EEGMMIDB",
    "BNCI2a",
    "Dataset",
    "available_tasks",
    "get_dataset",
    "list_datasets",
    "load_bundle",
]
