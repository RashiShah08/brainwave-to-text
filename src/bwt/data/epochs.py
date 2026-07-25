"""Turn EDF recordings into a labelled, subject-tagged epoch tensor.

Design rules enforced here:

* **No resampling.** The data is 160 Hz; the previous pipeline upsampled to
  256 Hz, which adds no information, costs time, and invited a 256-sample
  constant to leak through the rest of the codebase.
* **No filtering, no artifact correction.** Every signal-processing step lives
  inside the fitted scikit-learn pipeline instead, so that training and serving
  are guaranteed to apply the identical transform. This module's only job is to
  cut correctly-labelled windows out of the raw recording.
* **Subject identity is a first-class column.** ``EpochBundle.groups`` follows
  the data everywhere so that a subject can never straddle a train/test split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from bwt.data.physionet import (
    EXPECTED_N_CHANNELS,
    EXPECTED_SFREQ,
    TaskSpec,
    available_subjects,
    edf_path,
    get_task,
    subject_id,
)
from bwt.logging_utils import get_logger
from bwt.paths import cache_dir, raw_data_dir

log = get_logger(__name__)

#: Bump when the on-disk cache format or the epoching semantics change.
CACHE_VERSION = 2

#: Default analysis window relative to cue onset, in seconds. Trials are 4.1 s
#: long. Event-related desynchronisation of the mu/beta rhythms builds up over
#: roughly 0.5-3.5 s after the cue, so the first half-second (dominated by the
#: visual evoked response to the cue itself) is skipped.
DEFAULT_TMIN = 0.5
DEFAULT_TMAX = 3.5


@dataclass
class EpochBundle:
    """A labelled epoch tensor plus everything needed to interpret it.

    Attributes
    ----------
    X
        ``(n_trials, n_channels, n_times)`` float32, in **microvolts**.
    y
        ``(n_trials,)`` int label; index into :attr:`classes`.
    groups
        ``(n_trials,)`` int subject number. Used as the grouping variable for
        every cross-validation split in the project.
    runs
        ``(n_trials,)`` int run number the trial came from.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    runs: np.ndarray
    classes: tuple[str, ...]
    ch_names: tuple[str, ...]
    sfreq: float
    tmin: float
    tmax: float
    task: str
    units: str = "uV"

    # -- introspection ---------------------------------------------------- #

    def __post_init__(self) -> None:
        n = len(self.X)
        for name in ("y", "groups", "runs"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(
                    f"EpochBundle.{name} has {len(arr)} rows but X has {n}"
                )
        if self.X.ndim != 3:
            raise ValueError(f"X must be 3-D (trials, channels, times), got {self.X.shape}")
        if self.X.shape[1] != len(self.ch_names):
            raise ValueError(
                f"X has {self.X.shape[1]} channels but {len(self.ch_names)} names"
            )

    @property
    def n_trials(self) -> int:
        return len(self.X)

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def n_times(self) -> int:
        return self.X.shape[2]

    @property
    def subjects(self) -> list[int]:
        return sorted(set(int(g) for g in self.groups))

    def class_counts(self) -> dict[str, int]:
        return {
            name: int((self.y == i).sum()) for i, name in enumerate(self.classes)
        }

    def subset(self, mask: np.ndarray) -> "EpochBundle":
        """Return a new bundle restricted to ``mask`` (boolean or index array)."""
        return EpochBundle(
            X=self.X[mask],
            y=self.y[mask],
            groups=self.groups[mask],
            runs=self.runs[mask],
            classes=self.classes,
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            tmin=self.tmin,
            tmax=self.tmax,
            task=self.task,
            units=self.units,
        )

    def for_subject(self, subject: int) -> "EpochBundle":
        return self.subset(self.groups == subject)

    def metadata(self) -> dict:
        """JSON-serialisable description, embedded into model cards."""
        return {
            "task": self.task,
            "classes": list(self.classes),
            "class_counts": self.class_counts(),
            "n_trials": self.n_trials,
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "n_subjects": len(self.subjects),
            "subjects": self.subjects,
            "sfreq": self.sfreq,
            "tmin": self.tmin,
            "tmax": self.tmax,
            "units": self.units,
            "ch_names": list(self.ch_names),
        }

    def summary(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in self.class_counts().items())
        return (
            f"{self.task}: {self.n_trials} trials from {len(self.subjects)} subjects "
            f"| {self.n_channels}ch x {self.n_times} samples @ {self.sfreq:g} Hz "
            f"({self.tmin:g}-{self.tmax:g}s) | {counts}"
        )

    # -- persistence ------------------------------------------------------ #

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {k: v for k, v in asdict(self).items()
                if k not in {"X", "y", "groups", "runs"}}
        meta["classes"] = list(self.classes)
        meta["ch_names"] = list(self.ch_names)
        meta["cache_version"] = CACHE_VERSION
        # np.savez appends ".npz" unless the name already ends with it, so the
        # temp name must keep that suffix or the rename below targets a file
        # that was never written.
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(
            tmp,
            X=self.X,
            y=self.y,
            groups=self.groups,
            runs=self.runs,
            meta=np.array(json.dumps(meta)),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "EpochBundle":
        with np.load(path, allow_pickle=False) as handle:
            meta = json.loads(str(handle["meta"]))
            if meta.get("cache_version") != CACHE_VERSION:
                raise ValueError(
                    f"cache at {path} was written by format version "
                    f"{meta.get('cache_version')}, this build expects {CACHE_VERSION}"
                )
            return cls(
                X=handle["X"],
                y=handle["y"],
                groups=handle["groups"],
                runs=handle["runs"],
                classes=tuple(meta["classes"]),
                ch_names=tuple(meta["ch_names"]),
                sfreq=float(meta["sfreq"]),
                tmin=float(meta["tmin"]),
                tmax=float(meta["tmax"]),
                task=meta["task"],
                units=meta.get("units", "uV"),
            )


def concat_bundles(bundles: Sequence[EpochBundle]) -> EpochBundle:
    """Concatenate per-subject bundles, verifying they are actually compatible."""
    bundles = [b for b in bundles if b is not None and b.n_trials > 0]
    if not bundles:
        raise ValueError("no non-empty bundles to concatenate")

    head = bundles[0]
    for other in bundles[1:]:
        if other.ch_names != head.ch_names:
            raise ValueError(
                "channel names differ between subjects; refusing to concatenate "
                "(this would silently scramble the spatial filters)"
            )
        if other.classes != head.classes:
            raise ValueError("class sets differ between subjects")
        if other.sfreq != head.sfreq:
            raise ValueError(f"sfreq differs: {other.sfreq} vs {head.sfreq}")
        if other.n_times != head.n_times:
            raise ValueError(f"epoch length differs: {other.n_times} vs {head.n_times}")

    return EpochBundle(
        X=np.concatenate([b.X for b in bundles], axis=0),
        y=np.concatenate([b.y for b in bundles], axis=0),
        groups=np.concatenate([b.groups for b in bundles], axis=0),
        runs=np.concatenate([b.runs for b in bundles], axis=0),
        classes=head.classes,
        ch_names=head.ch_names,
        sfreq=head.sfreq,
        tmin=head.tmin,
        tmax=head.tmax,
        task=head.task,
        units=head.units,
    )


# --------------------------------------------------------------------------- #
# Raw reading
# --------------------------------------------------------------------------- #


def read_standardised_raw(path: Path):
    """Read one EDF and normalise its channel naming to the 10-05 system.

    The EEGMMIDB files label channels ``Fc5.``, ``Cz..``, ``Iz..`` and so on.
    ``mne.datasets.eegbci.standardize`` rewrites those to canonical names
    (``FC5``, ``Cz``, ``Iz``), which is what lets us attach a real montage and
    reason about electrode positions.
    """
    import mne
    from mne.datasets import eegbci

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"),
                    on_missing="warn", verbose="ERROR")
    return raw


def _validate_raw(raw, path: Path) -> None:
    sfreq = float(raw.info["sfreq"])
    if sfreq != EXPECTED_SFREQ:
        raise ValueError(
            f"{path.name}: sampling rate is {sfreq} Hz, expected {EXPECTED_SFREQ} Hz. "
            "This recording does not follow the protocol and must not be pooled "
            "with the rest of the database."
        )
    if len(raw.ch_names) != EXPECTED_N_CHANNELS:
        raise ValueError(
            f"{path.name}: {len(raw.ch_names)} channels, expected {EXPECTED_N_CHANNELS}"
        )


def epochs_from_raw(
    raw,
    task: TaskSpec,
    run: int,
    tmin: float = DEFAULT_TMIN,
    tmax: float = DEFAULT_TMAX,
):
    """Cut labelled epochs out of one raw recording.

    Events are built by hand from the annotations rather than via
    ``mne.events_from_annotations`` so that the run-dependent meaning of
    ``T1``/``T2`` is applied explicitly. See :mod:`bwt.data.physionet`.
    """
    import mne

    sfreq = float(raw.info["sfreq"])
    rows: list[tuple[int, int, int]] = []
    for onset, description in zip(raw.annotations.onset, raw.annotations.description):
        label = task.label_of(run, str(description))
        if label is None:
            continue
        rows.append((int(round(float(onset) * sfreq)), 0, label))

    if not rows:
        return None

    events = np.array(sorted(rows), dtype=int)
    present = sorted(set(events[:, 2]))
    event_id = {task.classes[i]: i for i in present}

    return mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,          # no baseline correction: CSP works on covariance
        preload=True,
        picks="eeg",
        reject=None,            # rejection is an explicit, opt-in step
        on_missing="ignore",
        event_repeated="drop",
        verbose="ERROR",
    )


def load_subject_epochs(
    subject: int,
    task: TaskSpec | str,
    *,
    tmin: float = DEFAULT_TMIN,
    tmax: float = DEFAULT_TMAX,
    root: Path | None = None,
    strict: bool = False,
) -> EpochBundle | None:
    """Load every trial for one subject under one task.

    Returns ``None`` when the subject contributes no usable trials. With
    ``strict=True`` a malformed run raises instead of being skipped.
    """
    if isinstance(task, str):
        task = get_task(task)
    root = root or raw_data_dir()

    per_run_X: list[np.ndarray] = []
    per_run_y: list[np.ndarray] = []
    per_run_r: list[np.ndarray] = []
    ch_names: tuple[str, ...] | None = None
    sfreq: float | None = None

    for run in task.runs:
        path = edf_path(subject, run, root)
        if not path.is_file():
            log.warning("missing file, skipping: %s", path)
            if strict:
                raise FileNotFoundError(path)
            continue
        try:
            raw = read_standardised_raw(path)
            _validate_raw(raw, path)
            epochs = epochs_from_raw(raw, task, run, tmin=tmin, tmax=tmax)
        except Exception as exc:  # noqa: BLE001 - one bad run must not kill a run of 105
            log.warning("%s: %s", path.name, exc)
            if strict:
                raise
            continue

        if epochs is None or len(epochs) == 0:
            log.warning("%s: no trials matched task %s", path.name, task.name)
            continue

        data = epochs.get_data(copy=True) * 1e6  # volts -> microvolts
        per_run_X.append(data.astype(np.float32, copy=False))
        per_run_y.append(epochs.events[:, -1].astype(np.int64))
        per_run_r.append(np.full(len(epochs), run, dtype=np.int64))

        run_channels = tuple(epochs.ch_names)
        if ch_names is None:
            ch_names, sfreq = run_channels, float(epochs.info["sfreq"])
        elif run_channels != ch_names:
            raise ValueError(
                f"{path.name}: channel names differ from earlier runs of the "
                "same subject"
            )

    if not per_run_X:
        return None

    # Runs occasionally differ by one sample at the window edge; clip to the
    # shortest so the tensor is well-formed rather than silently ragged.
    n_times = min(a.shape[2] for a in per_run_X)
    per_run_X = [a[:, :, :n_times] for a in per_run_X]

    X = np.concatenate(per_run_X, axis=0)
    return EpochBundle(
        X=X,
        y=np.concatenate(per_run_y),
        groups=np.full(len(X), subject, dtype=np.int64),
        runs=np.concatenate(per_run_r),
        classes=task.classes,
        ch_names=ch_names,  # type: ignore[arg-type]
        sfreq=float(sfreq),  # type: ignore[arg-type]
        tmin=tmin,
        tmax=tmax,
        task=task.name,
    )


__all__ = [
    "CACHE_VERSION",
    "DEFAULT_TMIN",
    "DEFAULT_TMAX",
    "EpochBundle",
    "concat_bundles",
    "epochs_from_raw",
    "load_subject_epochs",
    "read_standardised_raw",
]
