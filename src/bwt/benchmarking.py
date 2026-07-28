"""Resumable benchmarking.

Cross-validating six pipelines over 105 subjects takes hours, and anything that
takes hours will eventually be interrupted -- a laptop sleeps, a session ends, a
process is killed. A benchmark that loses all its work when that happens is not
usable in practice, so this module checkpoints at the level of a single fold.

Every fold's predictions are written to a JSON store as soon as they exist.
Re-running skips whatever is already recorded, so progress accumulates across as
many interrupted attempts as it takes. A wall-clock budget lets a run stop
cleanly at a fold boundary rather than being killed mid-fit.

    bwt benchmark --time-budget 600     # run for ten minutes, then stop
    bwt benchmark --time-budget 600     # ...continue from where it stopped

The store also makes results auditable: the per-fold true and predicted labels
are kept, not just the summary statistics, so any reported number can be
recomputed without re-training.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold

from bwt.data.epochs import EpochBundle
from bwt.evaluation import CVResult, _levels, _score_fold, assert_no_subject_leakage
from bwt.logging_utils import get_logger

log = get_logger(__name__)

#: Bump when the meaning of a stored fold changes.
STORE_VERSION = 1


class BudgetExhausted(Exception):
    """Raised internally when the wall-clock budget runs out."""


@dataclass
class WorkItem:
    """One unit of resumable work: a single fold of one protocol."""

    pipeline: str
    protocol: str
    key: str  # fold index, or subject id for within-subject

    @property
    def store_key(self) -> str:
        return f"{self.pipeline}|{self.protocol}|{self.key}"


class CheckpointStore:
    """A JSON-backed record of completed folds.

    Written atomically after every fold, because the whole point is surviving an
    abrupt termination.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict = {"version": STORE_VERSION, "folds": {}}
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text())
                if loaded.get("version") == STORE_VERSION:
                    self.data = loaded
                else:
                    log.warning("checkpoint %s has version %s, expected %s; "
                                "starting fresh", self.path,
                                loaded.get("version"), STORE_VERSION)
            except Exception as exc:
                log.warning("unreadable checkpoint %s (%s); starting fresh",
                            self.path, exc)

    def has(self, item: WorkItem) -> bool:
        return item.store_key in self.data["folds"]

    def get(self, item: WorkItem) -> dict:
        return self.data["folds"][item.store_key]

    def put(self, item: WorkItem, y_true, y_pred, subjects, seconds: float,
            n_train: int) -> None:
        self.data["folds"][item.store_key] = {
            "y_true": [int(v) for v in y_true],
            "y_pred": [int(v) for v in y_pred],
            "subjects": [int(s) for s in subjects],
            "seconds": float(seconds),
            "n_train": int(n_train),
        }
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(self.data))
        tmp.replace(self.path)

    def folds_for(self, pipeline: str, protocol: str) -> dict[str, dict]:
        prefix = f"{pipeline}|{protocol}|"
        return {k[len(prefix):]: v for k, v in self.data["folds"].items()
                if k.startswith(prefix)}


def plan(
    bundle: EpochBundle,
    pipelines: Sequence[str],
    protocols: Sequence[str],
    n_splits: int = 5,
) -> list[WorkItem]:
    """Enumerate every fold that a full benchmark would need."""
    items: list[WorkItem] = []
    for pipeline in pipelines:
        for protocol in protocols:
            if protocol == "cross_subject":
                items.extend(
                    WorkItem(pipeline, protocol, str(i)) for i in range(n_splits)
                )
            elif protocol == "within_subject":
                items.extend(
                    WorkItem(pipeline, protocol, str(s)) for s in bundle.subjects
                )
            else:
                raise ValueError(f"unsupported protocol {protocol!r}")
    return items


def _run_cross_subject_fold(bundle, factory, fold_index: int, n_splits: int):
    X, y, groups = bundle.X, bundle.y, bundle.groups
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))
    train_idx, test_idx = splits[fold_index]
    assert_no_subject_leakage(groups, train_idx, test_idx)

    model = factory()
    started = time.time()
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    return (y[test_idx], pred, np.unique(groups[test_idx]),
            time.time() - started, len(train_idx))


def _run_within_subject_fold(bundle, factory, subject: int, n_splits: int,
                             min_trials: int = 20):
    mask = bundle.groups == subject
    Xs, ys = bundle.X[mask], bundle.y[mask]
    n_classes = len(bundle.classes)
    if len(ys) < min_trials or len(np.unique(ys)) < 2:
        return None
    counts = np.bincount(ys, minlength=n_classes)
    splits = int(min(n_splits, counts[counts > 0].min()))
    if splits < 2:
        return None

    cv = StratifiedKFold(splits, shuffle=True, random_state=42)
    truth, pred = [], []
    started = time.time()
    for train_idx, test_idx in cv.split(Xs, ys):
        model = factory()
        model.fit(Xs[train_idx], ys[train_idx])
        pred.append(model.predict(Xs[test_idx]))
        truth.append(ys[test_idx])
    return (np.concatenate(truth), np.concatenate(pred), [subject],
            time.time() - started, len(ys))


def execute(
    bundle: EpochBundle,
    items: Sequence[WorkItem],
    store: CheckpointStore,
    factory_for,
    *,
    n_splits: int = 5,
    time_budget: float | None = None,
) -> Iterator[tuple[WorkItem, str]]:
    """Run outstanding work items, checkpointing each one.

    Yields ``(item, status)`` where status is ``cached``, ``done``, ``skipped``
    or ``error``. Stops cleanly once ``time_budget`` seconds have elapsed,
    always at a fold boundary.

    At least one outstanding item is always attempted, even if the budget is
    already exhausted. Without that guarantee a budget shorter than a single
    fold would make no progress at all, and repeated invocations would loop
    forever without ever finishing the benchmark.
    """
    started = time.time()
    attempted = 0

    for item in items:
        if store.has(item):
            yield item, "cached"
            continue

        if (time_budget is not None and attempted > 0
                and time.time() - started >= time_budget):
            log.info("time budget reached; %d item(s) still outstanding",
                     sum(1 for i in items if not store.has(i)))
            return
        attempted += 1

        try:
            factory = factory_for(item.pipeline)
            if item.protocol == "cross_subject":
                outcome = _run_cross_subject_fold(
                    bundle, factory, int(item.key), n_splits
                )
            else:
                outcome = _run_within_subject_fold(
                    bundle, factory, int(item.key), n_splits
                )

            if outcome is None:
                # Record the skip so it is not retried on every resume.
                store.put(item, [], [], [int(item.key)], 0.0, 0)
                yield item, "skipped"
                continue

            y_true, y_pred, subjects, seconds, n_train = outcome
            store.put(item, y_true, y_pred, subjects, seconds, n_train)
            accuracy = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
            log.info("  %s %s fold %s: acc=%.4f (%.0fs)",
                     item.pipeline, item.protocol, item.key, accuracy, seconds)
            yield item, "done"

        except Exception as exc:
            log.error("%s failed: %s: %s", item.store_key, type(exc).__name__, exc)
            yield item, "error"


def assemble(
    bundle: EpochBundle,
    store: CheckpointStore,
    pipeline: str,
    protocol: str,
) -> CVResult | None:
    """Build a :class:`CVResult` from checkpointed folds, or ``None`` if empty."""
    folds_raw = store.folds_for(pipeline, protocol)
    usable = {k: v for k, v in folds_raw.items() if v["y_true"]}
    if not usable:
        return None

    n_classes = len(bundle.classes)
    folds = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for key in sorted(usable, key=lambda k: int(k)):
        entry = usable[key]
        truth = np.asarray(entry["y_true"])
        pred = np.asarray(entry["y_pred"])
        all_true.append(truth)
        all_pred.append(pred)
        folds.append(
            _score_fold(truth, pred, int(key), entry["n_train"], len(truth),
                        entry["subjects"], entry["seconds"])
        )

    chance, majority = _levels(bundle.y, n_classes)
    return CVResult(
        protocol=protocol,
        pipeline=pipeline,
        task=bundle.task,
        classes=list(bundle.classes),
        folds=folds,
        chance_level=chance,
        majority_level=majority,
        confusion=confusion_matrix(
            np.concatenate(all_true), np.concatenate(all_pred),
            labels=list(range(n_classes)),
        ).tolist(),
        n_trials=bundle.n_trials,
        n_subjects=len(bundle.subjects),
        elapsed_seconds=sum(f.fit_seconds for f in folds),
    )


def progress(items: Sequence[WorkItem], store: CheckpointStore) -> dict:
    """How much of the plan is complete, broken down by pipeline and protocol."""
    out: dict[str, dict[str, list[int]]] = {}
    for item in items:
        row = out.setdefault(item.pipeline, {}).setdefault(item.protocol, [0, 0])
        row[1] += 1
        if store.has(item):
            row[0] += 1
    return out


__all__ = [
    "STORE_VERSION",
    "CheckpointStore",
    "WorkItem",
    "assemble",
    "execute",
    "plan",
    "progress",
]
