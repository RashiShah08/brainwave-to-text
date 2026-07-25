"""Cross-validation protocols that cannot leak between subjects.

The previous version of this project reported 94.3% accuracy. That number came
from evaluating on the same rows the model was trained on. When the model was
scored on genuinely held-out trials it managed 72.1% against a 67.9% majority-
class baseline -- and even that was optimistic, because a plain random split
puts adjacent one-second windows from the *same* recording on both sides of the
split.

This module makes the honest measurement the easy one:

* :func:`within_subject_cv` -- the number a real BCI would deliver, since a BCI
  is calibrated on its user. Stratified k-fold *inside* each subject, reported
  as a distribution over subjects rather than a single pooled figure.
* :func:`cross_subject_cv` -- the zero-calibration number. Whole subjects are
  held out, so nothing about the test subject is ever seen during training.
* :func:`permutation_test` -- is the score distinguishable from chance at all?

Every split is checked by :func:`assert_no_subject_leakage` before it is used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

from bwt.data.epochs import EpochBundle
from bwt.logging_utils import get_logger

log = get_logger(__name__)

PipelineFactory = Callable[[], object]


# --------------------------------------------------------------------------- #
# Leakage guards
# --------------------------------------------------------------------------- #


def assert_no_subject_leakage(
    groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    """Raise if any subject appears on both sides of a split.

    Called on every fold of every cross-subject evaluation. It is cheap, and it
    converts the single most damaging silent bug in this domain into a crash.
    """
    if np.intersect1d(train_idx, test_idx).size:
        raise AssertionError("train and test index sets overlap")
    shared = np.intersect1d(np.unique(groups[train_idx]), np.unique(groups[test_idx]))
    if shared.size:
        raise AssertionError(
            f"subject(s) {shared.tolist()} appear in both train and test; "
            "this split leaks subject identity and its score is meaningless"
        )


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


def _ci95(values: Sequence[float]) -> tuple[float, float]:
    """Normal-approximation 95% CI of the mean. Returns (lo, hi)."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return (float("nan"), float("nan"))
    half = 1.96 * arr.std(ddof=1) / np.sqrt(arr.size)
    return (float(arr.mean() - half), float(arr.mean() + half))


@dataclass
class FoldResult:
    fold: int
    accuracy: float
    balanced_accuracy: float
    kappa: float
    f1_macro: float
    n_train: int
    n_test: int
    test_subjects: list[int] = field(default_factory=list)
    fit_seconds: float = 0.0


@dataclass
class CVResult:
    """Outcome of one evaluation protocol."""

    protocol: str
    pipeline: str
    task: str
    classes: list[str]
    folds: list[FoldResult]
    chance_level: float
    majority_level: float
    confusion: list[list[int]]
    n_trials: int
    n_subjects: int
    elapsed_seconds: float = 0.0
    permutation_p: float | None = None
    permutation_scores: list[float] = field(default_factory=list)

    @property
    def accuracies(self) -> np.ndarray:
        return np.array([f.accuracy for f in self.folds])

    @property
    def mean_accuracy(self) -> float:
        return float(self.accuracies.mean())

    @property
    def std_accuracy(self) -> float:
        return float(self.accuracies.std(ddof=1)) if len(self.folds) > 1 else 0.0

    @property
    def ci95(self) -> tuple[float, float]:
        return _ci95(self.accuracies)

    @property
    def mean_kappa(self) -> float:
        return float(np.mean([f.kappa for f in self.folds]))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            mean_accuracy=self.mean_accuracy,
            std_accuracy=self.std_accuracy,
            ci95_low=self.ci95[0],
            ci95_high=self.ci95[1],
            mean_kappa=self.mean_kappa,
        )
        return payload

    def summary(self) -> str:
        lo, hi = self.ci95
        parts = [
            f"{self.protocol} / {self.pipeline} / {self.task}",
            f"  accuracy      {self.mean_accuracy:.4f} +/- {self.std_accuracy:.4f}"
            f"  (95% CI {lo:.4f}-{hi:.4f}, n={len(self.folds)})",
            f"  kappa         {self.mean_kappa:.4f}",
            f"  chance        {self.chance_level:.4f}"
            f"   majority {self.majority_level:.4f}",
        ]
        if self.permutation_p is not None:
            parts.append(f"  permutation p {self.permutation_p:.4g}")
        return "\n".join(parts)


def _score_fold(y_true, y_pred, fold, n_train, n_test, subjects, seconds) -> FoldResult:
    return FoldResult(
        fold=fold,
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        kappa=float(cohen_kappa_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        n_train=int(n_train),
        n_test=int(n_test),
        test_subjects=[int(s) for s in subjects],
        fit_seconds=float(seconds),
    )


def _levels(y: np.ndarray, n_classes: int) -> tuple[float, float]:
    """Chance level (uniform) and majority-class level for this label vector."""
    counts = np.bincount(y, minlength=n_classes)
    return 1.0 / n_classes, float(counts.max() / counts.sum())


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #


def cross_subject_cv(
    bundle: EpochBundle,
    factory: PipelineFactory,
    *,
    pipeline_name: str = "pipeline",
    n_splits: int = 5,
    n_jobs: int = 1,
) -> CVResult:
    """Hold out whole subjects. This is the zero-calibration generalisation score.

    Uses :class:`GroupKFold` on the subject id, with an explicit leakage
    assertion on every fold.
    """
    started = time.time()
    X, y, groups = bundle.X, bundle.y, bundle.groups
    n_classes = len(bundle.classes)
    splitter = GroupKFold(n_splits=n_splits)

    folds: list[FoldResult] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    def _run(fold: int, train_idx: np.ndarray, test_idx: np.ndarray):
        assert_no_subject_leakage(groups, train_idx, test_idx)
        model = factory()
        t0 = time.time()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        return fold, test_idx, pred, time.time() - t0

    jobs = list(enumerate(splitter.split(X, y, groups=groups), start=1))
    if n_jobs == 1:
        outputs = [_run(f, tr, te) for f, (tr, te) in jobs]
    else:
        from joblib import Parallel, delayed

        outputs = Parallel(n_jobs=n_jobs)(
            delayed(_run)(f, tr, te) for f, (tr, te) in jobs
        )

    for fold, test_idx, pred, seconds in outputs:
        truth = y[test_idx]
        all_true.append(truth)
        all_pred.append(pred)
        result = _score_fold(
            truth, pred, fold, len(X) - len(test_idx), len(test_idx),
            np.unique(groups[test_idx]), seconds,
        )
        folds.append(result)
        log.info(
            "  cross-subject fold %d/%d: acc=%.4f (%d test trials, %d subjects, %.1fs)",
            fold, n_splits, result.accuracy, result.n_test,
            len(result.test_subjects), seconds,
        )

    chance, majority = _levels(y, n_classes)
    return CVResult(
        protocol="cross_subject",
        pipeline=pipeline_name,
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
        elapsed_seconds=time.time() - started,
    )


def within_subject_cv(
    bundle: EpochBundle,
    factory: PipelineFactory,
    *,
    pipeline_name: str = "pipeline",
    n_splits: int = 5,
    n_jobs: int = 1,
    min_trials: int = 20,
) -> CVResult:
    """Calibrate and test inside each subject separately.

    Each subject yields one accuracy, and the reported figure is the mean over
    subjects with a confidence interval -- which is what the BCI literature
    means by "accuracy" for this dataset. Reporting the spread matters: subject
    performance on motor imagery is famously bimodal, and a mean alone hides the
    well-documented fraction of users for whom the paradigm simply does not work.
    """
    started = time.time()
    n_classes = len(bundle.classes)

    def _one(subject: int, Xs: np.ndarray, ys: np.ndarray):
        if len(ys) < min_trials or len(np.unique(ys)) < 2:
            return None
        counts = np.bincount(ys, minlength=n_classes)
        splits = int(min(n_splits, counts[counts > 0].min()))
        if splits < 2:
            return None

        cv = StratifiedKFold(splits, shuffle=True, random_state=42)
        truth, pred = [], []
        t0 = time.time()
        for train_idx, test_idx in cv.split(Xs, ys):
            model = factory()
            model.fit(Xs[train_idx], ys[train_idx])
            pred.append(model.predict(Xs[test_idx]))
            truth.append(ys[test_idx])
        truth, pred = np.concatenate(truth), np.concatenate(pred)
        return subject, truth, pred, time.time() - t0

    # Slice per subject up front and pass the small arrays as explicit
    # arguments. Capturing `bundle` in the closure instead would ship the whole
    # multi-hundred-megabyte tensor to every worker process.
    subjects = bundle.subjects
    per_subject = [(s, bundle.X[bundle.groups == s], bundle.y[bundle.groups == s])
                   for s in subjects]

    if n_jobs == 1:
        outputs = []
        for index, args in enumerate(per_subject, start=1):
            outputs.append(_one(*args))
            if index % 10 == 0 or index == len(subjects):
                log.info("  within-subject %d/%d subjects", index, len(subjects))
    else:
        from joblib import Parallel, delayed

        outputs = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_one)(*args) for args in per_subject
        )

    folds: list[FoldResult] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    for output in outputs:
        if output is None:
            continue
        subject, truth, pred, seconds = output
        all_true.append(truth)
        all_pred.append(pred)
        folds.append(
            _score_fold(truth, pred, subject, 0, len(truth), [subject], seconds)
        )

    if not folds:
        raise RuntimeError("no subject had enough trials for within-subject CV")

    skipped = len(subjects) - len(folds)
    if skipped:
        log.warning("%d subject(s) skipped (too few trials)", skipped)

    chance, majority = _levels(bundle.y, n_classes)
    return CVResult(
        protocol="within_subject",
        pipeline=pipeline_name,
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
        elapsed_seconds=time.time() - started,
    )


def out_of_fold_probabilities(
    bundle: EpochBundle,
    factory: PipelineFactory,
    *,
    n_splits: int = 5,
) -> np.ndarray:
    """Class probabilities for every trial, each predicted by a model that
    never saw that trial's subject.

    Needed by :func:`bwt.streaming.evaluate_accumulation`, which must reason
    about the real error structure of a model -- including the fact that a given
    subject's errors are correlated with each other -- rather than assuming
    independent draws.
    """
    X, y, groups = bundle.X, bundle.y, bundle.groups
    proba = np.zeros((len(X), len(bundle.classes)), dtype=np.float64)

    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=n_splits).split(X, y, groups=groups), start=1
    ):
        assert_no_subject_leakage(groups, train_idx, test_idx)
        model = factory()
        model.fit(X[train_idx], y[train_idx])
        proba[test_idx] = model.predict_proba(X[test_idx])
        log.info("  out-of-fold probabilities: fold %d/%d", fold, n_splits)
    return proba


def permutation_test(
    bundle: EpochBundle,
    factory: PipelineFactory,
    *,
    observed: float,
    n_permutations: int = 100,
    n_splits: int = 5,
    n_jobs: int = 1,
    random_state: int = 42,
) -> tuple[float, list[float]]:
    """Estimate how often a shuffled-label model reaches ``observed``.

    Labels are permuted **within each subject**, which destroys the class
    signal while preserving subject structure -- the correct null for a
    grouped design.

    Returns ``(p_value, scores)`` with the conventional ``(hits + 1) / (n + 1)``
    estimator, so the p-value is never reported as exactly zero.
    """
    rng = np.random.default_rng(random_state)
    X, y, groups = bundle.X, bundle.y, bundle.groups
    scores: list[float] = []

    for index in range(n_permutations):
        shuffled = y.copy()
        for subject in np.unique(groups):
            mask = groups == subject
            shuffled[mask] = rng.permutation(y[mask])

        splitter = GroupKFold(n_splits=n_splits)
        fold_scores = []
        for train_idx, test_idx in splitter.split(X, shuffled, groups=groups):
            assert_no_subject_leakage(groups, train_idx, test_idx)
            model = factory()
            model.fit(X[train_idx], shuffled[train_idx])
            fold_scores.append(
                accuracy_score(shuffled[test_idx], model.predict(X[test_idx]))
            )
        scores.append(float(np.mean(fold_scores)))
        if (index + 1) % 10 == 0:
            log.info("  permutation %d/%d (mean so far %.4f)",
                     index + 1, n_permutations, float(np.mean(scores)))

    hits = int(np.sum(np.asarray(scores) >= observed))
    return (hits + 1) / (n_permutations + 1), scores


__all__ = [
    "CVResult",
    "FoldResult",
    "assert_no_subject_leakage",
    "cross_subject_cv",
    "out_of_fold_probabilities",
    "permutation_test",
    "within_subject_cv",
]
