"""Adapting a population model to one person.

A model trained across many subjects generalises weakly, because most of the
variance in EEG is *between* people rather than between mental states: skull
thickness, electrode impedance, cortical folding and rhythm frequency all differ
enough that a spatial filter tuned to the population is a compromise for
everyone. Real brain-computer interfaces therefore run a short calibration
session before use.

This module measures that honestly. :func:`calibration_curve` sweeps the number
of calibration trials and reports accuracy at each point, always evaluating on
trials the adapted model has never seen. The result answers the question a user
actually has -- *how long do I have to sit still before this works?* -- rather
than reporting a single number that hides the cost.

Three strategies:

``none``
    The population model, unchanged. The baseline to beat.
``finetune``
    Continue training a pretrained network on the target subject's trials. Only
    available for the neural pipelines.
``refit``
    Fit a fresh classical pipeline on the target subject's trials alone,
    ignoring the population. Strong once there are enough trials, useless when
    there are few.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit

from bwt.data.epochs import EpochBundle
from bwt.logging_utils import get_logger

log = get_logger(__name__)

STRATEGIES = ("none", "finetune", "refit")


@dataclass
class CalibrationPoint:
    """Accuracy for one subject at one calibration budget."""

    subject: int
    strategy: str
    n_calibration_trials: int
    accuracy: float
    kappa: float
    n_eval_trials: int
    seconds: float = 0.0


@dataclass
class CalibrationResult:
    pipeline: str
    task: str
    dataset: str
    classes: list[str]
    points: list[CalibrationPoint] = field(default_factory=list)
    chance_level: float = 0.5

    def by_budget(self, strategy: str) -> dict[int, list[float]]:
        out: dict[int, list[float]] = {}
        for point in self.points:
            if point.strategy == strategy:
                out.setdefault(point.n_calibration_trials, []).append(point.accuracy)
        return dict(sorted(out.items()))

    def _points(self, strategy: str) -> dict[int, list[CalibrationPoint]]:
        out: dict[int, list[CalibrationPoint]] = {}
        for point in self.points:
            if point.strategy == strategy:
                out.setdefault(point.n_calibration_trials, []).append(point)
        return dict(sorted(out.items()))

    def summary_table(self) -> list[dict]:
        rows = []
        for strategy in STRATEGIES:
            for budget, points in self._points(strategy).items():
                arr = np.asarray([p.accuracy for p in points])
                rows.append({
                    "strategy": strategy,
                    "n_calibration_trials": budget,
                    "mean_accuracy": float(arr.mean()),
                    "std_accuracy": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    # Distinguished deliberately: every budget above zero is
                    # repeated `n_splits` times per subject, so the number of
                    # measurements is a multiple of the number of people. Only
                    # the subject count says how much the estimate generalises.
                    "n_subjects": len({p.subject for p in points}),
                    "n_measurements": len(arr),
                })
        return rows

    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline,
            "task": self.task,
            "dataset": self.dataset,
            "classes": self.classes,
            "chance_level": self.chance_level,
            "table": self.summary_table(),
            "points": [asdict(p) for p in self.points],
        }

    def summary(self) -> str:
        lines = [f"calibration / {self.pipeline} / {self.dataset}:{self.task}",
                 f"  chance {self.chance_level:.3f}"]
        for row in self.summary_table():
            lines.append(
                f"  {row['strategy']:9s} n={row['n_calibration_trials']:3d} "
                f"acc={row['mean_accuracy']:.4f} "
                f"+/-{row['std_accuracy']:.4f} "
                f"({row['n_subjects']} subjects, "
                f"{row['n_measurements']} measurements)"
            )
        return "\n".join(lines)


def _is_deep(model) -> bool:
    from bwt.deep import TorchClassifier

    final = model[-1] if hasattr(model, "__getitem__") else model
    return isinstance(final, TorchClassifier)


def adapt(model, X: np.ndarray, y: np.ndarray, *, strategy: str = "finetune",
          epochs: int = 40, lr: float | None = None,
          freeze_features: bool = False):
    """Return a copy of ``model`` adapted to one subject's trials.

    The input model is never mutated, so the same population model can be
    adapted to many subjects in a loop.
    """
    if strategy == "none":
        return model
    if strategy != "finetune":
        raise ValueError(f"adapt() handles 'none' and 'finetune', not {strategy!r}")
    if not _is_deep(model):
        raise TypeError(
            "fine-tuning requires a neural pipeline; use strategy='refit' for "
            "classical models"
        )

    import copy as _copy

    adapted = _copy.deepcopy(model)
    classifier = adapted[-1]
    inner = classifier.clone_for_finetuning(freeze_features=freeze_features)
    adapted.steps[-1] = (adapted.steps[-1][0], inner)

    # Run the calibration trials through the frozen preprocessing steps, then
    # fine-tune only the network.
    Xt = X
    for _, step in adapted.steps[:-1]:
        Xt = step.transform(Xt)
    inner.partial_fit(Xt, y, epochs=epochs, lr=lr)
    return adapted


def calibration_curve(
    bundle: EpochBundle,
    factory: Callable[[], object],
    *,
    pipeline_name: str = "pipeline",
    dataset: str = "eegmmidb",
    budgets: Sequence[int] = (0, 5, 10, 20, 40),
    strategies: Sequence[str] = ("none", "finetune"),
    n_splits: int = 5,
    finetune_epochs: int = 40,
    max_subjects: int | None = None,
    random_state: int = 42,
) -> CalibrationResult:
    """Measure accuracy as a function of calibration trials per subject.

    For each held-out subject: a population model is trained on **all other
    subjects**, then adapted using ``n`` of the target's trials, then scored on
    the target's remaining trials. The population model never sees the target
    subject, and the calibration trials never appear in the evaluation set.
    """
    classes = list(bundle.classes)
    n_classes = len(classes)
    result = CalibrationResult(
        pipeline=pipeline_name, task=bundle.task, dataset=dataset,
        classes=classes, chance_level=1.0 / n_classes,
    )

    subjects = bundle.subjects
    if max_subjects is not None:
        rng = np.random.default_rng(random_state)
        subjects = sorted(rng.choice(subjects,
                                     size=min(max_subjects, len(subjects)),
                                     replace=False).tolist())

    for position, subject in enumerate(subjects, start=1):
        target = bundle.groups == subject
        X_pop, y_pop = bundle.X[~target], bundle.y[~target]
        X_sub, y_sub = bundle.X[target], bundle.y[target]

        if len(np.unique(y_sub)) < n_classes or len(y_sub) < max(budgets) + 5:
            log.warning("subject %s: too few trials for the requested budgets",
                        subject)
            continue

        log.info("calibration %d/%d: subject %s (population=%d trials)",
                 position, len(subjects), subject, len(y_pop))

        t0 = time.time()
        population = factory()
        population.fit(X_pop, y_pop)
        log.info("  population model fitted in %.0fs", time.time() - t0)

        for budget in budgets:
            if budget == 0:
                for strategy in strategies:
                    if strategy != "none":
                        continue
                    t1 = time.time()
                    pred = population.predict(X_sub)
                    result.points.append(CalibrationPoint(
                        subject=subject, strategy="none", n_calibration_trials=0,
                        accuracy=float(accuracy_score(y_sub, pred)),
                        kappa=float(cohen_kappa_score(y_sub, pred)),
                        n_eval_trials=len(y_sub), seconds=time.time() - t1,
                    ))
                continue

            # Repeat the split so a single lucky calibration set does not drive
            # the estimate.
            splitter = StratifiedShuffleSplit(
                n_splits=n_splits, train_size=budget, random_state=random_state
            )
            for calib_idx, eval_idx in splitter.split(X_sub, y_sub):
                X_cal, y_cal = X_sub[calib_idx], y_sub[calib_idx]
                X_ev, y_ev = X_sub[eval_idx], y_sub[eval_idx]
                if len(np.unique(y_cal)) < 2:
                    continue

                for strategy in strategies:
                    if strategy == "none":
                        continue
                    t1 = time.time()
                    try:
                        if strategy == "finetune":
                            adapted = adapt(
                                population, X_cal, y_cal, strategy="finetune",
                                epochs=finetune_epochs,
                            )
                        elif strategy == "refit":
                            adapted = factory()
                            adapted.fit(X_cal, y_cal)
                        else:
                            raise ValueError(f"unknown strategy {strategy!r}")
                        pred = adapted.predict(X_ev)
                    except Exception as exc:
                        log.warning("  %s @ n=%d failed: %s", strategy, budget, exc)
                        continue

                    result.points.append(CalibrationPoint(
                        subject=subject, strategy=strategy,
                        n_calibration_trials=budget,
                        accuracy=float(accuracy_score(y_ev, pred)),
                        kappa=float(cohen_kappa_score(y_ev, pred)),
                        n_eval_trials=len(y_ev), seconds=time.time() - t1,
                    ))

        done = {r["strategy"]: r for r in result.summary_table()}
        log.info("  subject %s done (%.0fs total)", subject, time.time() - t0)

    return result


__all__ = [
    "STRATEGIES",
    "CalibrationPoint",
    "CalibrationResult",
    "adapt",
    "calibration_curve",
]
