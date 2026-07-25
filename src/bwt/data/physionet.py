"""The PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB) protocol.

This module exists because the single largest defect in the previous version of
this project was an incorrect understanding of what the ``T0``/``T1``/``T2``
annotations mean. Their meaning is **run-dependent**:

    * In runs 3, 4, 7, 8, 11, 12 -> ``T1`` = left fist,  ``T2`` = right fist.
    * In runs 5, 6, 9, 10, 13, 14 -> ``T1`` = both fists, ``T2`` = both feet.

Merging those two families produces a class ``1`` that means "right fist OR both
feet" and a class ``0`` that means "left fist OR both fists" -- a target with no
coherent physiological meaning. Runs also differ in whether the movement was
*executed* or only *imagined*; a brain-computer interface is only meaningful on
imagined movement, because executed movement is contaminated by real EMG.

Every one of those distinctions is made explicit and non-optional below, so the
mistake cannot be repeated by accident.

Reference: Schalk et al. (2004), "BCI2000: A General-Purpose Brain-Computer
Interface (BCI) System", IEEE TBME 51(6):1034-1043; and the EEGMMIDB
documentation at https://physionet.org/content/eegmmidb/1.0.0/
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from bwt.paths import raw_data_dir

# --------------------------------------------------------------------------- #
# Subjects that must not be used
# --------------------------------------------------------------------------- #

#: Subjects whose recordings deviate from the protocol and are excluded from all
#: analyses. Verified directly against the EDF headers in this repository rather
#: than taken on trust:
#:
#:   S088, S092, S100 -- recorded at 128 Hz (not 160 Hz) with 5.125 s trials
#:                       (not 4.1 s), i.e. a different acquisition configuration.
#:   S089             -- baseline runs carry a single 60 s "T1" annotation
#:                       instead of a rest marker, and R03 is 181 s with 22 rest
#:                       events; its annotation timing is not trustworthy.
#:
#: This matches the exclusion set used throughout the EEGMMIDB literature.
EXCLUDED_SUBJECTS: frozenset[int] = frozenset({88, 89, 92, 100})

#: Number of subjects distributed in the full database.
N_SUBJECTS_TOTAL = 109

#: The sampling rate every usable recording must have.
EXPECTED_SFREQ = 160.0

#: The channel count every usable recording must have.
EXPECTED_N_CHANNELS = 64


# --------------------------------------------------------------------------- #
# Protocol vocabulary
# --------------------------------------------------------------------------- #


class Execution(str, Enum):
    """Whether a run asked for real movement or imagined movement."""

    EXECUTED = "executed"
    IMAGINED = "imagined"
    NONE = "none"  # baseline runs: no task at all


class Movement(str, Enum):
    """The physical or imagined act a trial represents."""

    REST = "rest"
    LEFT_FIST = "left_fist"
    RIGHT_FIST = "right_fist"
    BOTH_FISTS = "both_fists"
    BOTH_FEET = "both_feet"
    EYES_OPEN = "eyes_open"
    EYES_CLOSED = "eyes_closed"


@dataclass(frozen=True)
class RunSpec:
    """What one run number means."""

    run: int
    execution: Execution
    #: Maps the raw EDF annotation description to the act it denotes *in this run*.
    annotation_map: dict[str, Movement]
    description: str

    def movement_of(self, annotation: str) -> Movement | None:
        return self.annotation_map.get(annotation.strip())


# Run groups, named so that call sites read unambiguously.
BASELINE_RUNS = (1, 2)
EXECUTED_LR_RUNS = (3, 7, 11)
IMAGINED_LR_RUNS = (4, 8, 12)
EXECUTED_FF_RUNS = (5, 9, 13)
IMAGINED_FF_RUNS = (6, 10, 14)

TASK_RUNS = EXECUTED_LR_RUNS + IMAGINED_LR_RUNS + EXECUTED_FF_RUNS + IMAGINED_FF_RUNS

_LR_ANNOTATIONS = {
    "T0": Movement.REST,
    "T1": Movement.LEFT_FIST,
    "T2": Movement.RIGHT_FIST,
}
_FF_ANNOTATIONS = {
    "T0": Movement.REST,
    "T1": Movement.BOTH_FISTS,
    "T2": Movement.BOTH_FEET,
}

RUN_PROTOCOL: dict[int, RunSpec] = {}
for _r in BASELINE_RUNS:
    RUN_PROTOCOL[_r] = RunSpec(
        run=_r,
        execution=Execution.NONE,
        annotation_map={
            "T0": Movement.EYES_OPEN if _r == 1 else Movement.EYES_CLOSED
        },
        description="baseline, eyes open" if _r == 1 else "baseline, eyes closed",
    )
for _r in EXECUTED_LR_RUNS:
    RUN_PROTOCOL[_r] = RunSpec(_r, Execution.EXECUTED, dict(_LR_ANNOTATIONS),
                               "executed left or right fist")
for _r in IMAGINED_LR_RUNS:
    RUN_PROTOCOL[_r] = RunSpec(_r, Execution.IMAGINED, dict(_LR_ANNOTATIONS),
                               "imagined left or right fist")
for _r in EXECUTED_FF_RUNS:
    RUN_PROTOCOL[_r] = RunSpec(_r, Execution.EXECUTED, dict(_FF_ANNOTATIONS),
                               "executed both fists or both feet")
for _r in IMAGINED_FF_RUNS:
    RUN_PROTOCOL[_r] = RunSpec(_r, Execution.IMAGINED, dict(_FF_ANNOTATIONS),
                               "imagined both fists or both feet")


def run_spec(run: int) -> RunSpec:
    try:
        return RUN_PROTOCOL[run]
    except KeyError:
        raise ValueError(
            f"run {run} is not part of the EEGMMIDB protocol (valid: 1-14)"
        ) from None


# --------------------------------------------------------------------------- #
# Decoding tasks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskSpec:
    """A well-posed classification problem over the database.

    A task fixes *which runs* contribute trials and *which act maps to which
    class*. Acts that appear in those runs but are absent from ``label_map`` are
    discarded, which is how rest trials are excluded from a pure motor-imagery
    task without any implicit relabelling.
    """

    name: str
    description: str
    runs: tuple[int, ...]
    label_map: dict[Movement, str]
    #: Fixed class order; the index into this list is the integer label.
    classes: tuple[str, ...] = field(default=())
    #: Set only for tasks whose whole point is "is the user doing anything at
    #: all", where collapsing hand and foot movement into one class is the
    #: intent rather than the v1 labelling bug. Nothing else may set this, and
    #: the test suite enforces that.
    merges_effectors: bool = False

    def __post_init__(self) -> None:
        if not self.classes:
            seen: list[str] = []
            for value in self.label_map.values():
                if value not in seen:
                    seen.append(value)
            object.__setattr__(self, "classes", tuple(sorted(seen)))

        executions = {run_spec(r).execution for r in self.runs}
        if Execution.NONE in executions and len(executions) > 1:
            raise ValueError(
                f"task {self.name!r} mixes baseline runs with task runs; "
                "baseline recordings have no trial structure and cannot be "
                "pooled with cued trials"
            )

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def executions(self) -> set[Execution]:
        return {run_spec(r).execution for r in self.runs}

    def class_index(self, class_name: str) -> int:
        return self.classes.index(class_name)

    def class_of(self, run: int, annotation: str) -> str | None:
        """Return the class name for an annotation in a run, or ``None`` to drop it."""
        movement = run_spec(run).movement_of(annotation)
        if movement is None:
            return None
        return self.label_map.get(movement)

    def label_of(self, run: int, annotation: str) -> int | None:
        name = self.class_of(run, annotation)
        return None if name is None else self.class_index(name)


TASKS: dict[str, TaskSpec] = {
    # The canonical BCI benchmark: imagined left vs right hand. Everything about
    # this task is uncontaminated by real movement.
    "mi_left_right": TaskSpec(
        name="mi_left_right",
        description="Imagined left fist vs imagined right fist (runs 4, 8, 12)",
        runs=IMAGINED_LR_RUNS,
        label_map={
            Movement.LEFT_FIST: "left_fist",
            Movement.RIGHT_FIST: "right_fist",
        },
        classes=("left_fist", "right_fist"),
    ),
    # Same contrast but with real movement. Scores higher than the imagined task
    # because muscle activity leaks into the EEG; useful as an upper reference,
    # never as a BCI result.
    "me_left_right": TaskSpec(
        name="me_left_right",
        description="Executed left fist vs executed right fist (runs 3, 7, 11)",
        runs=EXECUTED_LR_RUNS,
        label_map={
            Movement.LEFT_FIST: "left_fist",
            Movement.RIGHT_FIST: "right_fist",
        },
        classes=("left_fist", "right_fist"),
    ),
    "mi_fists_feet": TaskSpec(
        name="mi_fists_feet",
        description="Imagined both fists vs imagined both feet (runs 6, 10, 14)",
        runs=IMAGINED_FF_RUNS,
        label_map={
            Movement.BOTH_FISTS: "both_fists",
            Movement.BOTH_FEET: "both_feet",
        },
        classes=("both_fists", "both_feet"),
    ),
    # Four imagined classes. This is what drives the 4-ary speller.
    "mi_four_class": TaskSpec(
        name="mi_four_class",
        description=(
            "Imagined left fist / right fist / both fists / both feet "
            "(runs 4, 6, 8, 10, 12, 14)"
        ),
        runs=tuple(sorted(IMAGINED_LR_RUNS + IMAGINED_FF_RUNS)),
        label_map={
            Movement.LEFT_FIST: "left_fist",
            Movement.RIGHT_FIST: "right_fist",
            Movement.BOTH_FISTS: "both_fists",
            Movement.BOTH_FEET: "both_feet",
        },
        classes=("left_fist", "right_fist", "both_fists", "both_feet"),
    ),
    # Asynchronous BCI gate: is the user attempting anything at all? A speller
    # that runs continuously needs this, otherwise it emits a letter every time
    # the user blinks or looks away. Here the merge across effectors is the
    # point -- the positive class is "any imagined movement" -- which is why
    # this is the one task allowed to set `merges_effectors`.
    "mi_move_vs_rest": TaskSpec(
        name="mi_move_vs_rest",
        description=(
            "Any imagined movement vs cued rest, for asynchronous control "
            "(runs 4, 6, 8, 10, 12, 14)"
        ),
        runs=tuple(sorted(IMAGINED_LR_RUNS + IMAGINED_FF_RUNS)),
        label_map={
            Movement.REST: "rest",
            Movement.LEFT_FIST: "movement",
            Movement.RIGHT_FIST: "movement",
            Movement.BOTH_FISTS: "movement",
            Movement.BOTH_FEET: "movement",
        },
        classes=("rest", "movement"),
        merges_effectors=True,
    ),
    # Adds the cued rest period as an explicit third class. Note this is the
    # *within-run* rest cue, not the separate baseline recordings.
    "mi_left_right_rest": TaskSpec(
        name="mi_left_right_rest",
        description="Imagined left fist vs right fist vs cued rest (runs 4, 8, 12)",
        runs=IMAGINED_LR_RUNS,
        label_map={
            Movement.LEFT_FIST: "left_fist",
            Movement.RIGHT_FIST: "right_fist",
            Movement.REST: "rest",
        },
        classes=("left_fist", "right_fist", "rest"),
    ),
}

DEFAULT_TASK = "mi_left_right"


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(
            f"unknown task {name!r}; available: {', '.join(sorted(TASKS))}"
        ) from None


# --------------------------------------------------------------------------- #
# Filesystem discovery
# --------------------------------------------------------------------------- #

_SUBJECT_RE = re.compile(r"^S(\d{3})$")


def subject_id(subject: int) -> str:
    return f"S{subject:03d}"


def edf_path(subject: int, run: int, root: Path | None = None) -> Path:
    root = root or raw_data_dir()
    return root / subject_id(subject) / f"{subject_id(subject)}R{run:02d}.edf"


def available_subjects(
    root: Path | None = None,
    *,
    include_excluded: bool = False,
    require_runs: tuple[int, ...] | None = None,
) -> list[int]:
    """List subjects present on disk, in ascending order.

    Parameters
    ----------
    include_excluded
        When ``False`` (the default) the protocol-violating subjects listed in
        :data:`EXCLUDED_SUBJECTS` are omitted.
    require_runs
        Only return subjects that have every one of these runs on disk.
    """
    root = root or raw_data_dir()
    if not root.is_dir():
        return []

    found: list[int] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = _SUBJECT_RE.match(entry.name)
        if not match:
            continue
        number = int(match.group(1))
        if not include_excluded and number in EXCLUDED_SUBJECTS:
            continue
        if require_runs and not all(
            edf_path(number, r, root).is_file() for r in require_runs
        ):
            continue
        found.append(number)
    return found


__all__ = [
    "EXCLUDED_SUBJECTS",
    "EXPECTED_N_CHANNELS",
    "EXPECTED_SFREQ",
    "N_SUBJECTS_TOTAL",
    "DEFAULT_TASK",
    "TASKS",
    "TASK_RUNS",
    "BASELINE_RUNS",
    "EXECUTED_LR_RUNS",
    "IMAGINED_LR_RUNS",
    "EXECUTED_FF_RUNS",
    "IMAGINED_FF_RUNS",
    "RUN_PROTOCOL",
    "Execution",
    "Movement",
    "RunSpec",
    "TaskSpec",
    "available_subjects",
    "edf_path",
    "get_task",
    "run_spec",
    "subject_id",
]
