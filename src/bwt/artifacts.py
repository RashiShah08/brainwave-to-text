"""Versioned model persistence.

A saved model here is a *directory*, not a bare pickle:

    artifacts/mi_left_right__riemann_ts/
        pipeline.joblib     the fitted scikit-learn pipeline
        model_card.json     what it is, what it was trained on, how well it works

The card exists because the failure mode this project is recovering from was
precisely a ``.pkl`` with no record of what its two output classes meant. Every
number a user might quote is stored next to the weights, together with the
library versions needed to load them and the exact input contract the pipeline
expects.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from bwt import ARTIFACT_SCHEMA_VERSION, __version__
from bwt.logging_utils import get_logger
from bwt.paths import artifacts_dir

log = get_logger(__name__)

PIPELINE_FILE = "pipeline.joblib"
CARD_FILE = "model_card.json"


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "bwt": __version__,
    }
    for module in ("numpy", "scipy", "sklearn", "mne", "pyriemann", "joblib"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            versions[module] = "not-installed"
    return versions


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class ModelCard:
    """Everything needed to interpret, reproduce, and safely serve a model."""

    name: str
    task: str
    task_description: str
    pipeline: str
    classes: list[str]

    # -- input contract: violate any of these and predictions are meaningless --
    sfreq: float
    n_channels: int
    n_times: int
    ch_names: list[str]
    tmin: float
    tmax: float
    units: str = "uV"

    # -- provenance --
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    created_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    library_versions: dict[str, str] = field(default_factory=_library_versions)
    git_commit: str | None = field(default_factory=_git_commit)

    # -- training data --
    n_train_trials: int = 0
    train_subjects: list[int] = field(default_factory=list)
    excluded_subjects: list[int] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)

    # -- measured performance; keyed by protocol --
    evaluation: dict[str, Any] = field(default_factory=dict)
    chance_level: float = 0.0
    majority_level: float = 0.0

    # -- serving flags --
    requires_batch_recentering: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> ModelCard:
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})

    def headline(self) -> str:
        """One-line honest performance statement, or a clear absence of one."""
        within = self.evaluation.get("within_subject", {})
        cross = self.evaluation.get("cross_subject", {})
        bits = []
        if within.get("mean_accuracy") is not None:
            bits.append(
                f"within-subject {within['mean_accuracy']:.1%} "
                f"(+/-{within.get('std_accuracy', 0):.1%})"
            )
        if cross.get("mean_accuracy") is not None:
            bits.append(f"cross-subject {cross['mean_accuracy']:.1%}")
        if not bits:
            return "no evaluation recorded"
        return f"{'; '.join(bits)} vs {self.chance_level:.1%} chance"


def artifact_path(task: str, pipeline: str, root: Path | None = None) -> Path:
    return (root or artifacts_dir()) / f"{task}__{pipeline}"


def save_artifact(model, card: ModelCard, path: Path | None = None) -> Path:
    """Persist a fitted pipeline and its card atomically enough for a service.

    The card is written *after* the pipeline, so a reader that finds a card can
    rely on the weights being complete.
    """
    path = Path(path or artifact_path(card.task, card.pipeline))
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path / PIPELINE_FILE, compress=3)
    (path / CARD_FILE).write_text(
        json.dumps(card.to_dict(), indent=2, sort_keys=False), encoding="utf-8"
    )

    size_mb = (path / PIPELINE_FILE).stat().st_size / 1e6
    log.info("saved artifact to %s (%.1f MB) - %s", path, size_mb, card.headline())
    return path


def load_artifact(path: Path | str, *, strict_versions: bool = False):
    """Load a pipeline and its card, refusing incompatible schema versions.

    Returns ``(pipeline, card)``.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(
            f"no artifact directory at {path}. Train one with "
            "`bwt train --task mi_left_right`."
        )

    card_file = path / CARD_FILE
    if not card_file.is_file():
        raise FileNotFoundError(
            f"{path} has no {CARD_FILE}; refusing to serve a model whose output "
            "classes are undocumented"
        )

    card = ModelCard.from_dict(json.loads(card_file.read_text(encoding="utf-8")))
    if card.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"artifact {path.name} uses schema version {card.schema_version}, "
            f"this build requires {ARTIFACT_SCHEMA_VERSION}. Retrain it."
        )

    current = _library_versions()
    drifted = {
        lib: (card.library_versions.get(lib), current.get(lib))
        for lib in ("sklearn", "mne", "pyriemann", "numpy")
        if card.library_versions.get(lib) not in (None, "not-installed")
        and card.library_versions.get(lib) != current.get(lib)
    }
    if drifted:
        message = "; ".join(f"{k}: trained {v[0]}, running {v[1]}"
                            for k, v in drifted.items())
        if strict_versions:
            raise RuntimeError(f"library version drift for {path.name} -- {message}")
        log.warning("library version drift for %s -- %s", path.name, message)

    model = joblib.load(path / PIPELINE_FILE)
    log.info("loaded %s (%s)", path.name, card.headline())
    return model, card


def list_artifacts(root: Path | None = None) -> list[tuple[Path, ModelCard]]:
    """Every readable artifact under ``root``, newest first."""
    root = Path(root or artifacts_dir())
    if not root.is_dir():
        return []
    found = []
    for entry in sorted(root.iterdir()):
        card_file = entry / CARD_FILE
        if not card_file.is_file():
            continue
        try:
            found.append(
                (entry, ModelCard.from_dict(
                    json.loads(card_file.read_text(encoding="utf-8"))))
            )
        except Exception as exc:
            log.warning("skipping unreadable card %s (%s)", card_file, exc)
    return sorted(found, key=lambda pair: pair[1].created_utc, reverse=True)


def resolve_artifact(
    spec: str | Path | None = None, root: Path | None = None
) -> Path:
    """Turn a user-supplied model reference into a concrete directory.

    Accepts a full path, a ``task__pipeline`` directory name, or ``None`` to
    mean "the most recently created artifact".
    """
    root = Path(root or artifacts_dir())
    if spec:
        candidate = Path(spec)
        if candidate.is_dir():
            return candidate
        candidate = root / str(spec)
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"no artifact matching {spec!r} under {root}")

    available = list_artifacts(root)
    if not available:
        raise FileNotFoundError(
            f"no trained models under {root}. Run `bwt train` first."
        )
    return available[0][0]


__all__ = [
    "CARD_FILE",
    "PIPELINE_FILE",
    "ModelCard",
    "artifact_path",
    "list_artifacts",
    "load_artifact",
    "resolve_artifact",
    "save_artifact",
]
