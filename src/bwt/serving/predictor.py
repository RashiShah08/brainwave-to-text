"""Inference: EDF in, decoded commands out.

The single most important property of this module is that it does **not**
reimplement feature extraction. It reproduces the epoching contract recorded in
the model card and hands the resulting array straight to the fitted pipeline,
which carries its own filtering and spatial filters. The previous version of
this project had a separate hand-written feature extractor in the web app that
had silently drifted from the training one -- different preprocessing, a fixed
first-second window, and no way to notice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bwt.artifacts import ModelCard, load_artifact, resolve_artifact
from bwt.data.epochs import read_standardised_raw
from bwt.logging_utils import get_logger
from bwt.metrics import information_transfer_rate

log = get_logger(__name__)


class InputContractError(ValueError):
    """Raised when input does not match what the model was trained on.

    Always preferred over silently coercing the data: a wrong channel order or
    sampling rate produces confident, meaningless predictions.
    """


@dataclass
class Prediction:
    index: int
    label: str
    confidence: float
    probabilities: dict[str, float]
    onset_seconds: float | None = None
    source: str = "epoch"

    def to_dict(self) -> dict:
        payload = {
            "index": self.index,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "source": self.source,
        }
        if self.onset_seconds is not None:
            payload["onset_seconds"] = round(self.onset_seconds, 3)
        return payload


@dataclass
class PredictionBatch:
    predictions: list[Prediction]
    model: str
    task: str
    classes: list[str]
    epoching: str
    warnings: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.predictions)

    def majority_label(self) -> str | None:
        if not self.predictions:
            return None
        counts: dict[str, int] = {}
        for prediction in self.predictions:
            counts[prediction.label] = counts.get(prediction.label, 0) + 1
        return max(counts, key=counts.__getitem__)

    def mean_confidence(self) -> float:
        if not self.predictions:
            return 0.0
        return float(np.mean([p.confidence for p in self.predictions]))

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "task": self.task,
            "classes": self.classes,
            "epoching": self.epoching,
            "n_epochs": self.n,
            "majority_label": self.majority_label(),
            "mean_confidence": round(self.mean_confidence(), 4),
            "predictions": [p.to_dict() for p in self.predictions],
            "warnings": self.warnings,
        }


class Predictor:
    """A loaded model plus the epoching contract it requires."""

    def __init__(self, model, card: ModelCard, name: str,
                 max_epochs: int = 256):
        self.model = model
        self.card = card
        self.name = name
        self.max_epochs = max_epochs

    # -- construction ----------------------------------------------------- #

    @classmethod
    def load(cls, spec: str | Path | None = None, *,
             strict_versions: bool = False, max_epochs: int = 256) -> Predictor:
        path = resolve_artifact(spec)
        model, card = load_artifact(path, strict_versions=strict_versions)
        return cls(model, card, name=path.name, max_epochs=max_epochs)

    # -- contract enforcement --------------------------------------------- #

    @property
    def n_times(self) -> int:
        return self.card.n_times

    def _align_channels(self, ch_names: Sequence[str]) -> list[int]:
        """Return picks reordering ``ch_names`` into the model's channel order."""
        expected = list(self.card.ch_names)
        lookup = {name.upper(): i for i, name in enumerate(ch_names)}
        missing = [name for name in expected if name.upper() not in lookup]
        if missing:
            raise InputContractError(
                f"recording is missing {len(missing)} channel(s) the model "
                f"requires, e.g. {missing[:5]}. Expected the 64-channel "
                "EEGMMIDB montage."
            )
        return [lookup[name.upper()] for name in expected]

    def validate_array(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[None, ...]
        if X.ndim != 3:
            raise InputContractError(
                f"expected (trials, channels, times), got shape {X.shape}"
            )
        if X.shape[1] != self.card.n_channels:
            raise InputContractError(
                f"model expects {self.card.n_channels} channels, got {X.shape[1]}"
            )
        if X.shape[2] != self.card.n_times:
            raise InputContractError(
                f"model expects {self.card.n_times} samples per epoch "
                f"({self.card.tmin}-{self.card.tmax}s at {self.card.sfreq:g} Hz), "
                f"got {X.shape[2]}"
            )
        return X

    # -- prediction ------------------------------------------------------- #

    def predict_array(
        self, X: np.ndarray, *, onsets: Sequence[float] | None = None,
        source: str = "epoch",
    ) -> list[Prediction]:
        X = self.validate_array(X)
        classes = list(self.card.classes)

        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(X))
            indices = proba.argmax(axis=1)
        else:  # pragma: no cover - every registry pipeline exposes predict_proba
            indices = np.asarray(self.model.predict(X))
            proba = np.eye(len(classes))[indices]

        out: list[Prediction] = []
        for position, class_index in enumerate(indices):
            row = proba[position]
            out.append(
                Prediction(
                    index=position,
                    label=classes[int(class_index)],
                    confidence=float(row[int(class_index)]),
                    probabilities={c: float(v) for c, v in zip(classes, row, strict=True)},
                    onset_seconds=(None if onsets is None else float(onsets[position])),
                    source=source,
                )
            )
        return out

    # -- EDF entry point --------------------------------------------------- #

    def epochs_from_edf(self, path: Path) -> tuple[np.ndarray, list[float], str, list[str]]:
        """Cut epochs from an uploaded recording under the model's contract.

        Two modes, chosen automatically:

        * **cue-locked** -- the file carries ``T1``/``T2`` annotations, so we cut
          exactly where the experiment cued a movement. This is what the model
          was trained on and gives the trustworthy answer.
        * **sliding-window** -- no usable annotations, so the recording is
          chopped into consecutive windows of the model's length. Predictions
          are then per-window guesses with no cue to align to, which is reported
          back to the caller rather than hidden.
        """
        warnings_out: list[str] = []
        raw = read_standardised_raw(path)

        sfreq = float(raw.info["sfreq"])
        if not math.isclose(sfreq, self.card.sfreq, rel_tol=1e-6):
            raise InputContractError(
                f"recording is {sfreq:g} Hz but the model was trained at "
                f"{self.card.sfreq:g} Hz. Resampling here would change the "
                "spectral content the spatial filters were fitted to; upload a "
                f"{self.card.sfreq:g} Hz recording instead."
            )

        picks = self._align_channels(raw.ch_names)
        data = raw.get_data(picks=picks) * 1e6  # -> microvolts, model's units

        window = self.card.n_times
        offset = round(self.card.tmin * sfreq)

        cue_onsets = [
            float(onset)
            for onset, description in zip(raw.annotations.onset,
                                          raw.annotations.description,
                                          strict=True)
            if str(description).strip() in {"T1", "T2"}
        ]

        starts: list[int]
        if cue_onsets:
            mode = "cue_locked"
            starts = [round(o * sfreq) + offset for o in cue_onsets]
            onsets = list(cue_onsets)
        else:
            mode = "sliding_window"
            warnings_out.append(
                "No T1/T2 cue annotations found. The recording was split into "
                "consecutive fixed windows, so each prediction is an unaligned "
                "guess rather than a decoded trial."
            )
            starts = list(range(0, data.shape[1] - window + 1, window))
            onsets = [s / sfreq for s in starts]

        kept: list[np.ndarray] = []
        kept_onsets: list[float] = []
        for start, onset in zip(starts, onsets, strict=True):
            if start < 0 or start + window > data.shape[1]:
                continue
            kept.append(data[:, start:start + window])
            kept_onsets.append(onset)

        if not kept:
            raise InputContractError(
                f"recording is too short: need at least {window} samples "
                f"({window / sfreq:.2f}s) after the cue offset, got "
                f"{data.shape[1]} samples total"
            )

        if len(kept) > self.max_epochs:
            warnings_out.append(
                f"recording yielded {len(kept)} epochs; only the first "
                f"{self.max_epochs} were decoded"
            )
            kept = kept[: self.max_epochs]
            kept_onsets = kept_onsets[: self.max_epochs]

        return (np.stack(kept).astype(np.float32), kept_onsets, mode, warnings_out)

    def predict_edf(self, path: Path) -> PredictionBatch:
        X, onsets, mode, warnings_out = self.epochs_from_edf(Path(path))

        if self.card.requires_batch_recentering and len(X) < 8:
            warnings_out.append(
                "This model recenters each recording by its own covariance mean "
                f"but only {len(X)} epoch(s) were available; it fell back to the "
                "training reference, which reduces accuracy."
            )

        predictions = self.predict_array(X, onsets=onsets, source=mode)

        batch = PredictionBatch(
            predictions=predictions,
            model=self.name,
            task=self.card.task,
            classes=list(self.card.classes),
            epoching=mode,
            warnings=warnings_out,
        )
        return batch

    # -- streaming ---------------------------------------------------------- #

    def stream_from_edf(self, path: Path, *, speed: float = 0.0,
                        step_seconds: float = 0.5):
        """Build a sliding-window stream over a recording, in the model's channel order."""
        from bwt.data.epochs import read_standardised_raw
        from bwt.streaming import EDFStream

        raw = read_standardised_raw(Path(path))
        picks = self._align_channels(raw.ch_names)
        return EDFStream.from_edf(
            Path(path), self.card, speed=speed, step_seconds=step_seconds,
            picks=picks,
        )

    def streaming_decoder(self, *, threshold: float = 0.9,
                          max_windows: int = 40, min_windows: int = 2,
                          leak: float = 0.95):
        from bwt.streaming import StreamingDecoder

        return StreamingDecoder(
            self, threshold=threshold, max_windows=max_windows,
            min_windows=min_windows, leak=leak,
        )

    # -- the cue behind each class ----------------------------------------- #

    @property
    def cue_positions(self) -> dict[str, str]:
        """Where the target appeared on screen for each class, when known.

        The subject was not asked to produce a class name -- a target appeared
        somewhere on the screen and they responded to it, which is what the
        decoder is really recovering. BCI2000 records exactly this as its
        ``TargetCode`` state variable. Empty for datasets whose protocol this
        codebase does not describe.
        """
        try:
            from bwt.data.physionet import TASKS
        except Exception:
            return {}
        task = TASKS.get(self.card.task)
        if task is None:
            return {}
        return {c: p for c, p in task.cue_positions.items() if c in self.card.classes}

    @property
    def cue_axes(self) -> list[str]:
        from bwt.data.physionet import CUE_AXIS

        seen = {CUE_AXIS[p] for p in self.cue_positions.values()}
        return [a for a in ("horizontal", "vertical") if a in seen]

    # -- reporting --------------------------------------------------------- #

    def performance_note(self) -> dict:
        """What this model actually achieves, for display alongside a result."""
        within = self.card.evaluation.get("within_subject", {})
        cross = self.card.evaluation.get("cross_subject", {})
        accuracy = within.get("mean_accuracy") or cross.get("mean_accuracy") or 0.0
        n_classes = max(2, len(self.card.classes))
        trial_seconds = max(0.1, self.card.tmax - self.card.tmin)
        return {
            "within_subject_accuracy": within.get("mean_accuracy"),
            "within_subject_std": within.get("std_accuracy"),
            "cross_subject_accuracy": cross.get("mean_accuracy"),
            "chance_level": self.card.chance_level,
            "itr_bits_per_minute": round(
                information_transfer_rate(accuracy, n_classes, trial_seconds), 2
            ),
        }


__all__ = ["InputContractError", "Prediction", "PredictionBatch", "Predictor"]
