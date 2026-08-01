"""Continuous decoding with evidence accumulation.

A single 3-second trial is decoded at roughly 61% accuracy, which is far too
unreliable to act on directly: five consecutive decisions at that rate land on
the intended character about a tenth of the time. The standard fix is not a
better classifier but a better *decision rule* -- keep decoding overlapping
windows and accumulate the evidence until it is decisive, then commit.

This is sequential probability ratio testing. Each window contributes its
log-likelihood ratio to a running total, and a decision is emitted only when the
total crosses a confidence threshold. The cost is time rather than accuracy: a
confident decision might take two seconds or ten, depending on how clear that
person's signal is. That trade is exactly what makes a 61% decoder usable.

The accumulator is deliberately independent of any model, so it can be unit
tested against synthetic probability streams -- see ``tests/test_streaming.py``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bwt.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class Decision:
    """A committed decision, or a timeout."""

    label: str | None
    confidence: float
    n_windows: int
    elapsed_seconds: float
    posterior: dict[str, float]
    timed_out: bool = False

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "n_windows": self.n_windows,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "posterior": {k: round(v, 4) for k, v in self.posterior.items()},
            "timed_out": self.timed_out,
        }


class EvidenceAccumulator:
    """Accumulate per-window class probabilities until one class is decisive.

    Parameters
    ----------
    classes
        Ordered class names, matching the model's ``predict_proba`` columns.
    threshold
        Posterior probability required to commit. 0.9 means "commit when one
        class is 90% likely given everything seen so far".
    max_windows
        Give up after this many windows and report ``timed_out``. Without a cap
        an ambiguous stretch would stall the interface forever.
    min_windows
        Never commit before this many windows, which stops one confident-but-
        wrong window from firing immediately.
    leak
        Per-window decay applied to accumulated evidence, in [0, 1]. A value
        below 1 lets the accumulator forget old evidence, which matters when the
        user changes their mind mid-decision. 1.0 disables forgetting.
    """

    def __init__(
        self,
        classes: Sequence[str],
        *,
        threshold: float = 0.9,
        max_windows: int = 40,
        min_windows: int = 2,
        leak: float = 0.95,
    ):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie strictly between 0 and 1")
        if not 0.0 < leak <= 1.0:
            raise ValueError("leak must lie in (0, 1]")
        self.classes = list(classes)
        self.threshold = threshold
        self.max_windows = max_windows
        self.min_windows = max(1, min_windows)
        self.leak = leak
        self.reset()

    def reset(self) -> None:
        # Uniform prior, held in log space so repeated multiplication is stable.
        self.log_evidence = np.zeros(len(self.classes), dtype=np.float64)
        self.n_windows = 0
        self.started = time.time()

    @property
    def posterior(self) -> np.ndarray:
        shifted = self.log_evidence - self.log_evidence.max()
        weights = np.exp(shifted)
        return weights / weights.sum()

    def posterior_dict(self) -> dict[str, float]:
        return {c: float(p) for c, p in zip(self.classes, self.posterior, strict=True)}

    def update(self, probabilities: Sequence[float]) -> Decision | None:
        """Feed one window's class probabilities. Returns a decision or ``None``.

        A decision is returned only when the posterior crosses ``threshold`` or
        the window budget is exhausted.
        """
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape != (len(self.classes),):
            raise ValueError(
                f"expected {len(self.classes)} probabilities, got {probs.shape}"
            )
        probs = np.clip(probs, 1e-9, 1.0)

        self.log_evidence = self.log_evidence * self.leak + np.log(probs)
        self.n_windows += 1

        posterior = self.posterior
        best = int(posterior.argmax())
        confidence = float(posterior[best])

        if self.n_windows >= self.min_windows and confidence >= self.threshold:
            return Decision(
                label=self.classes[best], confidence=confidence,
                n_windows=self.n_windows,
                elapsed_seconds=time.time() - self.started,
                posterior=self.posterior_dict(),
            )
        if self.n_windows >= self.max_windows:
            return Decision(
                label=None, confidence=confidence, n_windows=self.n_windows,
                elapsed_seconds=time.time() - self.started,
                posterior=self.posterior_dict(), timed_out=True,
            )
        return None


@dataclass
class StreamWindow:
    """One window handed to the decoder by a stream source."""

    index: int
    start_sample: int
    onset_seconds: float
    data: np.ndarray  # (channels, times), microvolts


def channel_band_power(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float] = (8.0, 30.0),
) -> np.ndarray:
    """Log power per channel in one band, for a single window.

    Used to drive the 3D visualiser. This is a *display* quantity computed
    alongside the decoder, not an input to it -- the model does its own
    filtering internally, and nothing here feeds back into a prediction.
    """
    from scipy.signal import butter, sosfiltfilt

    nyquist = sfreq / 2.0
    sos = butter(4, [band[0] / nyquist, band[1] / nyquist],
                 btype="bandpass", output="sos")
    padlen = min(27, data.shape[-1] - 1)
    filtered = sosfiltfilt(sos, np.asarray(data, dtype=np.float64),
                           axis=-1, padlen=padlen)
    return np.log(np.var(filtered, axis=-1) + 1e-12)


@dataclass
class StreamEvent:
    """Everything that happened at one step of the stream."""

    window: int
    onset_seconds: float
    probabilities: dict[str, float]
    top_label: str
    posterior: dict[str, float]
    decision: Decision | None = None
    #: Per-electrode mu/beta power for this window, normalised to roughly
    #: [0, 1] against a running baseline. Present only when the caller asks for
    #: it, since it costs a filter pass per window.
    band_power: list[float] | None = None
    #: The recording itself, for a subset of channels: the samples this step
    #: advanced by, in microvolts, so successive events tile without overlap.
    #: Band power is an envelope; this is the waveform underneath it.
    raw: dict | None = None

    def to_dict(self) -> dict:
        payload = {
            "window": self.window,
            "onset_seconds": round(self.onset_seconds, 3),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "top_label": self.top_label,
            "posterior": {k: round(v, 4) for k, v in self.posterior.items()},
        }
        if self.decision is not None:
            payload["decision"] = self.decision.to_dict()
        if self.band_power is not None:
            payload["band_power"] = [round(v, 4) for v in self.band_power]
        if self.raw is not None:
            payload["raw"] = self.raw
        return payload


class EDFStream:
    """Replay a recording as a sliding window, optionally at wall-clock speed.

    ``speed`` scales real time: 1.0 replays at the rate the data was recorded,
    which is what makes a demonstration feel like a live session; ``0`` (the
    default) runs as fast as possible for evaluation.
    """

    def __init__(
        self,
        data: np.ndarray,
        sfreq: float,
        window_samples: int,
        step_samples: int,
        *,
        speed: float = 0.0,
        start_sample: int = 0,
    ):
        if data.ndim != 2:
            raise ValueError(f"expected (channels, times), got {data.shape}")
        if window_samples <= 0 or step_samples <= 0:
            raise ValueError("window and step must be positive")
        self.data = data
        self.sfreq = sfreq
        self.window_samples = window_samples
        self.step_samples = step_samples
        self.speed = speed
        self.start_sample = start_sample

    @classmethod
    def from_edf(cls, path: Path, card, *, speed: float = 0.0,
                 step_seconds: float = 0.5, picks: Sequence[int] | None = None):
        """Build a stream from a recording, honouring a model's input contract."""
        from bwt.data.epochs import read_standardised_raw

        raw = read_standardised_raw(Path(path))
        sfreq = float(raw.info["sfreq"])
        if abs(sfreq - card.sfreq) > 1e-6:
            raise ValueError(
                f"recording is {sfreq:g} Hz but the model expects "
                f"{card.sfreq:g} Hz"
            )
        data = raw.get_data(picks=picks) * 1e6
        return cls(
            data=data,
            sfreq=sfreq,
            window_samples=card.n_times,
            step_samples=max(1, round(step_seconds * sfreq)),
            speed=speed,
        )

    def __len__(self) -> int:
        span = self.data.shape[1] - self.start_sample - self.window_samples
        return max(0, span // self.step_samples + 1)

    def __iter__(self) -> Iterator[StreamWindow]:
        wall_start = time.time()
        starts = range(self.start_sample,
                       self.data.shape[1] - self.window_samples + 1,
                       self.step_samples)
        for index, start in enumerate(starts):
            onset = start / self.sfreq
            if self.speed > 0:
                # Pace against the recording's own clock so drift does not
                # accumulate over a long stream.
                target = wall_start + (onset - self.start_sample / self.sfreq) / self.speed
                delay = target - time.time()
                if delay > 0:
                    time.sleep(delay)
            yield StreamWindow(
                index=index,
                start_sample=start,
                onset_seconds=onset,
                data=self.data[:, start:start + self.window_samples],
            )


class StreamingDecoder:
    """Drive a model over a stream, accumulating evidence into decisions."""

    def __init__(
        self,
        predictor,
        *,
        threshold: float = 0.9,
        max_windows: int = 40,
        min_windows: int = 2,
        leak: float = 0.95,
    ):
        self.predictor = predictor
        self.classes = list(predictor.card.classes)
        self.accumulator = EvidenceAccumulator(
            self.classes, threshold=threshold, max_windows=max_windows,
            min_windows=min_windows, leak=leak,
        )

    def run(self, stream: EDFStream,
            on_event: Callable[[StreamEvent], None] | None = None
            ) -> list[StreamEvent]:
        """Consume a stream, returning every event produced."""
        events: list[StreamEvent] = []
        self.accumulator.reset()

        for window in stream:
            probabilities = self.predictor.model.predict_proba(
                window.data[None, ...].astype(np.float32)
            )[0]
            probs = {c: float(p) for c, p in zip(self.classes, probabilities, strict=True)}
            top = self.classes[int(np.argmax(probabilities))]

            decision = self.accumulator.update(probabilities)
            event = StreamEvent(
                window=window.index,
                onset_seconds=window.onset_seconds,
                probabilities=probs,
                top_label=top,
                posterior=self.accumulator.posterior_dict(),
                decision=decision,
            )

            if decision is not None:
                self.accumulator.reset()

            events.append(event)
            if on_event is not None:
                on_event(event)

        return events


def simulate_accumulation_throughput(
    accuracy: float, n_classes: int, seconds_per_window: float,
    *, threshold: float = 0.9, max_windows: int = 40,
    n_trials: int = 2000, random_state: int = 0,
) -> dict:
    """Monte-Carlo the accuracy/speed trade of evidence accumulation.

    .. warning::
       This simulation assumes each window is an **independent** draw. That
       assumption holds reasonably well when evidence is accumulated over
       *repeated trials* -- the user imagines the same movement again, several
       seconds apart -- but it is badly wrong for *overlapping* sliding windows
       within a single trial, which share most of their samples and therefore
       share their errors. Treat the numbers here as an upper bound, and use
       :func:`evaluate_accumulation` for a figure measured on real predictions.

    Returns the decision accuracy, how often a decision is reached at all, and
    the time it takes.
    """
    rng = np.random.default_rng(random_state)
    classes = [f"c{i}" for i in range(n_classes)]
    off = (1.0 - accuracy) / max(1, n_classes - 1)

    correct = 0
    committed = 0
    windows_used: list[int] = []

    for _ in range(n_trials):
        truth = int(rng.integers(n_classes))
        accumulator = EvidenceAccumulator(
            classes, threshold=threshold, max_windows=max_windows, min_windows=1,
            leak=1.0,
        )
        while True:
            drawn = truth if rng.random() < accuracy else int(rng.integers(n_classes))
            probs = np.full(n_classes, off)
            probs[drawn] = accuracy
            decision = accumulator.update(probs / probs.sum())
            if decision is not None:
                windows_used.append(decision.n_windows)
                if decision.label is not None:
                    committed += 1
                    if classes.index(decision.label) == truth:
                        correct += 1
                break

    mean_windows = float(np.mean(windows_used)) if windows_used else 0.0
    decision_accuracy = correct / committed if committed else 0.0
    seconds = mean_windows * seconds_per_window
    return {
        "per_window_accuracy": accuracy,
        "threshold": threshold,
        "decision_accuracy": decision_accuracy,
        "commit_rate": committed / n_trials,
        "mean_windows_per_decision": mean_windows,
        "seconds_per_decision": seconds,
        "bits_per_minute": (
            math.log2(n_classes) * 60.0 / seconds if seconds > 0 else 0.0
        ),
    }


def evaluate_accumulation(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    classes: Sequence[str],
    *,
    threshold: float = 0.9,
    max_windows: int = 20,
    n_sequences: int = 400,
    random_state: int = 0,
) -> dict:
    """Measure accumulation on *real* model output, not a simulation.

    Takes held-out per-trial probabilities from an actual cross-validated model
    and builds synthetic repetition sequences: for a randomly chosen subject and
    intended class, repeatedly draw one of that subject's real trials of that
    class and feed its real probability vector to the accumulator. Errors
    therefore carry the true correlation structure of that subject and model,
    including the fact that some subjects are simply undecodable.

    Trials are sampled without replacement within a sequence where possible, so
    the same trial does not supply the same evidence twice.
    """
    rng = np.random.default_rng(random_state)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true)
    groups = np.asarray(groups)
    n_classes = len(classes)

    index: dict[tuple[int, int], np.ndarray] = {}
    for subject in np.unique(groups):
        for label in range(n_classes):
            rows = np.where((groups == subject) & (y_true == label))[0]
            if len(rows):
                index[(int(subject), label)] = rows

    usable = [
        s for s in np.unique(groups)
        if all((int(s), c) in index for c in range(n_classes))
    ]
    if not usable:
        raise ValueError("no subject has trials of every class")

    correct = committed = 0
    windows_used: list[int] = []
    per_subject: dict[int, list[bool]] = {}

    for _ in range(n_sequences):
        subject = int(rng.choice(usable))
        truth = int(rng.integers(n_classes))
        pool = list(index[(subject, truth)])
        rng.shuffle(pool)

        accumulator = EvidenceAccumulator(
            classes, threshold=threshold, max_windows=max_windows,
            min_windows=1, leak=1.0,
        )
        draw = 0
        while True:
            if draw >= len(pool):  # exhausted: reshuffle and reuse
                rng.shuffle(pool)
                draw = 0
            decision = accumulator.update(probabilities[pool[draw]])
            draw += 1
            if decision is not None:
                windows_used.append(decision.n_windows)
                if decision.label is not None:
                    committed += 1
                    hit = classes.index(decision.label) == truth
                    correct += hit
                    per_subject.setdefault(subject, []).append(bool(hit))
                break

    subject_means = [float(np.mean(v)) for v in per_subject.values() if v]
    return {
        "threshold": threshold,
        "decision_accuracy": correct / committed if committed else 0.0,
        "commit_rate": committed / n_sequences,
        "mean_trials_per_decision": float(np.mean(windows_used)) if windows_used else 0.0,
        "n_sequences": n_sequences,
        "per_subject_accuracy_mean": float(np.mean(subject_means)) if subject_means else 0.0,
        "per_subject_accuracy_std": (
            float(np.std(subject_means, ddof=1)) if len(subject_means) > 1 else 0.0
        ),
        "n_subjects": len(subject_means),
    }


class BandPowerNormaliser:
    """Map raw log band power onto roughly [0, 1] for display.

    A running mean and spread per channel, updated as the stream plays. Without
    this the visualiser would be dominated by the fact that some electrodes
    simply sit at higher impedance than others, and every head would look the
    same regardless of what the subject was doing.
    """

    def __init__(self, n_channels: int, momentum: float = 0.05):
        self.mean = np.zeros(n_channels)
        self.var = np.ones(n_channels)
        self.momentum = momentum
        self.seen = 0

    def __call__(self, power: np.ndarray) -> np.ndarray:
        power = np.asarray(power, dtype=np.float64)
        if self.seen == 0:
            self.mean = power.copy()
        else:
            delta = power - self.mean
            self.mean += self.momentum * delta
            self.var = (1 - self.momentum) * self.var + self.momentum * delta ** 2
        self.seen += 1

        spread = np.sqrt(np.maximum(self.var, 1e-9))
        z = (power - self.mean) / spread
        # Squash to [0, 1]. +-1.5 sigma spans the full range: EEG band power
        # varies over a narrow band once each electrode's own baseline is
        # removed, so a wider window would leave every contact sitting near
        # mid-grey and the display would carry no information.
        return np.clip(0.5 + z / 3.0, 0.0, 1.0)


__all__ = [
    "BandPowerNormaliser",
    "Decision",
    "EDFStream",
    "EvidenceAccumulator",
    "StreamEvent",
    "StreamWindow",
    "StreamingDecoder",
    "channel_band_power",
    "evaluate_accumulation",
    "simulate_accumulation_throughput",
]
