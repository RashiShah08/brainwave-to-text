"""Builders shared by the adversarial, property-based and browser suites.

Everything here is synthetic and hermetic, so the suites that use it run on a
checkout with no dataset and no trained model on disk:

* :func:`write_edf` writes a real EDF through MNE's exporter -- the same reader
  path an uploaded recording takes -- with any sampling rate, channel set,
  annotation list or pathological signal a test needs.
* :func:`build_artifact` trains a registry pipeline on synthetic epochs and
  saves it with a model card, so the serving layer loads it exactly as it loads
  a production artifact.
* :class:`LiveServer` runs the application under waitress on an ephemeral port,
  which is what the browser suite and the concurrency tests talk to. The Flask
  test client is single-threaded and cannot reproduce either.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from conftest import EEGBCI_CHANNELS, SFREQ, make_epochs

#: Seconds between cue onsets, and how long each cued trial lasts, matching
#: the EEGMMIDB protocol closely enough for epoching to behave identically.
TRIAL_SPACING = 4.2
TRIAL_SECONDS = 4.1


def raw_style_names(names: Sequence[str]) -> list[str]:
    """Channel names as EEGMMIDB files spell them (``Fc5.``, ``Cz..``)."""
    styled = (n.capitalize() if n.isupper() else n for n in names)
    return [name.ljust(4, ".") if len(name) < 4 else name for name in styled]


def alternating_cues(duration: float, *, include_rest: bool = True,
                     start: float = 0.0) -> list[tuple[float, str]]:
    """``T0``/``T1``/``T2`` cues every :data:`TRIAL_SPACING` seconds."""
    order = ["T0", "T1", "T2"] if include_rest else ["T1", "T2"]
    cues, onset, i = [], start, 0
    while onset + TRIAL_SECONDS + 0.1 < duration:
        cues.append((round(onset, 3), order[i % len(order)]))
        onset += TRIAL_SPACING
        i += 1
    return cues


def write_edf(
    path: Path,
    *,
    duration: float = 60.0,
    sfreq: float = SFREQ,
    ch_names: Sequence[str] = EEGBCI_CHANNELS,
    cues: Sequence[tuple[float, str]] | None = None,
    signal: str = "learnable",
    amplitude_uv: float = 8.0,
    separation: float = 1.6,
    seed: int = 7,
) -> Path:
    """Write a synthetic recording and return its path.

    ``signal``:
        ``"learnable"`` -- noise plus a 10 Hz rhythm that is suppressed over
        class-specific channels during each ``T1``/``T2`` trial, the same
        caricature of desynchronisation :func:`conftest.make_epochs` trains on,
        so a model fitted there decodes this file well above chance.
        ``"noise"`` -- Gaussian noise only.
        ``"flat"`` -- every sample exactly zero.
    ``cues``:
        ``(onset_seconds, description)`` pairs, or ``None`` for
        :func:`alternating_cues`. Pass ``[]`` for a recording with no
        annotations at all.
    """
    import mne

    rng = np.random.default_rng(seed)
    n_samples = round(duration * sfreq)
    names = list(ch_names)
    n_channels = len(names)
    cues = alternating_cues(duration) if cues is None else list(cues)

    if signal == "flat":
        data_uv = np.zeros((n_channels, n_samples))
    else:
        data_uv = rng.standard_normal((n_channels, n_samples)) * amplitude_uv
        if signal == "learnable":
            t = np.arange(n_samples) / sfreq
            mu = np.sin(2 * np.pi * 10.0 * t) * 12.0
            data_uv += mu[None, :]
            index = {name.upper(): i for i, name in enumerate(names)}
            for onset, description in cues:
                if description not in {"T1", "T2"}:
                    continue
                class_index = 0 if description == "T1" else 1
                a = round(onset * sfreq)
                b = min(n_samples, a + round(TRIAL_SECONDS * sfreq))
                # Attenuate the same two channels make_epochs attenuates,
                # located by name so a reordered montage still carries the
                # signal on the right electrodes.
                for k in (8 + class_index * 3, 9 + class_index * 3):
                    target = EEGBCI_CHANNELS[k % len(EEGBCI_CHANNELS)].upper()
                    if target in index:
                        data_uv[index[target], a:b] -= separation * mu[a:b]
        elif signal != "noise":
            raise ValueError(f"unknown signal {signal!r}")

    info = mne.create_info(raw_style_names(names), sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose="ERROR")
    if cues:
        onsets, descriptions = zip(*cues, strict=True)
        raw.set_annotations(
            mne.Annotations(list(onsets), [TRIAL_SECONDS] * len(onsets),
                            list(descriptions)),
            verbose="ERROR",
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(str(path), raw, fmt="edf", overwrite=True,
                          verbose="ERROR")
    return path


def cue_labels(cues: Sequence[tuple[float, str]], classes: Sequence[str]) -> list[str]:
    """The class each ``T1``/``T2`` cue stands for, in order."""
    return [classes[0] if d == "T1" else classes[1]
            for _, d in cues if d in {"T1", "T2"}]


def build_artifact(
    path: Path,
    *,
    pipeline: str = "csp_lda",
    task: str = "mi_left_right",
    n_subjects: int = 6,
    evaluation: dict | None = None,
) -> Path:
    """Train ``pipeline`` on synthetic epochs for ``task`` and save it."""
    from bwt.artifacts import ModelCard, save_artifact
    from bwt.data.physionet import TASKS
    from bwt.pipelines import TRANSDUCTIVE_PIPELINES, build_pipeline

    classes = list(TASKS[task].classes)
    parts_X, parts_y, subjects = [], [], []
    for subject in range(1, n_subjects + 1):
        X, y = make_epochs(n_per_class=12, n_classes=len(classes), seed=subject)
        parts_X.append(X)
        parts_y.append(y)
        subjects.append(subject)
    X = np.concatenate(parts_X)
    y = np.concatenate(parts_y)

    model = build_pipeline(pipeline, sfreq=SFREQ, n_classes=len(classes))
    model.fit(X, y)

    chance = 1.0 / len(classes)
    card = ModelCard(
        name=f"{task}__{pipeline}",
        task=task,
        task_description=TASKS[task].description,
        pipeline=pipeline,
        classes=classes,
        sfreq=SFREQ,
        n_channels=X.shape[1],
        n_times=X.shape[2],
        ch_names=list(EEGBCI_CHANNELS),
        tmin=0.5,
        tmax=3.5,
        n_train_trials=len(X),
        train_subjects=subjects,
        class_counts={c: int((y == i).sum()) for i, c in enumerate(classes)},
        chance_level=chance,
        majority_level=chance,
        evaluation=evaluation if evaluation is not None else {
            "within_subject": {"mean_accuracy": 0.9, "std_accuracy": 0.05},
            "cross_subject": {"mean_accuracy": 0.8, "std_accuracy": 0.02},
        },
        requires_batch_recentering=pipeline in TRANSDUCTIVE_PIPELINES,
    )
    return save_artifact(model, card, path=Path(path))


def make_app(artifact: Path, *, max_upload_mb: int = 64, max_epochs: int = 256,
             threads: int = 4, frame_ancestors: str | None = None):
    """The production application factory, bound to a synthetic artifact.

    ``threads`` is the configured worker pool, which sizes the paced-replay
    budget; it does not start a server.
    """
    from bwt.config import Config
    from bwt.serving.app import create_app
    from bwt.serving.predictor import Predictor

    config = Config()
    config.serve.max_upload_mb = max_upload_mb
    config.serve.max_epochs_per_request = max_epochs
    config.serve.threads = threads
    if frame_ancestors is not None:
        config.serve.frame_ancestors = frame_ancestors
    predictor = Predictor.load(artifact, max_epochs=max_epochs)
    return create_app(config, predictor=predictor)


class LiveServer:
    """The application under waitress, on an ephemeral localhost port."""

    def __init__(self, app, *, threads: int = 4, options: dict | None = None):
        """``options`` are further waitress adjustments, e.g. the production
        :func:`bwt.serving.app.waitress_options` minus host and port."""
        from waitress.server import create_server

        #: The Flask application itself; waitress wraps it, so tests that need
        #: its state (the replay budget, the config) reach it here.
        self.app = app
        extra = {k: v for k, v in (options or {}).items()
                 if k not in {"host", "port", "threads"}}
        extra.setdefault("clear_untrusted_proxy_headers", True)
        self._server = create_server(app, host="127.0.0.1", port=0,
                                     threads=threads, **extra)
        self.port = self._server.effective_port
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        # The dispatcher raises on close from another thread; that is shutdown,
        # not a failure.
        with contextlib.suppress(Exception):
            self._server.run()

    def __enter__(self) -> LiveServer:
        import requests

        self._thread.start()
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                if requests.get(self.url + "/healthz", timeout=2).ok:
                    return self
            except requests.RequestException:
                time.sleep(0.05)
        raise RuntimeError("live server did not come up")

    def __exit__(self, *exc) -> None:
        self._server.close()
        self._thread.join(timeout=5)


__all__ = [
    "LiveServer",
    "alternating_cues",
    "build_artifact",
    "cue_labels",
    "make_app",
    "raw_style_names",
    "write_edf",
]
