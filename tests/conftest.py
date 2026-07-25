"""Shared fixtures.

Everything here is synthetic and hermetic: the unit suite must run on a
checkout with no dataset present. Tests that need the real recordings are
marked ``slow`` and skip themselves when ``raw_data/`` is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from bwt.data.epochs import EpochBundle

#: The 64 EEGMMIDB channels after `mne.datasets.eegbci.standardize`, in order.
EEGBCI_CHANNELS: tuple[str, ...] = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2", "Iz",
)

SFREQ = 160.0
N_TIMES = 481  # 0.5-3.5 s inclusive at 160 Hz


def make_epochs(
    n_per_class: int = 24,
    n_channels: int = len(EEGBCI_CHANNELS),
    n_times: int = N_TIMES,
    n_classes: int = 2,
    separation: float = 1.6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic epochs with a genuine, learnable class difference.

    Each class suppresses the 8-30 Hz band over a different pair of channels,
    which is a crude but faithful caricature of event-related desynchronisation
    -- enough that a spatial-filter pipeline should score well above chance.
    """
    rng = np.random.default_rng(seed)
    n_total = n_per_class * n_classes
    t = np.arange(n_times) / SFREQ

    X = rng.standard_normal((n_total, n_channels, n_times)) * 8.0
    y = np.repeat(np.arange(n_classes), n_per_class)

    # A 10 Hz rhythm present on every channel...
    mu = np.sin(2 * np.pi * 10.0 * t) * 12.0
    X += mu[None, None, :]

    # ...attenuated over class-specific channels.
    for class_index in range(n_classes):
        channels = [
            (8 + class_index * 3) % n_channels,
            (9 + class_index * 3) % n_channels,
        ]
        rows = np.where(y == class_index)[0]
        for channel in channels:
            X[np.ix_(rows, [channel])] -= separation * mu[None, None, :]

    return X.astype(np.float32), y.astype(np.int64)


@pytest.fixture
def synthetic_bundle() -> EpochBundle:
    """A small multi-subject bundle: 6 subjects x 24 trials x 2 classes."""
    parts_X, parts_y, parts_g, parts_r = [], [], [], []
    for subject in range(1, 7):
        X, y = make_epochs(n_per_class=12, seed=subject)
        parts_X.append(X)
        parts_y.append(y)
        parts_g.append(np.full(len(y), subject, dtype=np.int64))
        parts_r.append(np.full(len(y), 4, dtype=np.int64))

    return EpochBundle(
        X=np.concatenate(parts_X),
        y=np.concatenate(parts_y),
        groups=np.concatenate(parts_g),
        runs=np.concatenate(parts_r),
        classes=("left_fist", "right_fist"),
        ch_names=EEGBCI_CHANNELS,
        sfreq=SFREQ,
        tmin=0.5,
        tmax=3.5,
        task="mi_left_right",
    )


@pytest.fixture
def trained_artifact(tmp_path, synthetic_bundle):
    """A real saved artifact directory, trained on the synthetic bundle."""
    from bwt.artifacts import ModelCard, save_artifact
    from bwt.pipelines import build_pipeline

    bundle = synthetic_bundle
    model = build_pipeline("csp_lda", sfreq=bundle.sfreq, n_classes=2)
    model.fit(bundle.X, bundle.y)

    card = ModelCard(
        name="test__csp_lda",
        task=bundle.task,
        task_description="synthetic fixture",
        pipeline="csp_lda",
        classes=list(bundle.classes),
        sfreq=bundle.sfreq,
        n_channels=bundle.n_channels,
        n_times=bundle.n_times,
        ch_names=list(bundle.ch_names),
        tmin=bundle.tmin,
        tmax=bundle.tmax,
        n_train_trials=bundle.n_trials,
        train_subjects=bundle.subjects,
        class_counts=bundle.class_counts(),
        chance_level=0.5,
        majority_level=0.5,
        evaluation={"within_subject": {"mean_accuracy": 0.9, "std_accuracy": 0.05}},
    )
    return save_artifact(model, card, path=tmp_path / "test__csp_lda")


@pytest.fixture
def synthetic_edf(tmp_path):
    """A 64-channel, 160 Hz EDF with T0/T1/T2 cue annotations.

    Written through MNE's EDF exporter so the file exercises the same reader
    path as a real recording, including channel-name standardisation.
    """
    mne = pytest.importorskip("mne")
    pytest.importorskip("edfio")

    rng = np.random.default_rng(7)
    duration = 60.0
    n_samples = int(duration * SFREQ)

    # Channel names in the raw EEGMMIDB style, which standardize() normalises.
    raw_names = [
        name.ljust(4, ".") if len(name) < 4 else name
        for name in (n.capitalize() if n.isupper() else n for n in EEGBCI_CHANNELS)
    ]

    data = rng.standard_normal((len(raw_names), n_samples)) * 10e-6
    info = mne.create_info(raw_names, SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    onsets, durations, descriptions = [], [], []
    time = 0.0
    marker = 0
    while time + 4.2 < duration:
        onsets.append(time)
        durations.append(4.1)
        descriptions.append(["T0", "T1", "T2"][marker % 3])
        marker += 1
        time += 4.2
    raw.set_annotations(
        mne.Annotations(onsets, durations, descriptions), verbose="ERROR"
    )

    path = tmp_path / "synthetic_R04.edf"
    mne.export.export_raw(str(path), raw, fmt="edf", overwrite=True, verbose="ERROR")
    return path


@pytest.fixture
def real_data_root():
    """Path to the real dataset, or skip the test if it is not present."""
    from bwt.paths import raw_data_dir

    root = raw_data_dir()
    if not root.is_dir() or not (root / "S001").is_dir():
        pytest.skip("real dataset not present")
    return root
