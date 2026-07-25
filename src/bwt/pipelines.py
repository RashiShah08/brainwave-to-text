"""Model registry: feature extraction + classifier, as scikit-learn pipelines.

Every pipeline here accepts the *same* input that :class:`bwt.data.EpochBundle`
carries -- a ``(n_trials, n_channels, n_times)`` float array in microvolts at
160 Hz -- and does all of its own filtering. That is deliberate: the fitted
pipeline is the single artifact shipped to production, so there is no separate
"inference feature extractor" that can drift out of step with training.

The methods implemented are the established baselines for sensorimotor-rhythm
decoding:

``csp_lda``
    Common Spatial Patterns on the 8-30 Hz mu/beta band into shrinkage LDA.
    The reference method for two-class motor imagery.
``fbcsp_lda``
    Filter-Bank CSP: CSP within each of several narrow sub-bands, then mutual-
    information feature selection. Ang et al. (2008); the winner of BCI
    Competition IV datasets IIa/IIb.
``riemann_ts``
    Covariance matrices projected to the Riemannian tangent space, then
    logistic regression. Barachant et al. (2012).
``fb_riemann_ts``
    Tangent-space features computed per sub-band and concatenated. Usually the
    strongest of the four, and the project default.
``bandpower_rf``
    Per-channel log band power into a random forest. No spatial filtering, so
    it is weak, but it is fully interpretable and serves as a sanity floor.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#: Sub-bands used by the filter-bank methods, in Hz. Four-hertz bands spanning
#: theta through low gamma, which is the standard FBCSP configuration.
DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
    (28.0, 32.0),
    (32.0, 38.0),
)

#: The mu + beta band, where sensorimotor rhythm modulation lives.
SENSORIMOTOR_BAND = (8.0, 30.0)


# --------------------------------------------------------------------------- #
# Signal-processing transformers
# --------------------------------------------------------------------------- #


def _design(l_freq: float, h_freq: float, sfreq: float, order: int):
    nyq = sfreq / 2.0
    low, high = l_freq / nyq, h_freq / nyq
    if not 0 < low < high < 1:
        raise ValueError(
            f"band {l_freq}-{h_freq} Hz is not valid for a {sfreq} Hz signal "
            f"(Nyquist {nyq} Hz)"
        )
    return butter(order, [low, high], btype="bandpass", output="sos")


class BandpassFilter(BaseEstimator, TransformerMixin):
    """Zero-phase Butterworth band-pass applied along the time axis.

    Stateless: ``fit`` only records the input shape so that ``transform`` can
    reject mis-shaped data at serving time.
    """

    def __init__(self, l_freq: float = 8.0, h_freq: float = 30.0,
                 sfreq: float = 160.0, order: int = 4):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.sfreq = sfreq
        self.order = order

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"expected (trials, channels, times), got {X.shape}")
        self.n_channels_ = X.shape[1]
        self.n_times_ = X.shape[2]
        self.sos_ = _design(self.l_freq, self.h_freq, self.sfreq, self.order)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"expected (trials, channels, times), got {X.shape}")
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                f"expected {self.n_channels_} channels, got {X.shape[1]}"
            )
        # padlen must stay below the signal length for very short epochs.
        padlen = min(3 * (2 * self.order + 1), X.shape[-1] - 1)
        return sosfiltfilt(self.sos_, X, axis=-1, padlen=padlen)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.three_d_array = True
        tags.input_tags.two_d_array = False
        return tags


class LogVariance(BaseEstimator, TransformerMixin):
    """Log-variance of each channel: the classic band-power feature."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.log(np.var(X, axis=-1) + 1e-12)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.three_d_array = True
        tags.input_tags.two_d_array = False
        return tags


class FilterBank(BaseEstimator, TransformerMixin):
    """Apply a per-band feature extractor and concatenate the results.

    Bands are processed one at a time and the filtered signal is released
    immediately, so peak memory stays at one band rather than ``len(bands)``
    copies of the epoch tensor -- which matters at 105 subjects.
    """

    def __init__(self, make_estimator: Callable[[], BaseEstimator],
                 bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
                 sfreq: float = 160.0, order: int = 4):
        self.make_estimator = make_estimator
        self.bands = bands
        self.sfreq = sfreq
        self.order = order

    def _filter(self, X, band):
        sos = _design(band[0], band[1], self.sfreq, self.order)
        padlen = min(3 * (2 * self.order + 1), X.shape[-1] - 1)
        return sosfiltfilt(sos, np.asarray(X, dtype=np.float64),
                           axis=-1, padlen=padlen)

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"expected (trials, channels, times), got {X.shape}")
        self.n_channels_ = X.shape[1]
        self.estimators_ = []
        for band in self.bands:
            estimator = self.make_estimator()
            estimator.fit(self._filter(X, band), y)
            self.estimators_.append(estimator)
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                f"expected {self.n_channels_} channels, got {X.shape[1]}"
            )
        blocks = [
            np.asarray(est.transform(self._filter(X, band)))
            for est, band in zip(self.estimators_, self.bands)
        ]
        return np.hstack([b.reshape(len(X), -1) for b in blocks])

    def fit_transform(self, X, y=None, **fit_params):
        # Override the mixin default so each band is filtered once, not twice.
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"expected (trials, channels, times), got {X.shape}")
        self.n_channels_ = X.shape[1]
        self.estimators_ = []
        blocks = []
        for band in self.bands:
            filtered = self._filter(X, band)
            estimator = self.make_estimator()
            block = np.asarray(estimator.fit_transform(filtered, y))
            self.estimators_.append(estimator)
            blocks.append(block.reshape(len(X), -1))
        return np.hstack(blocks)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.three_d_array = True
        tags.input_tags.two_d_array = False
        return tags


# --------------------------------------------------------------------------- #
# Component factories
# --------------------------------------------------------------------------- #


def _make_csp(n_components: int = 4):
    from mne.decoding import CSP

    # Ledoit-Wolf shrinkage keeps the covariance estimate well-conditioned when
    # a single subject contributes only ~45 trials against 64 channels.
    return CSP(
        n_components=n_components,
        reg="ledoit_wolf",
        log=True,
        norm_trace=False,
        transform_into="average_power",
    )


class RiemannianRecenter(BaseEstimator, TransformerMixin):
    """Whiten a batch of covariance matrices by their own Riemannian mean.

    Most of the between-subject variance in EEG covariance is a constant offset
    per subject (electrode impedance, head geometry, session state) rather than
    anything task-related. Mapping each recording so that its own mean
    covariance sits at the identity removes that offset and is the standard fix
    for cross-subject transfer -- Zanini et al. (2018), "Transfer Learning: A
    Riemannian Geometry Framework".

    **This transform is transductive**: it uses statistics of the batch being
    transformed, not only of the training set. That is safe here because the
    reference mean is computed without labels, but it does mean predictions on
    a batch of epochs from one recording are not independent of each other. A
    batch smaller than ``min_batch`` falls back to the reference mean learned at
    fit time, so single-epoch prediction still works and simply forgoes the
    adaptation. Any artifact using this step records
    ``requires_batch_recentering`` in its model card.
    """

    #: Reference mean used for recentering. ``logeuclid`` is closed-form; the
    #: affine-invariant ``riemann`` mean is iterative and, on batches of a few
    #: thousand 64x64 matrices, routinely fails to converge inside its iteration
    #: budget -- costing tens of minutes per fold for no measurable accuracy
    #: gain. Log-Euclidean recentering is the standard practical substitute.
    def __init__(self, min_batch: int = 8, metric: str = "logeuclid"):
        self.min_batch = min_batch
        self.metric = metric

    def _mean(self, X):
        from pyriemann.utils.mean import mean_covariance

        return mean_covariance(X, metric=self.metric)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.reference_ = self._mean(X)
        return self

    def _whiten(self, X, reference):
        from pyriemann.utils.base import invsqrtm

        root = invsqrtm(reference)
        return root @ X @ root

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        reference = self._mean(X) if len(X) >= self.min_batch else self.reference_
        return self._whiten(X, reference)


def _make_tangent_space(recenter: bool = False):
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    steps = [("cov", Covariances(estimator="oas"))]
    if recenter:
        steps.append(("recenter", RiemannianRecenter()))
    steps.append(("ts", TangentSpace(metric="riemann")))
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def _csp_lda(sfreq: float, n_classes: int, random_state: int) -> Pipeline:
    return Pipeline([
        ("bandpass", BandpassFilter(*SENSORIMOTOR_BAND, sfreq=sfreq)),
        ("csp", _make_csp(n_components=8)),
        ("scale", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")),
    ])


def _fbcsp_lda(sfreq: float, n_classes: int, random_state: int) -> Pipeline:
    n_features = 4 * len(DEFAULT_BANDS)
    return Pipeline([
        ("fbcsp", FilterBank(lambda: _make_csp(n_components=4), sfreq=sfreq)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(mutual_info_classif, k=min(24, n_features))),
        ("clf", LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")),
    ])


def _riemann_ts(sfreq: float, n_classes: int, random_state: int) -> Pipeline:
    return Pipeline([
        ("bandpass", BandpassFilter(*SENSORIMOTOR_BAND, sfreq=sfreq)),
        ("ts", _make_tangent_space()),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000,
                                   random_state=random_state)),
    ])


def _fb_riemann_ts(sfreq: float, n_classes: int, random_state: int) -> Pipeline:
    # Narrower bank than FBCSP: tangent-space features are 64*65/2 = 2080 wide
    # per band, so four bands already give 8320 features. L2 keeps that in hand.
    bands = ((8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 32.0))
    return Pipeline([
        ("fbts", FilterBank(_make_tangent_space, bands=bands, sfreq=sfreq)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=3000,
                                   random_state=random_state)),
    ])


def _riemann_ts_aligned(sfreq: float, n_classes: int, random_state: int) -> Pipeline:
    return Pipeline([
        ("bandpass", BandpassFilter(*SENSORIMOTOR_BAND, sfreq=sfreq)),
        ("ts", _make_tangent_space(recenter=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000,
                                   random_state=random_state)),
    ])


def _deep(architecture: str, **overrides):
    """Factory for a deep pipeline: broadband filter, then the network.

    The 4-38 Hz band is the standard preprocessing for these architectures. It
    is applied by the same `BandpassFilter` the classical pipelines use, so the
    deep models inherit the identical train/serve guarantee.
    """

    def build(sfreq: float, n_classes: int, random_state: int) -> Pipeline:
        from bwt.deep import TorchClassifier

        params = dict(
            architecture=architecture, sfreq=sfreq, random_state=random_state,
        )
        params.update(overrides)
        return Pipeline([
            ("bandpass", BandpassFilter(4.0, 38.0, sfreq=sfreq)),
            ("clf", TorchClassifier(**params)),
        ])

    return build


def _bandpower_rf(sfreq: float, n_classes: int, random_state: int) -> Pipeline:
    return Pipeline([
        ("fb", FilterBank(LogVariance, sfreq=sfreq)),
        # n_jobs=1 deliberately: this estimator is always run inside an outer
        # parallel CV loop, and nesting joblib workers oversubscribes the CPU
        # badly enough on Windows to cost two orders of magnitude in wall time.
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=1,
            random_state=random_state, class_weight="balanced_subsample")),
    ])


REGISTRY: dict[str, Callable[[float, int, int], Pipeline]] = {
    # Classical
    "csp_lda": _csp_lda,
    "fbcsp_lda": _fbcsp_lda,
    "riemann_ts": _riemann_ts,
    "riemann_ts_aligned": _riemann_ts_aligned,
    "fb_riemann_ts": _fb_riemann_ts,
    "bandpower_rf": _bandpower_rf,
    # Neural. Require PyTorch; constructing one without it raises ImportError.
    "eegnet": _deep("eegnet", lr=1e-3, batch_size=64, max_epochs=300,
                    patience=50),
    "shallownet": _deep("shallownet", lr=1e-3, batch_size=64, max_epochs=300,
                        patience=50),
    "conformer": _deep("conformer", lr=5e-4, batch_size=64, max_epochs=300,
                       patience=50, weight_decay=1e-3),
}

#: Pipelines backed by a neural network. Used to decide whether a benchmark run
#: needs a GPU and to size batches during evaluation.
DEEP_PIPELINES = frozenset({"eegnet", "shallownet", "conformer"})

#: Pipelines whose transform depends on the batch it is given. Recorded in the
#: model card so the serving layer can warn when handed a single epoch.
TRANSDUCTIVE_PIPELINES = frozenset({"riemann_ts_aligned"})

#: Chosen empirically over the full 105-subject benchmark; see `bwt benchmark`
#: and the results table in the README.
#:
#: CSP+LDA is the default because it is the best *cross-subject* performer
#: (0.611 vs 0.578 for `riemann_ts`), and cross-subject is the regime a shipped
#: artifact actually operates in: it is fitted on the training subjects and then
#: applied to a person it has never seen. The Riemannian pipelines score higher
#: within-subject (0.631 for `riemann_ts_aligned`) and are the better choice
#: when the model is calibrated on its end user's own recordings.
DEFAULT_PIPELINE = "csp_lda"


def build_pipeline(
    name: str = DEFAULT_PIPELINE,
    *,
    sfreq: float = 160.0,
    n_classes: int = 2,
    random_state: int = 42,
) -> Pipeline:
    """Instantiate a fresh, unfitted pipeline by name."""
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown pipeline {name!r}; available: {', '.join(sorted(REGISTRY))}"
        ) from None
    return factory(sfreq, n_classes, random_state)


def pipeline_factory(name: str, *, sfreq: float, n_classes: int,
                     random_state: int = 42) -> Callable[[], Pipeline]:
    """Return a zero-argument callable producing fresh clones.

    Cross-validation needs a brand-new estimator per fold; handing the same
    object around is a classic source of state bleeding between folds.
    """
    template = build_pipeline(name, sfreq=sfreq, n_classes=n_classes,
                              random_state=random_state)
    return lambda: clone(template)


__all__ = [
    "DEEP_PIPELINES",
    "DEFAULT_BANDS",
    "DEFAULT_PIPELINE",
    "REGISTRY",
    "SENSORIMOTOR_BAND",
    "TRANSDUCTIVE_PIPELINES",
    "BandpassFilter",
    "FilterBank",
    "LogVariance",
    "RiemannianRecenter",
    "build_pipeline",
    "pipeline_factory",
]
