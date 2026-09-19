"""Tests for the feature/model pipelines."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from tests.conftest import make_epochs

from bwt.deep import torch_available
from bwt.pipelines import (
    DEEP_PIPELINES,
    REGISTRY,
    BandpassFilter,
    FilterBank,
    LogVariance,
    build_pipeline,
    pipeline_factory,
)


class TestBandpassFilter:
    def test_preserves_shape(self):
        X, _ = make_epochs(n_per_class=4)
        out = BandpassFilter(8, 30, sfreq=160.0).fit(X).transform(X)
        assert out.shape == X.shape

    def test_attenuates_out_of_band_power(self):
        sfreq, n = 160.0, 480
        t = np.arange(n) / sfreq
        # 2 Hz (below band) plus 10 Hz (in band)
        signal = (np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 10 * t))
        X = np.tile(signal, (4, 3, 1))

        out = BandpassFilter(8, 30, sfreq=sfreq).fit(X).transform(X)
        spectrum = np.abs(np.fft.rfft(out[0, 0]))
        freqs = np.fft.rfftfreq(n, 1 / sfreq)
        power_2hz = spectrum[np.argmin(np.abs(freqs - 2))]
        power_10hz = spectrum[np.argmin(np.abs(freqs - 10))]
        assert power_10hz > 20 * power_2hz

    def test_rejects_band_above_nyquist(self):
        X, _ = make_epochs(n_per_class=2)
        with pytest.raises(ValueError, match="Nyquist"):
            BandpassFilter(8, 200, sfreq=160.0).fit(X)

    def test_rejects_wrong_channel_count(self):
        X, _ = make_epochs(n_per_class=2, n_channels=8)
        transformer = BandpassFilter(8, 30, sfreq=160.0).fit(X)
        with pytest.raises(ValueError, match="channels"):
            transformer.transform(np.zeros((2, 5, 481)))

    def test_rejects_two_dimensional_input(self):
        with pytest.raises(ValueError, match="trials, channels, times"):
            BandpassFilter().fit(np.zeros((10, 100)))


class TestLogVariance:
    def test_output_shape_is_one_feature_per_channel(self):
        X, _ = make_epochs(n_per_class=5, n_channels=12)
        out = LogVariance().fit_transform(X)
        assert out.shape == (len(X), 12)

    def test_tracks_amplitude(self):
        quiet = np.ones((2, 1, 100)) * 0.0
        quiet[..., ::2] = 1.0
        loud = quiet * 10
        transformer = LogVariance()
        assert transformer.transform(loud)[0, 0] > transformer.transform(quiet)[0, 0]


class TestFilterBank:
    def test_concatenates_one_block_per_band(self):
        X, y = make_epochs(n_per_class=6, n_channels=10)
        bands = ((8.0, 12.0), (12.0, 16.0), (16.0, 24.0))
        bank = FilterBank(LogVariance, bands=bands, sfreq=160.0)
        out = bank.fit_transform(X, y)
        assert out.shape == (len(X), 10 * len(bands))

    def test_fit_transform_matches_fit_then_transform(self):
        X, y = make_epochs(n_per_class=6, n_channels=10)
        bands = ((8.0, 12.0), (12.0, 16.0))
        a = FilterBank(LogVariance, bands=bands, sfreq=160.0).fit_transform(X, y)
        b = FilterBank(LogVariance, bands=bands, sfreq=160.0).fit(X, y).transform(X)
        np.testing.assert_allclose(a, b, rtol=1e-10)


# The neural pipelines need PyTorch, an optional dependency the main CI jobs
# deliberately do not install; the CPU-torch job runs them. Skipped, not
# failed, without it, as the README promises.
_NEEDS_TORCH = pytest.mark.skipif(not torch_available(), reason="PyTorch not installed")


@pytest.mark.parametrize("name", [
    pytest.param(name, marks=_NEEDS_TORCH) if name in DEEP_PIPELINES else name
    for name in sorted(REGISTRY)
])
class TestEveryPipeline:
    def test_fits_and_predicts(self, name):
        X, y = make_epochs(n_per_class=16, n_channels=16)
        model = build_pipeline(name, sfreq=160.0, n_classes=2)
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert set(np.unique(predictions)) <= {0, 1}

    def test_exposes_calibrated_probabilities(self, name):
        X, y = make_epochs(n_per_class=16, n_channels=16)
        model = build_pipeline(name, sfreq=160.0, n_classes=2).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)

    def test_learns_the_synthetic_signal(self, name):
        X, y = make_epochs(n_per_class=24, n_channels=16, seed=3)
        model = build_pipeline(name, sfreq=160.0, n_classes=2).fit(X, y)
        assert (model.predict(X) == y).mean() > 0.7

    def test_is_deterministic(self, name):
        X, y = make_epochs(n_per_class=16, n_channels=16)
        a = build_pipeline(name, sfreq=160.0, n_classes=2).fit(X, y).predict(X)
        b = build_pipeline(name, sfreq=160.0, n_classes=2).fit(X, y).predict(X)
        np.testing.assert_array_equal(a, b)

    def test_rejects_wrong_epoch_length_at_predict(self, name):
        X, y = make_epochs(n_per_class=12, n_channels=16, n_times=481)
        model = build_pipeline(name, sfreq=160.0, n_classes=2).fit(X, y)
        # A different channel count must not silently produce a prediction.
        with pytest.raises(ValueError):
            model.predict(np.zeros((2, 8, 481), dtype=np.float32))


class TestFactory:
    def test_returns_independent_clones(self):
        factory = pipeline_factory("csp_lda", sfreq=160.0, n_classes=2)
        a, b = factory(), factory()
        assert a is not b
        X, y = make_epochs(n_per_class=10, n_channels=12)
        a.fit(X, y)
        # Fitting one must not fit the other.
        with pytest.raises(NotFittedError):
            b.predict(X)

    def test_clone_of_template_is_unfitted(self):
        model = build_pipeline("riemann_ts", sfreq=160.0, n_classes=2)
        X, y = make_epochs(n_per_class=10, n_channels=12)
        model.fit(X, y)
        fresh = clone(model)
        with pytest.raises(NotFittedError):
            fresh.predict(X)

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError, match="unknown pipeline"):
            build_pipeline("deep_telepathy_net")


class TestMultiClass:
    def test_four_class_pipelines_work(self):
        X, y = make_epochs(n_per_class=16, n_channels=16, n_classes=4)
        for name in ("csp_lda", "riemann_ts"):
            model = build_pipeline(name, sfreq=160.0, n_classes=4).fit(X, y)
            proba = model.predict_proba(X)
            assert proba.shape == (len(X), 4)
