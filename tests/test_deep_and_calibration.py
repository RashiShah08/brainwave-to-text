"""Tests for the neural estimators, calibration, datasets, and explainability."""

from __future__ import annotations

import numpy as np
import pytest

from bwt.deep import torch_available
from tests.conftest import make_epochs

torch = pytest.importorskip("torch") if torch_available() else None
pytestmark = pytest.mark.skipif(not torch_available(), reason="PyTorch not installed")


ARCHITECTURES = ["eegnet", "shallownet", "conformer"]


@pytest.fixture(scope="module")
def small_data():
    return make_epochs(n_per_class=20, n_channels=16, n_times=241, seed=5)


class TestModules:
    @pytest.mark.parametrize("name", ARCHITECTURES)
    def test_forward_shape(self, name):
        from bwt.deep.modules import build_module

        module = build_module(name, n_channels=16, n_times=241, n_classes=3,
                              sfreq=160.0)
        out = module(torch.zeros(4, 1, 16, 241))
        assert out.shape == (4, 3)

    @pytest.mark.parametrize("name", ARCHITECTURES)
    def test_kernels_scale_with_sampling_rate(self, name):
        """A kernel quoted for 128 Hz must not be reused verbatim at 250 Hz."""
        from bwt.deep.modules import build_module

        low = build_module(name, n_channels=8, n_times=200, n_classes=2, sfreq=100.0)
        high = build_module(name, n_channels=8, n_times=500, n_classes=2, sfreq=250.0)
        n_low = sum(p.numel() for p in low.parameters())
        n_high = sum(p.numel() for p in high.parameters())
        assert n_low != n_high

    def test_unknown_architecture_rejected(self):
        from bwt.deep.modules import build_module

        with pytest.raises(ValueError, match="unknown architecture"):
            build_module("mindreader", n_channels=8, n_times=100, n_classes=2)

    def test_eegnet_is_small(self):
        from bwt.deep.modules import build_module

        module = build_module("eegnet", n_channels=64, n_times=481,
                              n_classes=2, sfreq=160.0)
        n_params = sum(p.numel() for p in module.parameters())
        assert n_params < 20000, (
            "EEGNet's whole point is a tiny parameter count relative to the "
            "few thousand trials an EEG study yields"
        )


class TestTorchClassifier:
    @pytest.mark.parametrize("name", ARCHITECTURES)
    def test_fits_and_predicts(self, name, small_data):
        from bwt.deep import TorchClassifier

        X, y = small_data
        clf = TorchClassifier(architecture=name, max_epochs=15, patience=15,
                              device="cpu")
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_learns_the_synthetic_signal(self, small_data):
        from bwt.deep import TorchClassifier

        X, y = small_data
        clf = TorchClassifier(architecture="eegnet", max_epochs=120,
                              patience=120, device="cpu").fit(X, y)
        assert (clf.predict(X) == y).mean() > 0.7

    def test_rejects_wrong_channel_count_at_predict(self, small_data):
        from bwt.deep import TorchClassifier

        X, y = small_data
        clf = TorchClassifier(architecture="eegnet", max_epochs=5,
                              device="cpu").fit(X, y)
        with pytest.raises(ValueError, match="channels"):
            clf.predict(np.zeros((2, 8, 241), dtype=np.float32))

    def test_rejects_wrong_epoch_length_at_predict(self, small_data):
        from bwt.deep import TorchClassifier

        X, y = small_data
        clf = TorchClassifier(architecture="eegnet", max_epochs=5,
                              device="cpu").fit(X, y)
        with pytest.raises(ValueError, match="samples"):
            clf.predict(np.zeros((2, 16, 100), dtype=np.float32))

    def test_normalisation_stats_come_from_training_only(self, small_data):
        from bwt.deep import TorchClassifier

        X, y = small_data
        clf = TorchClassifier(architecture="eegnet", max_epochs=5,
                              device="cpu").fit(X, y)
        before = clf.mean_.copy()
        clf.predict(X * 100)  # wildly different scale
        np.testing.assert_array_equal(clf.mean_, before)

    def test_is_reproducible(self, small_data):
        from bwt.deep import TorchClassifier

        X, y = small_data
        a = TorchClassifier(architecture="eegnet", max_epochs=12,
                            random_state=7, device="cpu").fit(X, y).predict(X)
        b = TorchClassifier(architecture="eegnet", max_epochs=12,
                            random_state=7, device="cpu").fit(X, y).predict(X)
        np.testing.assert_array_equal(a, b)

    def test_survives_a_pickle_round_trip(self, small_data, tmp_path):
        import joblib

        from bwt.deep import TorchClassifier

        X, y = small_data
        clf = TorchClassifier(architecture="eegnet", max_epochs=15,
                              device="cpu").fit(X, y)
        expected = clf.predict_proba(X)

        path = tmp_path / "clf.joblib"
        joblib.dump(clf, path)
        restored = joblib.load(path)
        np.testing.assert_allclose(restored.predict_proba(X), expected, rtol=1e-5)

    def test_single_class_rejected(self):
        from bwt.deep import TorchClassifier

        X, _ = make_epochs(n_per_class=10, n_channels=8, n_times=200)
        with pytest.raises(ValueError, match="at least two classes"):
            TorchClassifier(max_epochs=2, device="cpu").fit(X, np.zeros(len(X), int))


class TestFineTuning:
    def test_partial_fit_changes_predictions(self, small_data):
        from bwt.deep import TorchClassifier

        X, y = small_data
        clf = TorchClassifier(architecture="eegnet", max_epochs=15,
                              device="cpu").fit(X, y)
        before = clf.predict_proba(X).copy()

        flipped = 1 - y
        clf.partial_fit(X, flipped, epochs=40, lr=1e-2)
        after = clf.predict_proba(X)
        assert not np.allclose(before, after)

    def test_clone_does_not_mutate_the_original(self, small_data):
        from bwt.deep import TorchClassifier

        X, y = small_data
        clf = TorchClassifier(architecture="eegnet", max_epochs=15,
                              device="cpu").fit(X, y)
        expected = clf.predict_proba(X).copy()

        clone = clf.clone_for_finetuning()
        clone.partial_fit(X, 1 - y, epochs=30, lr=1e-2)
        np.testing.assert_allclose(clf.predict_proba(X), expected, rtol=1e-5)

    def test_freeze_features_limits_trainable_parameters(self, small_data):
        from bwt.deep import TorchClassifier

        X, y = small_data
        clf = TorchClassifier(architecture="eegnet", max_epochs=10,
                              device="cpu").fit(X, y)
        frozen = clf.clone_for_finetuning(freeze_features=True)
        trainable = sum(p.numel() for p in frozen.module_.parameters()
                        if p.requires_grad)
        assert 0 < trainable < frozen.n_parameters_

    def test_partial_fit_before_fit_is_an_error(self, small_data):
        from bwt.deep import TorchClassifier

        X, y = small_data
        with pytest.raises(RuntimeError, match="fit"):
            TorchClassifier(device="cpu").partial_fit(X, y)

    def test_unseen_calibration_class_rejected(self, small_data):
        from bwt.deep import TorchClassifier

        X, y = small_data
        clf = TorchClassifier(architecture="eegnet", max_epochs=10,
                              device="cpu").fit(X, y)
        with pytest.raises(ValueError, match="unseen classes"):
            clf.partial_fit(X, np.full(len(X), 9))


class TestCalibrationAPI:
    def test_adapt_requires_a_neural_pipeline(self, synthetic_bundle):
        from bwt.calibration import adapt
        from bwt.pipelines import build_pipeline

        model = build_pipeline("csp_lda", sfreq=160.0, n_classes=2)
        model.fit(synthetic_bundle.X, synthetic_bundle.y)
        with pytest.raises(TypeError, match="neural pipeline"):
            adapt(model, synthetic_bundle.X[:10], synthetic_bundle.y[:10])

    def test_adapt_none_is_a_no_op(self, synthetic_bundle):
        from bwt.calibration import adapt
        from bwt.pipelines import build_pipeline

        model = build_pipeline("csp_lda", sfreq=160.0, n_classes=2)
        model.fit(synthetic_bundle.X, synthetic_bundle.y)
        assert adapt(model, None, None, strategy="none") is model

    def test_curve_holds_out_the_target_subject(self, synthetic_bundle):
        """The population model must never be trained on the target."""
        from bwt.calibration import calibration_curve
        from bwt.pipelines import pipeline_factory

        seen = []

        def factory():
            model = pipeline_factory("csp_lda", sfreq=160.0, n_classes=2)()
            original = model.fit

            def spy(X, y, **kw):
                seen.append(len(X))
                return original(X, y, **kw)

            model.fit = spy
            return model

        bundle = synthetic_bundle
        result = calibration_curve(
            bundle, factory, pipeline_name="csp_lda", budgets=(0,),
            strategies=("none",), max_subjects=2,
        )
        per_subject = bundle.n_trials // len(bundle.subjects)
        assert all(n == bundle.n_trials - per_subject for n in seen)
        assert result.points


class TestDatasetRegistry:
    def test_lists_both_datasets(self):
        from bwt.data.datasets import list_datasets

        names = {n for n, _ in list_datasets()}
        assert {"eegmmidb", "bnci2a"} <= names

    def test_unknown_dataset_rejected(self):
        from bwt.data.datasets import get_dataset

        with pytest.raises(ValueError, match="unknown dataset"):
            get_dataset("telepathy_corpus")

    def test_unknown_task_rejected(self):
        from bwt.data.datasets import get_dataset

        with pytest.raises(ValueError, match="no task"):
            get_dataset("bnci2a").task_classes("mi_fists_feet")

    def test_bnci_declares_its_native_rate(self):
        from bwt.data.datasets import get_dataset

        assert get_dataset("bnci2a").sfreq == 250.0


class TestExplain:
    def test_lateralisation_requires_c3_c4(self, synthetic_bundle):
        from bwt.explain import lateralisation_index

        stripped = synthetic_bundle.subset(np.ones(synthetic_bundle.n_trials, bool))
        object.__setattr__(stripped, "ch_names",
                           tuple(f"X{i}" for i in range(stripped.n_channels)))
        with pytest.raises(ValueError, match="C3/C4"):
            lateralisation_index(stripped)

    def test_band_power_shape(self, synthetic_bundle):
        from bwt.explain import band_power_timecourse

        times, curves = band_power_timecourse(synthetic_bundle, "C3")
        assert len(times) > 1
        assert set(curves) == set(synthetic_bundle.classes)
        for values in curves.values():
            assert len(values) == len(times)

    def test_warns_without_pre_cue_baseline(self, synthetic_bundle, caplog):
        from bwt.explain import lateralisation_index

        # The decoding window starts at 0.5 s, so there is no rest baseline.
        result = lateralisation_index(synthetic_bundle)
        assert result["has_pre_cue_baseline"] is False
        assert "WARNING" in result["interpretation"]

    def test_csp_patterns_rejects_a_non_csp_pipeline(self, synthetic_bundle,
                                                    trained_artifact, tmp_path):
        from bwt.artifacts import load_artifact
        from bwt.explain import csp_patterns
        from bwt.pipelines import build_pipeline

        _, card = load_artifact(trained_artifact)
        model = build_pipeline("riemann_ts", sfreq=160.0, n_classes=2)
        model.fit(synthetic_bundle.X, synthetic_bundle.y)
        with pytest.raises(ValueError, match="no CSP step"):
            csp_patterns(model, card, output=tmp_path / "x.png")
