"""Property-based and boundary tests for the core, below the HTTP layer.

Hypothesis generates thousands of inputs per property -- probability streams,
window geometries, band-power sequences -- and shrinks any failure to the
smallest input that still breaks the invariant. The hand-written cases cover
boundaries a generator is unlikely to hit on its own: exact thresholds,
empty and degenerate collections, environment overrides and on-disk
corruption.

As in ``test_api_edge_cases.py``, a defect in the current code is recorded as
``xfail(strict=True)`` with the reason stated, rather than hidden.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from bwt.metrics import information_transfer_rate
from bwt.streaming import (
    BandPowerNormaliser,
    EDFStream,
    EvidenceAccumulator,
    channel_band_power,
)

PROPERTY = settings(max_examples=300, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])

probability = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


# --------------------------------------------------------------------------- #
# Evidence accumulation
# --------------------------------------------------------------------------- #


class TestAccumulatorProperties:
    @PROPERTY
    @given(st.data())
    def test_every_update_obeys_the_decision_rule(self, data):
        k = data.draw(st.integers(2, 6), label="classes")
        threshold = data.draw(st.floats(0.501, 0.999), label="threshold")
        leak = data.draw(st.floats(0.01, 1.0), label="leak")
        max_windows = data.draw(st.integers(2, 25), label="max_windows")
        min_windows = data.draw(st.integers(1, max_windows), label="min_windows")
        steps = data.draw(st.lists(st.lists(probability, min_size=k, max_size=k),
                                   min_size=1, max_size=80), label="stream")

        classes = [f"c{i}" for i in range(k)]
        acc = EvidenceAccumulator(classes, threshold=threshold, leak=leak,
                                  max_windows=max_windows,
                                  min_windows=min_windows)
        since = 0
        for probs in steps:
            decision = acc.update(probs)
            since += 1
            posterior = acc.posterior
            assert np.isfinite(posterior).all()
            assert (posterior >= 0).all()
            assert math.isclose(posterior.sum(), 1.0, abs_tol=1e-9)
            assert acc.n_windows == since

            confident = since >= min_windows and posterior.max() >= threshold
            if decision is None:
                assert not confident and since < max_windows
                continue

            assert decision.n_windows == since
            assert math.isclose(sum(decision.posterior.values()), 1.0,
                                abs_tol=1e-9)
            if confident:
                assert not decision.timed_out
                assert decision.label == classes[int(posterior.argmax())]
                assert decision.confidence >= threshold
            else:
                assert decision.timed_out and decision.label is None
                assert since == max_windows
            acc.reset()
            since = 0

    @PROPERTY
    @given(st.lists(st.lists(st.floats(1e-6, 1.0), min_size=3, max_size=3),
                    min_size=1, max_size=30))
    def test_without_leak_the_posterior_is_the_normalised_product(self, steps):
        acc = EvidenceAccumulator(["a", "b", "c"], threshold=0.999999,
                                  max_windows=10_000, leak=1.0)
        for probs in steps:
            acc.update(probs)
        log_total = np.sum(np.log(np.asarray(steps)), axis=0)
        expected = np.exp(log_total - log_total.max())
        expected /= expected.sum()
        np.testing.assert_allclose(acc.posterior, expected, atol=1e-9)

    @PROPERTY
    @given(st.lists(st.floats(0.05, 0.95), min_size=2, max_size=2),
           st.floats(0.1, 100.0))
    def test_evidence_is_invariant_to_rescaling_each_window(self, probs, scale):
        """Only ratios between classes carry evidence."""
        a = EvidenceAccumulator(["x", "y"], threshold=0.99, max_windows=100)
        b = EvidenceAccumulator(["x", "y"], threshold=0.99, max_windows=100)
        scaled = [min(1.0, p * scale) for p in probs]
        assume(all(p < 1.0 for p in scaled))
        a.update(probs)
        b.update(scaled)
        np.testing.assert_allclose(a.posterior, b.posterior, atol=1e-9)

    def test_uninformative_windows_never_commit_and_always_time_out(self):
        acc = EvidenceAccumulator(["a", "b", "c", "d"], threshold=0.51,
                                  max_windows=7)
        outcomes = [acc.update([0.25] * 4) for _ in range(7)]
        assert outcomes[:6] == [None] * 6
        assert outcomes[6].timed_out and outcomes[6].label is None

    @pytest.mark.parametrize(("p", "commits"), [
        (0.9 + 1e-9, True), (0.9 - 1e-9, False), (0.95, True), (0.5, False),
    ])
    def test_the_threshold_is_a_sharp_boundary(self, p, commits):
        # The posterior is computed in log space, so exactly 0.9 against 0.1
        # may land a float's width either side; one step either way may not.
        acc = EvidenceAccumulator(["a", "b"], threshold=0.9, min_windows=1)
        decision = acc.update([p, 1 - p])
        assert (decision is not None and decision.label == "a") is commits

    def test_ties_resolve_to_the_first_class_deterministically(self):
        acc = EvidenceAccumulator(["a", "b"], threshold=0.5000001, max_windows=1,
                                  min_windows=1)
        decision = acc.update([0.5, 0.5])
        assert decision.timed_out

    @pytest.mark.parametrize("bad", [
        [0.5], [0.2, 0.3, 0.5], [], [[0.5, 0.5]],
    ])
    def test_wrong_shape_is_rejected(self, bad):
        acc = EvidenceAccumulator(["a", "b"])
        with pytest.raises(ValueError):
            acc.update(bad)

    @pytest.mark.parametrize(("kwargs", "message"), [
        ({"threshold": 0.0}, "threshold"), ({"threshold": 1.0}, "threshold"),
        ({"threshold": float("nan")}, "threshold"), ({"leak": 0.0}, "leak"),
        ({"leak": 1.0001}, "leak"), ({"leak": float("nan")}, "leak"),
    ])
    def test_invalid_construction_is_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            EvidenceAccumulator(["a", "b"], **kwargs)

    def test_a_nan_window_cannot_poison_later_evidence(self):
        acc = EvidenceAccumulator(["a", "b"], threshold=0.9)
        try:
            acc.update([float("nan"), 0.5])
        except ValueError:
            return
        for _ in range(10):
            acc.update([0.99, 0.01])
        assert np.isfinite(acc.posterior).all()

    @pytest.mark.parametrize(("max_windows", "min_windows"), [(1, 2), (0, 1), (-5, 1)])
    def test_impossible_window_budgets_are_rejected(self, max_windows,
                                                    min_windows):
        with pytest.raises(ValueError):
            EvidenceAccumulator(["a", "b"], max_windows=max_windows,
                                min_windows=min_windows)


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #


class TestEDFStreamProperties:
    @PROPERTY
    @given(n_channels=st.integers(1, 3), n_samples=st.integers(0, 400),
           window=st.integers(1, 150), step=st.integers(1, 90),
           start=st.integers(0, 300), sfreq=st.sampled_from([100.0, 160.0, 250.0]))
    def test_len_matches_iteration_and_windows_tile_correctly(
            self, n_channels, n_samples, window, step, start, sfreq):
        data = np.arange(n_channels * n_samples, dtype=float).reshape(
            n_channels, n_samples)
        stream = EDFStream(data, sfreq, window, step, start_sample=start)
        windows = list(stream)

        assert len(stream) == len(windows)
        for i, w in enumerate(windows):
            assert w.index == i
            assert w.start_sample == start + i * step
            assert w.start_sample + window <= n_samples
            assert w.data.shape == (n_channels, window)
            np.testing.assert_array_equal(w.data, data[:, w.start_sample:w.start_sample + window])
            assert w.onset_seconds == pytest.approx(w.start_sample / sfreq)
        # Maximal: one more step would not fit.
        next_start = start + len(windows) * step
        assert next_start + window > n_samples

    def test_iterating_twice_gives_the_same_windows(self):
        stream = EDFStream(np.random.default_rng(0).normal(size=(2, 300)),
                           160.0, 50, 20)
        first = [w.data.copy() for w in stream]
        second = [w.data for w in stream]
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize(("shape", "window", "step"), [
        ((10,), 5, 1), ((1, 1, 10), 5, 1), ((1, 10), 0, 1), ((1, 10), 5, 0),
        ((1, 10), -1, 1), ((1, 10), 5, -2),
    ])
    def test_invalid_geometry_is_rejected(self, shape, window, step):
        with pytest.raises(ValueError):
            EDFStream(np.zeros(shape), 160.0, window, step)


# --------------------------------------------------------------------------- #
# Display quantities
# --------------------------------------------------------------------------- #


class TestBandPower:
    @PROPERTY
    @given(st.lists(st.lists(st.floats(-1e6, 1e6), min_size=4, max_size=4),
                    min_size=1, max_size=60),
           st.floats(0.001, 1.0))
    def test_normaliser_output_is_always_finite_and_in_unit_range(self, frames,
                                                                  momentum):
        normalise = BandPowerNormaliser(4, momentum=momentum)
        for frame in frames:
            out = normalise(np.asarray(frame))
            assert out.shape == (4,)
            assert np.isfinite(out).all()
            assert ((out >= 0) & (out <= 1)).all()

    def test_first_frame_and_a_constant_signal_sit_at_mid_scale(self):
        normalise = BandPowerNormaliser(3)
        for _ in range(20):
            np.testing.assert_allclose(normalise(np.array([1.0, -4.0, 9.0])), 0.5)

    @pytest.mark.parametrize("signal", ["flat", "dc", "spike", "huge"])
    def test_band_power_is_finite_for_degenerate_signals(self, signal):
        n = 481
        data = {
            "flat": np.zeros((64, n)),
            "dc": np.full((64, n), 250.0),
            "spike": np.pad(np.full((64, 1), 1e5), ((0, 0), (240, 240))),
            "huge": np.random.default_rng(0).normal(scale=1e6, size=(64, n)),
        }[signal]
        power = channel_band_power(data, 160.0)
        assert power.shape == (64,)
        assert np.isfinite(power).all()

    def test_band_power_tracks_in_band_amplitude_not_out_of_band(self):
        t = np.arange(481) / 160.0
        in_band = np.sin(2 * np.pi * 12 * t)[None, :] * np.array([[1.0], [4.0]])
        out_band = np.sin(2 * np.pi * 60 * t)[None, :] * 50.0
        power_in = channel_band_power(in_band, 160.0)
        assert power_in[1] - power_in[0] == pytest.approx(np.log(16), abs=0.05)
        assert channel_band_power(out_band, 160.0)[0] < power_in[0]


# --------------------------------------------------------------------------- #
# Information transfer rate
# --------------------------------------------------------------------------- #


class TestInformationTransferRate:
    @PROPERTY
    @given(st.integers(2, 64), st.floats(0.1, 30.0), st.floats(0.0, 1.0),
           st.floats(0.0, 1.0))
    def test_bounded_and_monotone_in_accuracy(self, n, seconds, a, b):
        low, high = sorted((a, b))
        itr_low = information_transfer_rate(low, n, seconds)
        itr_high = information_transfer_rate(high, n, seconds)
        ceiling = math.log2(n) * 60.0 / seconds
        for value in (itr_low, itr_high):
            assert 0.0 <= value <= ceiling * (1 + 1e-9)
        assert itr_high >= itr_low - 1e-9

    @pytest.mark.parametrize(("accuracy", "n", "seconds", "expected"), [
        (1.0, 2, 1.0, 60.0),
        (1.0, 4, 3.0, 40.0),
        (0.5, 2, 3.0, 0.0),        # exactly chance
        (0.25, 4, 3.0, 0.0),
        (0.1, 2, 3.0, 0.0),        # below chance
        (1.5, 2, 1.0, 60.0),       # clamped
        (-1.0, 2, 1.0, 0.0),
        (0.9, 1, 3.0, 0.0),        # one class carries no information
        (0.9, 2, 0.0, 0.0),        # no time per trial
        (0.9, 2, -1.0, 0.0),
    ])
    def test_boundaries(self, accuracy, n, seconds, expected):
        assert information_transfer_rate(accuracy, n, seconds) == pytest.approx(expected)

    def test_known_value(self):
        # Wolpaw: P=0.8, N=2 -> 0.2781 bits/trial.
        assert information_transfer_rate(0.8, 2, 60.0) == pytest.approx(0.2781, abs=1e-4)

    def test_nan_accuracy_does_not_return_nan(self):
        assert math.isfinite(information_transfer_rate(float("nan"), 2, 3.0))


# --------------------------------------------------------------------------- #
# CLI argument parsing
# --------------------------------------------------------------------------- #


class TestSubjectList:
    @pytest.mark.parametrize(("text", "expected"), [
        (None, None), ("", None), ("7", [7]), ("1,2,5-9", [1, 2, 5, 6, 7, 8, 9]),
        (" 3 , 1 ,3 ", [1, 3]), ("5-5", [5]), ("2,1-3", [1, 2, 3]),
        ("1,,2,", [1, 2]),
    ])
    def test_valid_specifications(self, text, expected):
        from bwt.cli import _subject_list

        assert _subject_list(text) == expected

    @pytest.mark.parametrize("text", ["a", "1-b", "-3", "1.5", "one,two"])
    def test_malformed_specifications_raise(self, text):
        from bwt.cli import _subject_list

        with pytest.raises(ValueError):
            _subject_list(text)

    @pytest.mark.parametrize("text", ["9-5", ",", "1--3", " , ,", "-3", "5-",
                                      "a-b", "1-2-3"])
    def test_specifications_that_select_nothing_are_errors(self, text):
        from bwt.cli import _subject_list

        with pytest.raises(ValueError):
            _subject_list(text)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class TestConfigOverrides:
    @pytest.mark.parametrize(("raw", "expected"), [
        ("1", True), ("true", True), ("TRUE", True), (" yes ", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("", False),
        ("anything-else", False),
    ])
    def test_boolean_parsing(self, monkeypatch, raw, expected):
        from bwt.config import Config

        monkeypatch.setenv("BWT_SERVE_STRICT_VERSIONS", raw)
        assert Config().with_env().serve.strict_versions is expected

    def test_numeric_overrides_are_typed(self, monkeypatch):
        from bwt.config import Config

        monkeypatch.setenv("BWT_SERVE_PORT", "8080")
        monkeypatch.setenv("BWT_DATA_TMIN", "0.25")
        config = Config().with_env()
        assert config.serve.port == 8080 and isinstance(config.serve.port, int)
        assert config.data.tmin == 0.25

    @pytest.mark.parametrize(("key", "value"), [
        ("BWT_SERVE_PORT", "abc"), ("BWT_SERVE_PORT", "80.5"),
        ("BWT_DATA_TMIN", "early"),
    ])
    def test_malformed_numbers_fail_loudly(self, monkeypatch, key, value):
        from bwt.config import Config

        monkeypatch.setenv(key, value)
        with pytest.raises(ValueError):
            Config().with_env()

    def test_precedence_file_then_env(self, monkeypatch, tmp_path):
        from bwt.config import Config

        path = tmp_path / "c.yaml"
        path.write_text("serve:\n  port: 7000\n  host: 0.0.0.0\n", encoding="utf-8")
        monkeypatch.setenv("BWT_SERVE_PORT", "7001")
        config = Config.load(path)
        assert (config.serve.port, config.serve.host) == (7001, "0.0.0.0")

    @pytest.mark.parametrize("content", ["", "---\n", "~\n", "# only a comment\n"])
    def test_empty_config_files_mean_defaults(self, tmp_path, content):
        from bwt.config import Config

        path = tmp_path / "c.yaml"
        path.write_text(content, encoding="utf-8")
        assert Config.load(path).to_dict() == Config().with_env().to_dict()

    def test_unknown_config_key_names_the_key(self, tmp_path):
        from bwt.config import Config

        path = tmp_path / "c.yaml"
        path.write_text("serve:\n  prot: 8080\n", encoding="utf-8")
        with pytest.raises(TypeError, match="prot"):
            Config.load(path)

    def test_tuple_setting_can_be_overridden_from_the_environment(self,
                                                                  monkeypatch):
        from bwt.config import Config

        monkeypatch.setenv("BWT_TRAIN_PROTOCOLS", "cross_subject")
        assert tuple(Config().with_env().train.protocols) == ("cross_subject",)


# --------------------------------------------------------------------------- #
# Artifacts on disk
# --------------------------------------------------------------------------- #


class TestArtifactsOnDisk:
    def test_unknown_card_fields_are_ignored_for_forward_compatibility(
            self, trained_artifact):
        from bwt.artifacts import CARD_FILE, load_artifact

        card_file = trained_artifact / CARD_FILE
        payload = json.loads(card_file.read_text(encoding="utf-8"))
        payload["field_from_the_future"] = {"x": 1}
        card_file.write_text(json.dumps(payload), encoding="utf-8")
        _, card = load_artifact(trained_artifact)
        assert card.name == payload["name"]

    def test_card_missing_a_required_field_is_refused(self, trained_artifact):
        from bwt.artifacts import CARD_FILE, load_artifact

        card_file = trained_artifact / CARD_FILE
        payload = json.loads(card_file.read_text(encoding="utf-8"))
        del payload["classes"]
        card_file.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(TypeError, match="classes"):
            load_artifact(trained_artifact)

    def test_corrupt_card_json_is_an_error_not_a_silent_default(
            self, trained_artifact):
        from bwt.artifacts import CARD_FILE, load_artifact

        (trained_artifact / CARD_FILE).write_text("{not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_artifact(trained_artifact)

    def test_card_without_weights_is_refused(self, trained_artifact):
        from bwt.artifacts import PIPELINE_FILE, load_artifact

        (trained_artifact / PIPELINE_FILE).unlink()
        with pytest.raises(FileNotFoundError):
            load_artifact(trained_artifact)

    def test_strict_versions_refuses_drift_and_lenient_mode_warns(
            self, trained_artifact, caplog):
        from bwt.artifacts import CARD_FILE, load_artifact

        card_file = trained_artifact / CARD_FILE
        payload = json.loads(card_file.read_text(encoding="utf-8"))
        payload["library_versions"]["sklearn"] = "0.0.1"
        card_file.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match="sklearn"):
            load_artifact(trained_artifact, strict_versions=True)
        load_artifact(trained_artifact, strict_versions=False)

    def test_listing_skips_junk_and_orders_newest_first(self, trained_artifact,
                                                        tmp_path):
        import shutil

        from bwt.artifacts import CARD_FILE, list_artifacts, resolve_artifact

        root = tmp_path / "root"
        root.mkdir()
        for name, stamp in (("old", "2020-01-01T00:00:00Z"),
                            ("new", "2030-01-01T00:00:00Z")):
            shutil.copytree(trained_artifact, root / name)
            card = json.loads((root / name / CARD_FILE).read_text(encoding="utf-8"))
            card["created_utc"] = stamp
            (root / name / CARD_FILE).write_text(json.dumps(card), encoding="utf-8")
        (root / "broken").mkdir()
        (root / "broken" / CARD_FILE).write_text("][", encoding="utf-8")
        (root / "no_card").mkdir()
        (root / "stray_file.txt").write_text("x", encoding="utf-8")

        assert [p.name for p, _ in list_artifacts(root)] == ["new", "old"]
        assert resolve_artifact(None, root=root).name == "new"
        assert resolve_artifact("old", root=root).name == "old"
        with pytest.raises(FileNotFoundError):
            resolve_artifact("stray_file.txt", root=root)
        with pytest.raises(FileNotFoundError):
            resolve_artifact("absent", root=root)
        with pytest.raises(FileNotFoundError):
            resolve_artifact(None, root=tmp_path / "nowhere")


# --------------------------------------------------------------------------- #
# Epoch bundles and the cache
# --------------------------------------------------------------------------- #


class TestEpochBundleEdges:
    def test_round_trip_preserves_non_ascii_metadata(self, synthetic_bundle,
                                                     tmp_path):
        from dataclasses import replace

        from bwt.data.epochs import EpochBundle

        bundle = replace(synthetic_bundle, task="tâche_測試",
                         classes=("gauche ✋", "droite 🤚"))
        bundle.save(tmp_path / "b.npz")
        loaded = EpochBundle.load(tmp_path / "b.npz")
        assert loaded.task == bundle.task and loaded.classes == bundle.classes
        np.testing.assert_array_equal(loaded.X, bundle.X)

    def test_cache_from_another_format_version_is_refused(self, synthetic_bundle,
                                                          tmp_path):
        from bwt.data import epochs

        path = tmp_path / "b.npz"
        synthetic_bundle.save(path)
        with np.load(path) as handle:
            arrays = {k: handle[k] for k in handle.files}
        meta = json.loads(str(arrays["meta"]))
        meta["cache_version"] = epochs.CACHE_VERSION + 1
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez(path, **arrays)
        with pytest.raises(ValueError, match="format version"):
            epochs.EpochBundle.load(path)

    def test_empty_subset_is_a_valid_empty_bundle(self, synthetic_bundle):
        empty = synthetic_bundle.subset(np.zeros(synthetic_bundle.n_trials, bool))
        assert empty.n_trials == 0 and empty.subjects == []
        assert empty.class_counts() == dict.fromkeys(synthetic_bundle.classes, 0)

    def test_concat_skips_empty_bundles_and_refuses_all_empty(self,
                                                              synthetic_bundle):
        from bwt.data.epochs import concat_bundles

        empty = synthetic_bundle.subset(np.zeros(synthetic_bundle.n_trials, bool))
        assert concat_bundles([empty, synthetic_bundle, None]).n_trials == \
            synthetic_bundle.n_trials
        with pytest.raises(ValueError):
            concat_bundles([empty, None])

    @PROPERTY
    @given(st.lists(st.integers(1, 109), min_size=1, max_size=12, unique=True),
           st.floats(-1.0, 1.0), st.floats(1.5, 5.0))
    def test_cache_key_is_deterministic_and_separates_inputs(self, subjects,
                                                             tmin, tmax):
        from bwt.data.datasets import _cache_key

        key = _cache_key("eegmmidb", "mi_left_right", sorted(subjects), tmin, tmax)
        assert key == _cache_key("eegmmidb", "mi_left_right", sorted(subjects),
                                 tmin, tmax)
        assert key.endswith(".npz") and "/" not in key and "\\" not in key
        assert key != _cache_key("bnci2a", "mi_left_right", sorted(subjects),
                                 tmin, tmax)
        assert key != _cache_key("eegmmidb", "mi_left_right", sorted(subjects),
                                 tmin, tmax + 0.5)


# --------------------------------------------------------------------------- #
# Dataset protocol
# --------------------------------------------------------------------------- #


class TestProtocolEdges:
    @pytest.mark.parametrize("run", [0, 15, -1, 100])
    def test_runs_outside_the_protocol_are_refused(self, run):
        from bwt.data.physionet import run_spec

        with pytest.raises(ValueError, match="1-14"):
            run_spec(run)

    def test_unknown_task_lists_the_valid_ones(self):
        from bwt.data.physionet import TASKS, get_task

        with pytest.raises(ValueError) as info:
            get_task("mi_telepathy")
        assert all(name in str(info.value) for name in TASKS)

    def test_every_task_has_a_consistent_class_order(self):
        from bwt.data.physionet import TASKS

        for task in TASKS.values():
            assert len(set(task.classes)) == len(task.classes) >= 2
            assert set(task.label_map.values()) == set(task.classes)
            for cls in task.classes:
                assert task.classes[task.class_index(cls)] == cls

    def test_subject_discovery_ignores_everything_that_is_not_a_subject(
            self, tmp_path):
        from bwt.data.physionet import available_subjects, edf_path

        for name in ("S001", "S002", "S088", "S1000", "s003", "S04", "SXYZ"):
            (tmp_path / name).mkdir()
        (tmp_path / "S005").write_text("a file, not a directory", encoding="utf-8")
        edf_path(1, 4, tmp_path).write_bytes(b"x")

        assert available_subjects(tmp_path) == [1, 2]
        assert available_subjects(tmp_path, include_excluded=True) == [1, 2, 88]
        assert available_subjects(tmp_path, require_runs=(4,)) == [1]
        assert available_subjects(tmp_path / "missing") == []

    def test_checksum_verification_reports_every_failure_mode(self, tmp_path):
        import hashlib

        from bwt.data.physionet import CHECKSUM_FILE, verify_checksums

        (tmp_path / "S001").mkdir()
        good = tmp_path / "S001" / "S001R01.edf"
        good.write_bytes(b"good data")
        bad = tmp_path / "S001" / "S001R02.edf"
        bad.write_bytes(b"tampered")
        spaced = tmp_path / "S001" / "with space.edf"
        spaced.write_bytes(b"spaced")
        digest = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
        (tmp_path / CHECKSUM_FILE).write_text(
            f"{digest(b'good data')} S001/S001R01.edf\n"
            f"{digest(b'original')} S001/S001R02.edf\n"
            f"{digest(b'spaced')} S001/with space.edf\n"
            f"{digest(b'x')} S001/S001R03.edf\n"
            f"{digest(b'x')} RECORDS\n\n",
            encoding="utf-8",
        )
        report = verify_checksums(tmp_path)
        assert report["checked"] == 4
        assert report["verified"] == 2
        assert report["mismatched"] == ["S001/S001R02.edf"]
        assert report["missing"] == ["S001/S001R03.edf"]
        assert report["ok"] is False
        assert verify_checksums(tmp_path, limit=2)["checked"] == 2
        assert verify_checksums(tmp_path, limit=2) == verify_checksums(tmp_path, limit=2)

    def test_missing_manifest_is_an_explicit_error(self, tmp_path):
        from bwt.data.physionet import verify_checksums

        with pytest.raises(FileNotFoundError):
            verify_checksums(tmp_path)


# --------------------------------------------------------------------------- #
# Predictor contract
# --------------------------------------------------------------------------- #


class TestPredictorContract:
    @pytest.fixture
    def predictor(self, trained_artifact):
        from bwt.serving.predictor import Predictor

        return Predictor.load(trained_artifact)

    def test_channel_alignment_is_case_insensitive_and_reorders(self, predictor):
        names = [n.lower() for n in predictor.card.ch_names][::-1]
        picks = predictor._align_channels(names)
        assert [names[i] for i in picks] == [n.lower() for n in predictor.card.ch_names]

    @pytest.mark.parametrize("n", [1, 2, 300])
    def test_any_non_empty_batch_size(self, predictor, n):
        X = np.random.default_rng(n).normal(size=(n, 64, 481)).astype(np.float32)
        predictions = predictor.predict_array(X)
        assert [p.index for p in predictions] == list(range(n))

    def test_an_empty_batch_is_refused_not_answered(self, predictor):
        # Unreachable from the HTTP layer, which refuses a recording with no
        # usable epoch first; direct callers still get an error, not [].
        with pytest.raises(ValueError):
            predictor.predict_array(np.zeros((0, 64, 481), np.float32))

    @pytest.mark.parametrize("shape", [(64,), (1, 1, 64, 481), (2, 63, 481),
                                       (2, 64, 480), (2, 64, 482), (2, 481, 64)])
    def test_every_shape_violation_is_an_input_contract_error(self, predictor,
                                                              shape):
        from bwt.serving.predictor import InputContractError

        with pytest.raises(InputContractError):
            predictor.validate_array(np.zeros(shape))

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_non_finite_samples_are_an_input_contract_error(self, predictor,
                                                            value):
        from bwt.serving.predictor import InputContractError

        X = np.ones((2, 64, 481), np.float32)
        X[0, 10, 100] = value
        with pytest.raises(InputContractError):
            predictor.validate_array(X)

    def test_performance_note_survives_a_card_with_no_evaluation(self,
                                                                 predictor):
        predictor.card.evaluation = {}
        note = predictor.performance_note()
        assert note["cross_subject_accuracy"] is None
        assert note["itr_bits_per_minute"] == 0.0
        json.dumps(note, allow_nan=False)

    def test_empty_batch_summary(self):
        from bwt.serving.predictor import PredictionBatch

        batch = PredictionBatch([], "m", "t", ["a", "b"], "cue_locked")
        assert batch.majority_label() is None
        assert batch.mean_confidence() == 0.0
        assert batch.to_dict()["n_epochs"] == 0


# --------------------------------------------------------------------------- #
# Electrode geometry
# --------------------------------------------------------------------------- #


class TestGeometryEdges:
    def test_unknown_channels_are_reported_not_invented(self):
        from bwt.serving.geometry import electrode_geometry

        geometry = electrode_geometry(["C3", "c4", "NOT_A_SITE", "Cz"])
        assert [e["name"] for e in geometry["electrodes"]] == ["C3", "c4", "Cz"]
        assert geometry["unplaced"] == ["NOT_A_SITE"]
        regions = {e["name"]: e["region"] for e in geometry["electrodes"]}
        assert regions == {"C3": "left_motor", "c4": "right_motor",
                           "Cz": "midline_motor"}

    def test_coordinates_are_unit_scaled(self):
        from bwt.serving.geometry import electrode_geometry
        from conftest import EEGBCI_CHANNELS

        points = np.array([[e["x"], e["y"], e["z"]] for e in
                           electrode_geometry(EEGBCI_CHANNELS)["electrodes"]])
        assert np.abs(points).max() == pytest.approx(1.0, abs=1e-4)
        assert np.abs(points.mean(axis=0)).max() < 1e-3

    @pytest.mark.parametrize("names", [[], ["nope"], ["X1", "X2"]])
    def test_no_placeable_channel_is_an_error(self, names):
        from bwt.serving.geometry import electrode_geometry

        with pytest.raises(ValueError):
            electrode_geometry(names)

    def test_a_single_channel_does_not_divide_by_zero(self):
        from bwt.serving.geometry import electrode_geometry

        (only,) = electrode_geometry(["Cz"])["electrodes"]
        assert (only["x"], only["y"], only["z"]) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# Permutation test
# --------------------------------------------------------------------------- #


class TestPermutationTest:
    @pytest.fixture
    def factory(self, synthetic_bundle):
        from bwt.pipelines import pipeline_factory

        return pipeline_factory("csp_lda", sfreq=synthetic_bundle.sfreq,
                                n_classes=2)

    def test_resumed_run_equals_an_uninterrupted_one(self, synthetic_bundle,
                                                     factory, tmp_path):
        from bwt.evaluation import permutation_test

        straight_p, straight = permutation_test(
            synthetic_bundle, factory, observed=0.9, n_permutations=3, n_splits=3)

        checkpoint = tmp_path / "perm.json"
        permutation_test(synthetic_bundle, factory, observed=0.9,
                         n_permutations=2, n_splits=3, checkpoint=checkpoint)
        resumed_p, resumed = permutation_test(
            synthetic_bundle, factory, observed=0.9, n_permutations=3, n_splits=3,
            checkpoint=checkpoint)

        assert resumed == pytest.approx(straight)
        assert resumed_p == straight_p

    def test_p_value_is_never_zero_and_never_above_one(self, synthetic_bundle,
                                                       factory):
        from bwt.evaluation import permutation_test

        p_high, scores = permutation_test(synthetic_bundle, factory, observed=1.01,
                                          n_permutations=2, n_splits=3)
        assert p_high == pytest.approx(1 / 3)
        p_low, _ = permutation_test(synthetic_bundle, factory, observed=-1.0,
                                    n_permutations=2, n_splits=3)
        assert p_low == 1.0
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_corrupt_or_foreign_checkpoints_are_ignored(self, synthetic_bundle,
                                                        factory, tmp_path):
        from bwt.evaluation import permutation_test

        checkpoint = tmp_path / "perm.json"
        checkpoint.write_text("{corrupt", encoding="utf-8")
        _, scores = permutation_test(synthetic_bundle, factory, observed=0.5,
                                     n_permutations=1, n_splits=3,
                                     checkpoint=checkpoint)
        assert len(scores) == 1

        checkpoint.write_text(json.dumps({"random_state": 999, "scores": [0.1] * 5}),
                              encoding="utf-8")
        _, scores = permutation_test(synthetic_bundle, factory, observed=0.5,
                                     n_permutations=1, n_splits=3,
                                     checkpoint=checkpoint)
        assert len(scores) == 1 and scores != [0.1]
