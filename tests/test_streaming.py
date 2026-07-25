"""Tests for evidence accumulation and stream replay."""

from __future__ import annotations

import numpy as np
import pytest

from bwt.streaming import (
    EDFStream,
    EvidenceAccumulator,
    evaluate_accumulation,
    simulate_speller_throughput,
)


class TestEvidenceAccumulator:
    def test_rejects_invalid_threshold(self):
        for bad in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="threshold"):
                EvidenceAccumulator(["a", "b"], threshold=bad)

    def test_rejects_invalid_leak(self):
        with pytest.raises(ValueError, match="leak"):
            EvidenceAccumulator(["a", "b"], leak=0.0)

    def test_rejects_wrong_probability_length(self):
        acc = EvidenceAccumulator(["a", "b"])
        with pytest.raises(ValueError, match="expected 2 probabilities"):
            acc.update([0.3, 0.3, 0.4])

    def test_posterior_is_a_distribution(self):
        acc = EvidenceAccumulator(["a", "b", "c"])
        acc.update([0.5, 0.3, 0.2])
        assert acc.posterior.sum() == pytest.approx(1.0)
        assert (acc.posterior >= 0).all()

    def test_commits_on_consistent_evidence(self):
        acc = EvidenceAccumulator(["a", "b"], threshold=0.9, min_windows=1)
        decision = None
        for _ in range(10):
            decision = acc.update([0.7, 0.3])
            if decision:
                break
        assert decision is not None
        assert decision.label == "a"
        assert decision.confidence >= 0.9
        assert not decision.timed_out

    def test_respects_min_windows(self):
        acc = EvidenceAccumulator(["a", "b"], threshold=0.6, min_windows=3)
        assert acc.update([0.99, 0.01]) is None
        assert acc.update([0.99, 0.01]) is None
        assert acc.update([0.99, 0.01]) is not None

    def test_times_out_on_ambiguous_evidence(self):
        acc = EvidenceAccumulator(["a", "b"], threshold=0.99, max_windows=5,
                                  min_windows=1)
        decision = None
        for _ in range(5):
            decision = acc.update([0.5, 0.5])
        assert decision is not None
        assert decision.timed_out
        assert decision.label is None

    def test_conflicting_evidence_cancels(self):
        acc = EvidenceAccumulator(["a", "b"], threshold=0.95, max_windows=100,
                                  min_windows=1, leak=1.0)
        for _ in range(20):
            acc.update([0.8, 0.2])
            acc.update([0.2, 0.8])
        assert acc.posterior[0] == pytest.approx(0.5, abs=0.05)

    def test_more_evidence_increases_confidence(self):
        acc = EvidenceAccumulator(["a", "b"], threshold=0.999, max_windows=100,
                                  min_windows=1, leak=1.0)
        acc.update([0.6, 0.4])
        first = acc.posterior[0]
        acc.update([0.6, 0.4])
        assert acc.posterior[0] > first

    def test_leak_forgets_old_evidence(self):
        strong = EvidenceAccumulator(["a", "b"], threshold=0.999,
                                     max_windows=100, min_windows=1, leak=1.0)
        leaky = EvidenceAccumulator(["a", "b"], threshold=0.999,
                                    max_windows=100, min_windows=1, leak=0.5)
        for _ in range(6):
            strong.update([0.7, 0.3])
            leaky.update([0.7, 0.3])
        # The leaky accumulator has discounted the earlier windows.
        assert leaky.posterior[0] < strong.posterior[0]

    def test_reset_clears_state(self):
        acc = EvidenceAccumulator(["a", "b"], threshold=0.99, min_windows=1)
        acc.update([0.9, 0.1])
        acc.reset()
        assert acc.n_windows == 0
        assert acc.posterior[0] == pytest.approx(0.5)

    def test_zero_probability_does_not_produce_nan(self):
        acc = EvidenceAccumulator(["a", "b"], threshold=0.99, min_windows=1)
        acc.update([0.0, 1.0])
        assert np.isfinite(acc.posterior).all()


class TestEDFStream:
    def _stream(self, n_times=1000, **kwargs):
        data = np.random.default_rng(0).standard_normal((4, n_times))
        return EDFStream(data, sfreq=160.0, window_samples=160,
                         step_samples=80, **kwargs)

    def test_window_count_and_shapes(self):
        stream = self._stream()
        windows = list(stream)
        assert len(windows) == len(stream)
        for window in windows:
            assert window.data.shape == (4, 160)

    def test_windows_advance_by_the_step(self):
        windows = list(self._stream())
        assert windows[1].start_sample - windows[0].start_sample == 80

    def test_onsets_are_in_seconds(self):
        windows = list(self._stream())
        assert windows[1].onset_seconds == pytest.approx(0.5)

    def test_rejects_non_2d_data(self):
        with pytest.raises(ValueError, match="channels, times"):
            EDFStream(np.zeros((2, 3, 4)), 160.0, 10, 5)

    def test_rejects_nonpositive_step(self):
        with pytest.raises(ValueError, match="positive"):
            EDFStream(np.zeros((2, 100)), 160.0, 10, 0)

    def test_short_recording_yields_nothing(self):
        stream = EDFStream(np.zeros((4, 50)), 160.0, 160, 80)
        assert list(stream) == []


class TestThroughputSimulation:
    def test_accumulation_beats_single_window(self):
        result = simulate_speller_throughput(0.61, 2, 0.5, threshold=0.9,
                                             n_trials=800)
        assert result["decision_accuracy"] > 0.61

    def test_higher_threshold_costs_time(self):
        fast = simulate_speller_throughput(0.65, 2, 0.5, threshold=0.75,
                                           n_trials=800)
        careful = simulate_speller_throughput(0.65, 2, 0.5, threshold=0.99,
                                              n_trials=800)
        assert (careful["mean_windows_per_decision"]
                > fast["mean_windows_per_decision"])
        assert careful["decision_accuracy"] >= fast["decision_accuracy"]

    def test_chance_input_gives_no_throughput(self):
        result = simulate_speller_throughput(0.5, 2, 0.5, threshold=0.9,
                                             max_windows=10, n_trials=400)
        assert result["decision_accuracy"] < 0.65


class TestEmpiricalAccumulation:
    """The honest counterpart to the simulation, on realistic probabilities."""

    def _fixture(self, n_subjects=6, n_per_class=30, accuracy=0.62, seed=0):
        rng = np.random.default_rng(seed)
        rows, y, groups = [], [], []
        for subject in range(1, n_subjects + 1):
            # Subject-specific competence, which is what makes real errors
            # correlated within a subject.
            skill = np.clip(rng.normal(accuracy, 0.12), 0.35, 0.95)
            for label in (0, 1):
                for _ in range(n_per_class):
                    p = skill if rng.random() < skill else 1 - skill
                    probs = [p, 1 - p] if label == 0 else [1 - p, p]
                    rows.append(probs)
                    y.append(label)
                    groups.append(subject)
        return np.array(rows), np.array(y), np.array(groups)

    def test_returns_expected_fields(self):
        proba, y, groups = self._fixture()
        result = evaluate_accumulation(proba, y, groups, ["a", "b"],
                                       threshold=0.9, n_sequences=200)
        for key in ("decision_accuracy", "commit_rate",
                    "mean_trials_per_decision", "per_subject_accuracy_mean"):
            assert key in result

    def test_accumulation_helps_on_realistic_data(self):
        proba, y, groups = self._fixture()
        single = float((proba.argmax(axis=1) == y).mean())
        result = evaluate_accumulation(proba, y, groups, ["a", "b"],
                                       threshold=0.9, n_sequences=600)
        assert result["decision_accuracy"] > single

    def test_needs_every_class_present(self):
        proba = np.array([[0.9, 0.1]] * 10)
        y = np.zeros(10, dtype=int)
        groups = np.ones(10, dtype=int)
        with pytest.raises(ValueError, match="every class"):
            evaluate_accumulation(proba, y, groups, ["a", "b"])
