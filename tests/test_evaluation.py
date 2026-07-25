"""Tests for the cross-validation protocols and the leakage guard."""

from __future__ import annotations

import numpy as np
import pytest

from bwt.evaluation import (
    assert_no_subject_leakage,
    cross_subject_cv,
    within_subject_cv,
)
from bwt.pipelines import pipeline_factory


@pytest.fixture
def factory(synthetic_bundle):
    return pipeline_factory("csp_lda", sfreq=synthetic_bundle.sfreq, n_classes=2)


class TestLeakageGuard:
    def test_accepts_a_clean_split(self):
        groups = np.array([1, 1, 2, 2, 3, 3])
        assert_no_subject_leakage(groups, np.array([0, 1, 2, 3]), np.array([4, 5]))

    def test_rejects_shared_subject(self):
        groups = np.array([1, 1, 2, 2, 3, 3])
        with pytest.raises(AssertionError, match="both train and test"):
            assert_no_subject_leakage(groups, np.array([0, 1, 2]), np.array([3, 4, 5]))

    def test_rejects_overlapping_indices(self):
        groups = np.array([1, 1, 2, 2])
        with pytest.raises(AssertionError, match="overlap"):
            assert_no_subject_leakage(groups, np.array([0, 1, 2]), np.array([2, 3]))

    def test_catches_the_v1_bug_shape(self):
        """A plain random split of a grouped dataset must be rejected."""
        rng = np.random.default_rng(0)
        groups = np.repeat([1, 2, 3, 4], 25)
        indices = rng.permutation(len(groups))
        train, test = indices[:80], indices[80:]
        with pytest.raises(AssertionError):
            assert_no_subject_leakage(groups, train, test)


class TestCrossSubject:
    def test_runs_and_holds_out_whole_subjects(self, synthetic_bundle, factory):
        result = cross_subject_cv(
            synthetic_bundle, factory, pipeline_name="csp_lda", n_splits=3
        )
        assert len(result.folds) == 3
        assert result.protocol == "cross_subject"

        seen: set[int] = set()
        for fold in result.folds:
            assert fold.test_subjects
            assert not (seen & set(fold.test_subjects)), "a subject was tested twice"
            seen.update(fold.test_subjects)

    def test_learns_the_synthetic_signal(self, synthetic_bundle, factory):
        result = cross_subject_cv(synthetic_bundle, factory, n_splits=3)
        assert result.mean_accuracy > 0.7, (
            "the fixture contains a real class difference; failing here means "
            "the pipeline is broken, not that the data is hard"
        )

    def test_reports_chance_and_majority(self, synthetic_bundle, factory):
        result = cross_subject_cv(synthetic_bundle, factory, n_splits=3)
        assert result.chance_level == pytest.approx(0.5)
        assert 0.4 <= result.majority_level <= 0.6

    def test_confusion_matrix_totals_match(self, synthetic_bundle, factory):
        result = cross_subject_cv(synthetic_bundle, factory, n_splits=3)
        assert np.sum(result.confusion) == synthetic_bundle.n_trials


class TestWithinSubject:
    def test_one_fold_entry_per_subject(self, synthetic_bundle, factory):
        result = within_subject_cv(
            synthetic_bundle, factory, n_splits=3, min_trials=10
        )
        assert len(result.folds) == len(synthetic_bundle.subjects)
        assert {f.test_subjects[0] for f in result.folds} == set(
            synthetic_bundle.subjects
        )

    def test_reports_spread_not_just_mean(self, synthetic_bundle, factory):
        result = within_subject_cv(synthetic_bundle, factory, n_splits=3,
                                   min_trials=10)
        low, high = result.ci95
        assert low <= result.mean_accuracy <= high

    def test_skips_subjects_with_too_few_trials(self, synthetic_bundle, factory):
        trimmed = synthetic_bundle.subset(
            (synthetic_bundle.groups != 1) | (np.arange(synthetic_bundle.n_trials) % 100 == 0)
        )
        result = within_subject_cv(trimmed, factory, n_splits=3, min_trials=20)
        assert 1 not in [f.test_subjects[0] for f in result.folds]


class TestResultSerialisation:
    def test_to_dict_is_json_safe(self, synthetic_bundle, factory):
        import json

        result = cross_subject_cv(synthetic_bundle, factory, n_splits=3)
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["mean_accuracy"] == pytest.approx(result.mean_accuracy)
        assert "ci95_low" in payload and "mean_kappa" in payload

    def test_summary_mentions_chance(self, synthetic_bundle, factory):
        result = cross_subject_cv(synthetic_bundle, factory, n_splits=3)
        assert "chance" in result.summary()
