"""Tests for the resumable benchmark runner.

The point of checkpointing is that an interrupted run loses nothing, so these
tests interrupt runs deliberately -- by budget, and by simulating a crash -- and
assert that the work already done survives.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bwt.benchmarking import (
    STORE_VERSION,
    CheckpointStore,
    WorkItem,
    assemble,
    execute,
    plan,
    progress,
)
from bwt.pipelines import pipeline_factory


@pytest.fixture
def store(tmp_path):
    return CheckpointStore(tmp_path / "state.json")


@pytest.fixture
def factory_for(synthetic_bundle):
    def build(name: str):
        return pipeline_factory(name, sfreq=synthetic_bundle.sfreq, n_classes=2)

    return build


class TestPlan:
    def test_cross_subject_yields_one_item_per_fold(self, synthetic_bundle):
        items = plan(synthetic_bundle, ["csp_lda"], ["cross_subject"], n_splits=3)
        assert len(items) == 3
        assert {i.key for i in items} == {"0", "1", "2"}

    def test_within_subject_yields_one_item_per_subject(self, synthetic_bundle):
        items = plan(synthetic_bundle, ["csp_lda"], ["within_subject"])
        assert len(items) == len(synthetic_bundle.subjects)

    def test_covers_the_cartesian_product(self, synthetic_bundle):
        items = plan(synthetic_bundle, ["csp_lda", "riemann_ts"],
                     ["cross_subject"], n_splits=3)
        assert len(items) == 6
        assert {i.pipeline for i in items} == {"csp_lda", "riemann_ts"}

    def test_unknown_protocol_rejected(self, synthetic_bundle):
        with pytest.raises(ValueError, match="unsupported protocol"):
            plan(synthetic_bundle, ["csp_lda"], ["telepathy"])

    def test_store_keys_are_unique(self, synthetic_bundle):
        items = plan(synthetic_bundle, ["csp_lda", "riemann_ts"],
                     ["cross_subject", "within_subject"], n_splits=3)
        assert len({i.store_key for i in items}) == len(items)


class TestCheckpointStore:
    def test_round_trips_through_disk(self, tmp_path):
        path = tmp_path / "s.json"
        first = CheckpointStore(path)
        item = WorkItem("csp_lda", "cross_subject", "0")
        first.put(item, [0, 1, 1], [0, 1, 0], [3], 1.5, 42)

        second = CheckpointStore(path)
        assert second.has(item)
        assert second.get(item)["y_true"] == [0, 1, 1]
        assert second.get(item)["n_train"] == 42

    def test_rejects_a_foreign_version(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"version": STORE_VERSION + 5, "folds": {"x": {}}}))
        store = CheckpointStore(path)
        assert store.data["folds"] == {}

    def test_survives_a_corrupt_file(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("{not json at all")
        assert CheckpointStore(path).data["folds"] == {}

    def test_leaves_no_temp_file(self, tmp_path):
        store = CheckpointStore(tmp_path / "s.json")
        store.put(WorkItem("a", "cross_subject", "0"), [0], [0], [1], 0.1, 1)
        assert list(tmp_path.glob("*.tmp*")) == []

    def test_folds_for_filters_by_pipeline_and_protocol(self, tmp_path):
        store = CheckpointStore(tmp_path / "s.json")
        store.put(WorkItem("a", "cross_subject", "0"), [0], [0], [1], 0.1, 1)
        store.put(WorkItem("b", "cross_subject", "0"), [1], [1], [1], 0.1, 1)
        assert set(store.folds_for("a", "cross_subject")) == {"0"}


class TestExecute:
    def test_runs_every_fold_once(self, synthetic_bundle, store, factory_for):
        items = plan(synthetic_bundle, ["csp_lda"], ["cross_subject"], n_splits=3)
        statuses = [s for _, s in execute(synthetic_bundle, items, store,
                                         factory_for, n_splits=3)]
        assert statuses == ["done"] * 3

    def test_second_pass_is_entirely_cached(self, synthetic_bundle, store,
                                            factory_for):
        items = plan(synthetic_bundle, ["csp_lda"], ["cross_subject"], n_splits=3)
        list(execute(synthetic_bundle, items, store, factory_for, n_splits=3))
        statuses = [s for _, s in execute(synthetic_bundle, items, store,
                                         factory_for, n_splits=3)]
        assert statuses == ["cached"] * 3

    def test_budget_stops_early_but_keeps_completed_work(
        self, synthetic_bundle, store, factory_for
    ):
        items = plan(synthetic_bundle, ["csp_lda"], ["cross_subject"], n_splits=3)
        # A zero budget still runs the first item, then stops.
        statuses = [s for _, s in execute(synthetic_bundle, items, store,
                                         factory_for, n_splits=3, time_budget=0)]
        assert statuses == ["done"]
        assert sum(1 for i in items if store.has(i)) == 1

    def test_interrupted_run_resumes_where_it_stopped(
        self, synthetic_bundle, store, factory_for
    ):
        """Simulate a kill: abandon the generator mid-way, then resume."""
        items = plan(synthetic_bundle, ["csp_lda"], ["cross_subject"], n_splits=3)
        runner = execute(synthetic_bundle, items, store, factory_for, n_splits=3)
        next(runner)          # complete exactly one fold
        runner.close()        # the process "dies" here

        assert sum(1 for i in items if store.has(i)) == 1

        statuses = [s for _, s in execute(synthetic_bundle, items, store,
                                         factory_for, n_splits=3)]
        assert statuses.count("cached") == 1
        assert statuses.count("done") == 2
        assert all(store.has(i) for i in items)

    def test_a_failing_pipeline_is_reported_not_raised(
        self, synthetic_bundle, store
    ):
        def broken(name):
            def factory():
                raise RuntimeError("no such model")

            return factory

        items = plan(synthetic_bundle, ["nope"], ["cross_subject"], n_splits=2)
        statuses = [s for _, s in execute(synthetic_bundle, items, store, broken,
                                         n_splits=2)]
        assert statuses == ["error", "error"]
        assert not any(store.has(i) for i in items)

    def test_skipped_subject_is_recorded_so_it_is_not_retried(
        self, synthetic_bundle, store, factory_for
    ):
        # One trial for subject 1 -- far too few for within-subject CV.
        mask = (synthetic_bundle.groups != 1) | (
            np.arange(synthetic_bundle.n_trials) == 0
        )
        trimmed = synthetic_bundle.subset(mask)
        items = [WorkItem("csp_lda", "within_subject", "1")]
        statuses = [s for _, s in execute(trimmed, items, store, factory_for)]
        assert statuses == ["skipped"]
        assert store.has(items[0])


class TestAssemble:
    def test_builds_a_cvresult_from_checkpointed_folds(
        self, synthetic_bundle, store, factory_for
    ):
        items = plan(synthetic_bundle, ["csp_lda"], ["cross_subject"], n_splits=3)
        list(execute(synthetic_bundle, items, store, factory_for, n_splits=3))

        result = assemble(synthetic_bundle, store, "csp_lda", "cross_subject")
        assert result is not None
        assert len(result.folds) == 3
        assert result.protocol == "cross_subject"
        assert np.sum(result.confusion) == synthetic_bundle.n_trials
        assert 0.0 <= result.mean_accuracy <= 1.0

    def test_matches_the_non_resumable_implementation(
        self, synthetic_bundle, store, factory_for
    ):
        """Checkpointing must not change the number that comes out."""
        from bwt.evaluation import cross_subject_cv

        items = plan(synthetic_bundle, ["csp_lda"], ["cross_subject"], n_splits=3)
        list(execute(synthetic_bundle, items, store, factory_for, n_splits=3))
        resumable = assemble(synthetic_bundle, store, "csp_lda", "cross_subject")

        direct = cross_subject_cv(
            synthetic_bundle, factory_for("csp_lda"), n_splits=3
        )
        assert resumable.mean_accuracy == pytest.approx(direct.mean_accuracy)
        assert resumable.confusion == direct.confusion

    def test_returns_none_when_nothing_is_recorded(self, synthetic_bundle, store):
        assert assemble(synthetic_bundle, store, "csp_lda", "cross_subject") is None

    def test_partial_results_still_assemble(self, synthetic_bundle, store,
                                            factory_for):
        items = plan(synthetic_bundle, ["csp_lda"], ["cross_subject"], n_splits=3)
        list(execute(synthetic_bundle, items, store, factory_for, n_splits=3,
                     time_budget=0))
        result = assemble(synthetic_bundle, store, "csp_lda", "cross_subject")
        assert result is not None and len(result.folds) == 1


class TestProgress:
    def test_counts_done_against_planned(self, synthetic_bundle, store,
                                         factory_for):
        items = plan(synthetic_bundle, ["csp_lda"], ["cross_subject"], n_splits=3)
        assert progress(items, store)["csp_lda"]["cross_subject"] == [0, 3]

        list(execute(synthetic_bundle, items, store, factory_for, n_splits=3))
        assert progress(items, store)["csp_lda"]["cross_subject"] == [3, 3]
