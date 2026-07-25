"""Regression tests for the run/annotation protocol.

These encode the specific defect this project was rebuilt to fix: ``T1`` and
``T2`` mean different movements in different runs, and executed movement is not
imagined movement. If any of these break, the labels are wrong again.
"""

from __future__ import annotations

import pytest

from bwt.data.physionet import (
    EXCLUDED_SUBJECTS,
    TASKS,
    Execution,
    Movement,
    available_subjects,
    get_task,
    run_spec,
)


class TestRunProtocol:
    @pytest.mark.parametrize("run", [3, 4, 7, 8, 11, 12])
    def test_left_right_runs_map_t1_to_left_hand(self, run):
        spec = run_spec(run)
        assert spec.movement_of("T1") is Movement.LEFT_FIST
        assert spec.movement_of("T2") is Movement.RIGHT_FIST

    @pytest.mark.parametrize("run", [5, 6, 9, 10, 13, 14])
    def test_fists_feet_runs_map_t1_to_both_fists(self, run):
        spec = run_spec(run)
        assert spec.movement_of("T1") is Movement.BOTH_FISTS
        assert spec.movement_of("T2") is Movement.BOTH_FEET

    def test_same_annotation_means_different_things_across_run_families(self):
        """The exact confusion that produced the previous version's bad labels."""
        assert run_spec(4).movement_of("T1") is not run_spec(6).movement_of("T1")
        assert run_spec(4).movement_of("T2") is not run_spec(6).movement_of("T2")

    @pytest.mark.parametrize("run", [3, 5, 7, 9, 11, 13])
    def test_odd_task_runs_are_executed(self, run):
        assert run_spec(run).execution is Execution.EXECUTED

    @pytest.mark.parametrize("run", [4, 6, 8, 10, 12, 14])
    def test_even_task_runs_are_imagined(self, run):
        assert run_spec(run).execution is Execution.IMAGINED

    def test_baseline_runs_have_no_movement_cues(self):
        for run in (1, 2):
            spec = run_spec(run)
            assert spec.execution is Execution.NONE
            assert "T1" not in spec.annotation_map
            assert "T2" not in spec.annotation_map

    def test_unknown_run_rejected(self):
        with pytest.raises(ValueError, match="not part of the EEGMMIDB protocol"):
            run_spec(99)


class TestTasks:
    def test_default_task_is_imagined_only(self):
        task = get_task("mi_left_right")
        assert task.executions == {Execution.IMAGINED}
        assert task.runs == (4, 8, 12)

    def test_no_task_mixes_executed_and_imagined(self):
        for name, task in TASKS.items():
            assert len(task.executions) == 1, (
                f"task {name} mixes {task.executions}; executed movement carries "
                "EMG contamination and must not be pooled with imagery"
            )

    def test_no_task_mixes_hand_and_foot_contrasts(self):
        """Guards against re-creating the 'right fist OR both feet' class."""
        hand = {Movement.LEFT_FIST, Movement.RIGHT_FIST}
        foot = {Movement.BOTH_FISTS, Movement.BOTH_FEET}
        for name, task in TASKS.items():
            by_class: dict[str, set[Movement]] = {}
            for movement, class_name in task.label_map.items():
                by_class.setdefault(class_name, set()).add(movement)
            for class_name, movements in by_class.items():
                assert not (movements & hand and movements & foot), (
                    f"task {name} class {class_name!r} merges hand and foot "
                    f"movements: {movements}"
                )

    def test_rest_is_only_a_class_when_asked_for(self):
        assert "rest" not in get_task("mi_left_right").classes
        assert "rest" in get_task("mi_left_right_rest").classes

    def test_rest_annotation_is_dropped_not_relabelled(self):
        task = get_task("mi_left_right")
        assert task.label_of(4, "T0") is None

    def test_label_indices_match_class_order(self):
        task = get_task("mi_four_class")
        assert task.label_of(4, "T1") == task.classes.index("left_fist")
        assert task.label_of(4, "T2") == task.classes.index("right_fist")
        assert task.label_of(6, "T1") == task.classes.index("both_fists")
        assert task.label_of(6, "T2") == task.classes.index("both_feet")

    def test_four_class_keeps_all_four_distinct(self):
        task = get_task("mi_four_class")
        labels = {task.label_of(4, "T1"), task.label_of(4, "T2"),
                  task.label_of(6, "T1"), task.label_of(6, "T2")}
        assert labels == {0, 1, 2, 3}

    def test_unknown_task_rejected(self):
        with pytest.raises(ValueError, match="unknown task"):
            get_task("mi_telepathy")


class TestExclusions:
    def test_known_bad_subjects_are_excluded(self):
        # 88/92/100 are 128 Hz; 89 has corrupt baseline annotations.
        assert EXCLUDED_SUBJECTS == frozenset({88, 89, 92, 100})

    def test_available_subjects_omits_excluded(self, real_data_root):
        usable = available_subjects(real_data_root)
        assert not (set(usable) & EXCLUDED_SUBJECTS)
        assert len(usable) == 105

    def test_include_excluded_returns_them(self, real_data_root):
        every = available_subjects(real_data_root, include_excluded=True)
        assert EXCLUDED_SUBJECTS <= set(every)

    def test_missing_root_returns_empty(self, tmp_path):
        assert available_subjects(tmp_path / "nope") == []
