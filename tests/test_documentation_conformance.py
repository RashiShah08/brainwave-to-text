"""Conformance with the dataset's own published description.

Everything asserted here is quoted from the EEGMMIDB documentation at
https://archive.physionet.org/pn4/eegmmidb/ and checked against two things: the
constants this package encodes, and the recordings actually on disk.

The point is to catch drift in either direction. A wrong constant silently
mislabels every trial; a corrupt or partial download silently changes every
number. Both failure modes produce plausible output rather than an error, so
neither is caught by ordinary tests.

Tests needing the corpus are marked ``slow`` and skip when it is absent.
"""

from __future__ import annotations

import collections

import pytest

from bwt.data.physionet import (
    BASELINE_RUNS,
    EXECUTED_FF_RUNS,
    EXECUTED_LR_RUNS,
    EXPECTED_N_CHANNELS,
    EXPECTED_SFREQ,
    IMAGINED_FF_RUNS,
    IMAGINED_LR_RUNS,
    N_SUBJECTS_TOTAL,
    Execution,
    Movement,
    available_subjects,
    edf_path,
    run_spec,
)

# --------------------------------------------------------------------------- #
# The documented run order, transcribed verbatim:
#
#   1.  Baseline, eyes open
#   2.  Baseline, eyes closed
#   3.  Task 1 (open and close left or right fist)
#   4.  Task 2 (imagine opening and closing left or right fist)
#   5.  Task 3 (open and close both fists or both feet)
#   6.  Task 4 (imagine opening and closing both fists or both feet)
#   7.  Task 1        11. Task 1
#   8.  Task 2        12. Task 2
#   9.  Task 3        13. Task 3
#   10. Task 4        14. Task 4
# --------------------------------------------------------------------------- #

DOCUMENTED_RUN_ORDER: dict[int, str] = {
    1: "baseline_eyes_open",
    2: "baseline_eyes_closed",
    3: "task1", 4: "task2", 5: "task3", 6: "task4",
    7: "task1", 8: "task2", 9: "task3", 10: "task4",
    11: "task1", 12: "task2", 13: "task3", 14: "task4",
}

#: Task 1 = executed left/right fist; Task 2 = imagined left/right fist;
#: Task 3 = executed both fists/both feet; Task 4 = imagined both fists/feet.
DOCUMENTED_TASK_SEMANTICS = {
    "task1": (Execution.EXECUTED, Movement.LEFT_FIST, Movement.RIGHT_FIST),
    "task2": (Execution.IMAGINED, Movement.LEFT_FIST, Movement.RIGHT_FIST),
    "task3": (Execution.EXECUTED, Movement.BOTH_FISTS, Movement.BOTH_FEET),
    "task4": (Execution.IMAGINED, Movement.BOTH_FISTS, Movement.BOTH_FEET),
}


class TestDocumentedConstants:
    """"109 volunteers ... 64-channel EEG ... 160 samples per second"."""

    def test_subject_count(self):
        assert N_SUBJECTS_TOTAL == 109

    def test_channel_count(self):
        assert EXPECTED_N_CHANNELS == 64

    def test_sampling_rate(self):
        assert EXPECTED_SFREQ == 160.0


class TestDocumentedRunOrder:
    """The 14-run sequence, checked against how this package groups runs."""

    def test_every_run_is_accounted_for(self):
        grouped = (
            set(BASELINE_RUNS)
            | set(EXECUTED_LR_RUNS) | set(IMAGINED_LR_RUNS)
            | set(EXECUTED_FF_RUNS) | set(IMAGINED_FF_RUNS)
        )
        assert grouped == set(DOCUMENTED_RUN_ORDER)

    def test_run_groups_are_disjoint(self):
        groups = [BASELINE_RUNS, EXECUTED_LR_RUNS, IMAGINED_LR_RUNS,
                  EXECUTED_FF_RUNS, IMAGINED_FF_RUNS]
        flat = [r for g in groups for r in g]
        assert len(flat) == len(set(flat)) == 14

    def test_baselines_are_runs_one_and_two(self):
        assert set(BASELINE_RUNS) == {1, 2}

    def test_task_repetitions_are_three_each(self):
        for group in (EXECUTED_LR_RUNS, IMAGINED_LR_RUNS,
                      EXECUTED_FF_RUNS, IMAGINED_FF_RUNS):
            assert len(group) == 3, "each task is repeated three times"

    @pytest.mark.parametrize("run,label", sorted(DOCUMENTED_RUN_ORDER.items()))
    def test_each_run_matches_its_documented_task(self, run, label):
        spec = run_spec(run)

        if label.startswith("baseline"):
            assert spec.execution is Execution.NONE
            expected = (Movement.EYES_OPEN if label.endswith("open")
                        else Movement.EYES_CLOSED)
            assert spec.movement_of("T0") is expected
            return

        execution, t1, t2 = DOCUMENTED_TASK_SEMANTICS[label]
        assert spec.execution is execution, f"run {run} should be {execution}"
        assert spec.movement_of("T1") is t1, f"run {run}: T1 should be {t1}"
        assert spec.movement_of("T2") is t2, f"run {run}: T2 should be {t2}"
        assert spec.movement_of("T0") is Movement.REST


class TestDocumentedAnnotations:
    """"T0: rest. T1: left fist (tasks 1-2) or both fists (tasks 3-4).
    T2: right fist (tasks 1-2) or both feet (tasks 3-4)."."""

    def test_t0_is_rest_in_every_task_run(self):
        for run in (*EXECUTED_LR_RUNS, *IMAGINED_LR_RUNS,
                    *EXECUTED_FF_RUNS, *IMAGINED_FF_RUNS):
            assert run_spec(run).movement_of("T0") is Movement.REST

    def test_t1_t2_in_fist_tasks(self):
        for run in (*EXECUTED_LR_RUNS, *IMAGINED_LR_RUNS):
            assert run_spec(run).movement_of("T1") is Movement.LEFT_FIST
            assert run_spec(run).movement_of("T2") is Movement.RIGHT_FIST

    def test_t1_t2_in_fists_feet_tasks(self):
        for run in (*EXECUTED_FF_RUNS, *IMAGINED_FF_RUNS):
            assert run_spec(run).movement_of("T1") is Movement.BOTH_FISTS
            assert run_spec(run).movement_of("T2") is Movement.BOTH_FEET

    def test_unknown_annotation_is_not_silently_mapped(self):
        assert run_spec(4).movement_of("T7") is None


@pytest.mark.slow
class TestCorpusMatchesDocumentation:
    """The recordings on disk, against what the documentation promises."""

    def test_subject_count_on_disk(self, real_data_root):
        every = available_subjects(real_data_root, include_excluded=True)
        assert len(every) == 109, "documentation states 109 volunteers"

    def test_fourteen_runs_per_subject(self, real_data_root):
        for subject in available_subjects(real_data_root, include_excluded=True):
            present = [r for r in range(1, 15)
                       if edf_path(subject, r, real_data_root).is_file()]
            assert present == list(range(1, 15)), f"S{subject:03d} is incomplete"

    def test_recording_count(self, real_data_root):
        """"over 1500 EEG recordings" -- 109 x 14 = 1526."""
        found = sum(1 for _ in real_data_root.glob("S*/*.edf"))
        assert found == 1526
        assert found > 1500

    def test_every_file_has_an_event_annotation_file(self, real_data_root):
        missing = [p.name for p in real_data_root.glob("S*/*.edf")
                   if not p.with_suffix(".edf.event").is_file()]
        assert not missing, f"missing .event files: {missing[:5]}"

    def test_channel_count_and_rate_of_a_sample(self, real_data_root):
        import mne

        for subject in (1, 42, 109):
            raw = mne.io.read_raw_edf(
                str(edf_path(subject, 4, real_data_root)),
                preload=False, verbose="ERROR",
            )
            assert len(raw.ch_names) == EXPECTED_N_CHANNELS
            assert float(raw.info["sfreq"]) == EXPECTED_SFREQ

    def test_baseline_runs_last_about_one_minute(self, real_data_root):
        """"Baseline ... (1 minute)"."""
        import mne

        for run in BASELINE_RUNS:
            raw = mne.io.read_raw_edf(str(edf_path(1, run, real_data_root)),
                                      preload=False, verbose="ERROR")
            seconds = raw.n_times / raw.info["sfreq"]
            assert 55 <= seconds <= 65, f"run {run} is {seconds:.0f}s"

    def test_task_runs_last_about_two_minutes(self, real_data_root):
        """"three repetitions each of four tasks (2 minutes each)"."""
        import mne

        for run in (3, 4, 5, 6):
            raw = mne.io.read_raw_edf(str(edf_path(1, run, real_data_root)),
                                      preload=False, verbose="ERROR")
            seconds = raw.n_times / raw.info["sfreq"]
            assert 115 <= seconds <= 130, f"run {run} is {seconds:.0f}s"

    def test_task_runs_carry_t0_t1_t2_and_baselines_do_not(self, real_data_root):
        import mne

        for run, expect_cues in [(1, False), (2, False), (4, True), (6, True)]:
            raw = mne.io.read_raw_edf(str(edf_path(1, run, real_data_root)),
                                      preload=False, verbose="ERROR")
            seen = set(raw.annotations.description)
            assert "T0" in seen
            if expect_cues:
                assert {"T1", "T2"} <= seen, f"run {run} lacks movement cues"
            else:
                assert not ({"T1", "T2"} & seen), f"baseline run {run} has cues"

    def test_channel_names_are_ten_ten_labels(self, real_data_root):
        """Documentation: "the international 10-10 electrode system"."""
        from bwt.data.epochs import read_standardised_raw

        raw = read_standardised_raw(edf_path(1, 4, real_data_root))
        names = set(raw.ch_names)
        # Landmark 10-10 sites that must be present in a 64-channel cap.
        for expected in ("C3", "Cz", "C4", "Fz", "Pz", "Oz", "T7", "T8"):
            assert expected in names, f"{expected} missing from the montage"
        assert len(raw.ch_names) == EXPECTED_N_CHANNELS


@pytest.mark.slow
class TestExclusionsAreEmpiricallyJustified:
    """The documentation lists no errata, so each exclusion must be shown."""

    def test_three_subjects_are_not_at_the_documented_rate(self, real_data_root):
        import mne

        for subject in (88, 92, 100):
            raw = mne.io.read_raw_edf(str(edf_path(subject, 4, real_data_root)),
                                      preload=False, verbose="ERROR")
            assert float(raw.info["sfreq"]) != EXPECTED_SFREQ, (
                f"S{subject:03d} was excluded for its sampling rate; it now "
                "matches the documented 160 Hz, so revisit the exclusion"
            )

    def test_s089_baseline_annotations_are_malformed(self, real_data_root):
        import mne

        raw = mne.io.read_raw_edf(str(edf_path(89, 1, real_data_root)),
                                  preload=False, verbose="ERROR")
        seen = collections.Counter(raw.annotations.description)
        assert "T1" in seen, (
            "S089 was excluded because its baseline run carries a movement cue "
            "where the protocol allows only rest; that is no longer true"
        )
