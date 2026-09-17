"""Recordings built to break the decoder, and inference under contention.

Channel spelling, duplicated and surplus channels, cue annotations that are
malformed, duplicated, flooding or sitting on the last valid sample, every
wrong sampling rate, a ten-minute file, exact sample-level agreement between the
stream and the file decode, and a transductive model hammered from many threads
at once. Synthetic throughout (``tests/_factories.py``).
"""

from __future__ import annotations

import io
import json
import threading
import time

import numpy as np
import pytest

import _factories as fx
from conftest import EEGBCI_CHANNELS

SFREQ = 160.0
OFFSET = 80  # tmin 0.5 s at 160 Hz
WINDOW = 481


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    return fx.build_artifact(tmp_path_factory.mktemp("hostile") / "lr")


@pytest.fixture(scope="module")
def predictor(artifact):
    from bwt.serving.predictor import Predictor

    return Predictor.load(artifact)


@pytest.fixture(scope="module")
def client(artifact):
    return fx.make_app(artifact).test_client()


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("hostile_edfs")


def post(client, path, url="/api/v1/predict"):
    return client.post(url, data={"file": (io.BytesIO(path.read_bytes()), "rec.edf")},
                       content_type="multipart/form-data")


def frames(response):
    return [json.loads(x) for x in response.get_data(as_text=True).splitlines() if x]


def probabilities(batch):
    return np.array([[p.probabilities[c] for c in batch.classes]
                     for p in batch.predictions])


class TestChannelNaming:
    @pytest.mark.parametrize("spelling", ["lower", "upper"])
    def test_the_spelling_of_channel_names_does_not_change_the_answer(
            self, predictor, workdir, spelling):
        canonical = predictor.predict_edf(
            fx.write_edf(workdir / "canonical.edf", duration=30.0))
        names = [n.lower() if spelling == "lower" else n.upper()
                 for n in EEGBCI_CHANNELS]
        variant = predictor.predict_edf(
            fx.write_edf(workdir / f"{spelling}.edf", duration=30.0, ch_names=names))
        assert variant.n == canonical.n
        np.testing.assert_allclose(probabilities(variant), probabilities(canonical),
                                   atol=1e-6)

    def test_a_duplicated_channel_is_refused_naming_what_is_missing(self, client,
                                                                    workdir):
        names = list(EEGBCI_CHANNELS)
        names[names.index("C4")] = "C3"
        with pytest.warns(RuntimeWarning, match="not unique"):
            path = fx.write_edf(workdir / "duplicate.edf", duration=20.0, ch_names=names)
        body = post(client, path).get_json()
        assert body["error"] == "input_contract"
        assert "C4" in body["message"]

    def test_surplus_channels_are_ignored(self, client, workdir):
        names = [*EEGBCI_CHANNELS, "X1", "X2", "EOG1", "EOG2", "ECG", "EMG1"]
        path = fx.write_edf(workdir / "surplus.edf", duration=40.0, ch_names=names)
        body = post(client, path).get_json()
        cues = fx.alternating_cues(40.0)
        assert body["n_epochs"] == len(fx.cue_labels(cues, body["classes"]))
        labels = [p["label"] for p in body["predictions"]]
        truth = fx.cue_labels(cues, body["classes"])
        assert sum(a == b for a, b in zip(labels, truth, strict=True)) / len(truth) >= 0.8

    def test_a_montage_missing_half_its_channels_lists_them(self, client, workdir):
        path = fx.write_edf(workdir / "half.edf", duration=20.0,
                            ch_names=EEGBCI_CHANNELS[:32])
        body = post(client, path).get_json()
        assert body["error"] == "input_contract"
        assert "32 channel(s)" in body["message"]


class TestCues:
    def _pairs(self, duration):
        return [(o, d) for o, d in fx.alternating_cues(duration, include_rest=False)]

    def test_padded_cue_labels_are_still_cues(self, predictor, workdir):
        cues = self._pairs(30.0)
        clean = predictor.predict_edf(fx.write_edf(workdir / "clean.edf",
                                                   duration=30.0, cues=cues))
        padded = predictor.predict_edf(fx.write_edf(
            workdir / "padded.edf", duration=30.0,
            cues=[(o, f" {d} ") for o, d in cues]))
        assert padded.epoching == "cue_locked" and padded.n == clean.n
        assert [p.onset_seconds for p in padded.predictions] == \
            [p.onset_seconds for p in clean.predictions]

    def test_lowercase_labels_are_not_cues_and_the_reply_says_so(self, predictor,
                                                                 workdir):
        cues = [(o, d.lower()) for o, d in self._pairs(30.0)]
        batch = predictor.predict_edf(fx.write_edf(workdir / "lower_cues.edf",
                                                   duration=30.0, cues=cues))
        assert batch.epoching == "sliding_window"
        assert any("No T1/T2" in w for w in batch.warnings)

    def test_duplicated_cues_give_duplicated_identical_trials(self, predictor,
                                                              workdir):
        cues = [(1.0, "T1"), (1.0, "T1"), (6.0, "T2"), (6.0, "T2")]
        batch = predictor.predict_edf(fx.write_edf(workdir / "dupe_cues.edf",
                                                   duration=15.0, cues=cues))
        assert batch.n == 4
        assert batch.predictions[0].probabilities == batch.predictions[1].probabilities
        assert batch.predictions[2].probabilities == batch.predictions[3].probabilities

    def test_the_last_valid_sample_is_kept_and_one_past_it_is_dropped(
            self, predictor, workdir):
        n_samples = int(20.0 * SFREQ)
        last_start = n_samples - WINDOW
        fits = (last_start - OFFSET) / SFREQ
        overruns = (last_start + 1 - OFFSET) / SFREQ
        batch = predictor.predict_edf(fx.write_edf(
            workdir / "boundary.edf", duration=20.0,
            cues=[(1.0, "T1"), (fits, "T2"), (overruns, "T1")]))
        onsets = [round(p.onset_seconds, 5) for p in batch.predictions]
        assert round(fits, 5) in onsets
        assert round(overruns, 5) not in onsets

    @pytest.mark.parametrize("dropped", [0, 1, 5])
    def test_the_cap_warns_only_when_trials_were_actually_dropped(self, artifact,
                                                                  workdir, dropped):
        from bwt.serving.predictor import Predictor

        cues = self._pairs(60.0)
        path = workdir / "exact_cap.edf"
        if not path.exists():
            fx.write_edf(path, duration=60.0, cues=cues)
        cap = len(cues) - dropped
        batch = Predictor.load(artifact, max_epochs=cap).predict_edf(path)
        assert batch.n == cap
        truncation = [w for w in batch.warnings if "only the first" in w]
        if dropped:
            assert truncation == [f"recording yielded {len(cues)} epochs; only the "
                                  f"first {cap} were decoded"]
        else:
            assert truncation == [], "a recording of exactly the cap lost nothing"

    def test_a_flood_of_cues_is_capped_and_bounded_in_time(self, artifact, workdir):
        duration = 100.0
        cues = [(round(0.05 * i, 3), "T1" if i % 2 else "T2")
                for i in range(int((duration - 4) / 0.05))]
        path = fx.write_edf(workdir / "flood.edf", duration=duration, cues=cues)
        client = fx.make_app(artifact, max_epochs=256).test_client()
        started = time.perf_counter()
        body = post(client, path).get_json()
        assert time.perf_counter() - started < 60
        assert body["n_epochs"] == 256
        assert any("only the first 256" in w for w in body["warnings"])


class TestSamplingRates:
    @pytest.mark.parametrize("rate", [80.0, 128.0, 159.0, 161.0, 250.0, 320.0])
    def test_every_wrong_rate_is_refused_by_name_on_both_paths(self, client, workdir,
                                                               rate):
        path = fx.write_edf(workdir / f"rate_{rate:g}.edf", duration=20.0, sfreq=rate)
        body = post(client, path).get_json()
        assert body["error"] == "input_contract"
        assert f"{rate:g} Hz" in body["message"] and "160 Hz" in body["message"]
        assert frames(post(client, path, "/api/v1/stream")) == [
            {"type": "error", "message": body["message"]}]

    @pytest.mark.parametrize(("rate", "accepted"), [
        (160.0, True), (160.0 * (1 + 5e-7), True), (160.0 * (1 + 5e-6), False),
        (159.99, False),
    ])
    def test_the_rate_tolerance_is_relative_and_tight(self, predictor, rate, accepted):
        from bwt.serving.predictor import InputContractError

        if accepted:
            predictor._require_rate(rate)
        else:
            with pytest.raises(InputContractError):
                predictor._require_rate(rate)


class TestLongRecording:
    def test_ten_minutes_decodes_and_streams_within_the_caps(self, artifact, workdir):
        path = fx.write_edf(workdir / "ten_minutes.edf", duration=600.0)
        client = fx.make_app(artifact, max_epochs=256).test_client()

        started = time.perf_counter()
        body = post(client, path).get_json()
        n_cues = len(fx.cue_labels(fx.alternating_cues(600.0), body["classes"]))
        assert body["n_epochs"] == n_cues < 256
        assert not body["warnings"]

        events = frames(post(client, path, "/api/v1/stream"))
        types = [e["type"] for e in events]
        assert events[0]["n_windows"] == 256
        assert types.count("window") == 256
        assert types[-2:] == ["truncated", "end"]
        assert time.perf_counter() - started < 180


class TestStreamDecodeParity:
    def test_every_cue_locked_trial_is_a_stream_window_sample_for_sample(
            self, predictor, workdir):
        # A 0.05 s step is 8 samples, and every cue here starts on a multiple
        # of 8, so each decoded trial is exactly one of the stream's windows.
        path = fx.write_edf(workdir / "parity.edf", duration=30.0)
        X, onsets, mode, _ = predictor.epochs_from_edf(path)
        assert mode == "cue_locked"
        stream = predictor.stream_from_edf(path, step_seconds=0.05)
        windows = {w.start_sample: w for w in stream}
        batch_proba = predictor.predict_proba(predictor.validate_array(X))
        for epoch, onset, expected in zip(X, onsets, batch_proba, strict=True):
            start = round(onset * SFREQ) + OFFSET
            window = windows[start]
            np.testing.assert_array_equal(window.data.astype(np.float32), epoch)
            single = predictor.predict_proba(
                predictor.validate_array(window.data[None]))[0]
            np.testing.assert_allclose(single, expected, atol=1e-6)


class TestInferenceUnderContention:
    def test_a_transductive_model_gives_serial_answers_under_many_threads(
            self, tmp_path_factory, workdir):
        from bwt.serving.predictor import Predictor

        artifact = fx.build_artifact(tmp_path_factory.mktemp("contention") / "four",
                                     pipeline="riemann_ts_aligned", task="mi_four_class")
        predictor = Predictor.load(artifact)
        path = fx.write_edf(workdir / "contention.edf", duration=40.0)
        reference = probabilities(predictor.predict_edf(path))
        stream_windows = [w.data for w in predictor.stream_from_edf(path, step_seconds=1.0)]
        stream_reference = [predictor.predict_proba(predictor.validate_array(d[None]))[0]
                            for d in stream_windows[:10]]

        failures, lock = [], threading.Lock()

        def decoder():
            try:
                for _ in range(3):
                    got = probabilities(predictor.predict_edf(path))
                    np.testing.assert_allclose(got, reference, atol=1e-9)
            except BaseException as exc:
                with lock:
                    failures.append(repr(exc))

        def streamer():
            try:
                for data, expected in zip(stream_windows[:10], stream_reference,
                                          strict=True):
                    got = predictor.predict_proba(predictor.validate_array(data[None]))[0]
                    np.testing.assert_allclose(got, expected, atol=1e-9)
            except BaseException as exc:
                with lock:
                    failures.append(repr(exc))

        workers = ([threading.Thread(target=decoder) for _ in range(6)]
                   + [threading.Thread(target=streamer) for _ in range(6)])
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=300)
        assert not any(w.is_alive() for w in workers)
        assert failures == []
