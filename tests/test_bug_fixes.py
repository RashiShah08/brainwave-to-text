"""The defects the edge-case suites found, each pinned from every direction.

``test_api_edge_cases.py``, ``test_core_edge_cases.py`` and
``test_frontend_e2e.py`` hold the case that exposed each bug. This file pushes
the fixes themselves: the boundaries around each one, randomised inputs, and the
ways a fix can go wrong on its own -- a budget slot that is never given back, a
refusal that leaks library internals, a validator that rejects valid input, a
guard that lets a bad value mutate state before it objects.

Everything is synthetic (``tests/_factories.py``); nothing needs the dataset.
"""

from __future__ import annotations

import io
import json
import math
import threading
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import _factories as fx

PROPERTY = settings(max_examples=200, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])
STREAM = "/api/v1/stream"


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    return fx.build_artifact(tmp_path_factory.mktemp("fixes") / "lr")


@pytest.fixture(scope="module")
def edfs(tmp_path_factory):
    d = tmp_path_factory.mktemp("fix_edfs")
    return {
        "good": fx.write_edf(d / "good.edf", duration=30.0),
        "short10": fx.write_edf(d / "short10.edf", duration=10.0),
        "five": fx.write_edf(d / "five.edf", duration=5.0, cues=[]),
        "rate128": fx.write_edf(d / "rate128.edf", duration=20.0, sfreq=128.0),
        "flat": fx.write_edf(d / "flat.edf", duration=20.0, signal="flat"),
    }


@pytest.fixture(scope="module")
def app(artifact):
    return fx.make_app(artifact, max_upload_mb=2)


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(scope="module")
def predictor(artifact):
    from bwt.serving.predictor import Predictor

    return Predictor.load(artifact)


def post(client, source, url, *, filename="rec.edf"):
    payload = source.read_bytes() if isinstance(source, Path) else source
    return client.post(url, data={"file": (io.BytesIO(payload), filename)},
                       content_type="multipart/form-data")


def frames(response):
    return [json.loads(line) for line in response.get_data(as_text=True).splitlines()
            if line]


# --------------------------------------------------------------------------- #
# Paced replays have a budget, and every slot comes back
# --------------------------------------------------------------------------- #


class TestReplayBudget:
    @pytest.mark.parametrize(("threads", "slots"), [(1, 1), (2, 1), (4, 3), (9, 8)])
    def test_budget_is_one_less_than_the_worker_pool(self, artifact, edfs, threads,
                                                     slots):
        client = fx.make_app(artifact, threads=threads).test_client()
        # The test client does not start a streamed body until it is read, so
        # an unread response holds its slot exactly as a slow viewer would.
        held = [post(client, edfs["five"], f"{STREAM}?speed=20") for _ in range(slots)]
        try:
            assert all(r.status_code == 200 for r in held)
            refused = post(client, edfs["five"], f"{STREAM}?speed=20")
            assert refused.status_code == 503
            body = refused.get_json()
            assert body["error"] == "service_unavailable"
            assert "speed=0" in body["message"]
        finally:
            held.pop().close()
        again = post(client, edfs["five"], f"{STREAM}?speed=20")
        assert again.status_code == 200 and frames(again)[-1]["type"] == "end"
        for response in held:
            response.close()

    def test_unthrottled_streams_are_never_budgeted(self, artifact, edfs):
        client = fx.make_app(artifact, threads=2).test_client()
        holder = post(client, edfs["five"], f"{STREAM}?speed=20")
        try:
            for _ in range(3):
                free = post(client, edfs["five"], f"{STREAM}?speed=0")
                assert free.status_code == 200 and frames(free)[-1]["type"] == "end"
        finally:
            holder.close()

    def test_a_replay_read_to_the_end_gives_its_slot_back(self, artifact, edfs):
        client = fx.make_app(artifact, threads=2).test_client()
        for _ in range(5):
            response = post(client, edfs["five"], f"{STREAM}?speed=20")
            assert response.status_code == 200
            assert frames(response)[-1]["type"] == "end"

    @pytest.mark.parametrize("failure", ["bad_suffix", "no_file", "empty",
                                         "garbage", "wrong_rate", "flat"])
    def test_every_failure_path_gives_its_slot_back(self, artifact, edfs, failure):
        client = fx.make_app(artifact, threads=2).test_client()  # one slot
        url = f"{STREAM}?speed=20"
        for _ in range(4):
            if failure == "bad_suffix":
                response = post(client, edfs["five"], url, filename="rec.txt")
                assert response.status_code == 400
            elif failure == "no_file":
                response = client.post(url, data={}, content_type="multipart/form-data")
                assert response.status_code == 400
            elif failure == "empty":
                response = post(client, b"", url)
                assert response.status_code == 400
            else:
                source = {"garbage": b"\x00" * 6000, "wrong_rate": edfs["rate128"],
                          "flat": edfs["flat"]}[failure]
                response = post(client, source, url)
                assert response.status_code == 200
                assert frames(response)[-1]["type"] == "error"
            response.close()
        ok = post(client, edfs["five"], url)
        assert ok.status_code == 200 and frames(ok)[-1]["type"] == "end"

    def test_invalid_parameters_are_refused_before_a_slot_is_claimed(self, artifact,
                                                                     edfs):
        client = fx.make_app(artifact, threads=2).test_client()
        for query in ["speed=1&threshold=abc", "speed=1&step=0", "speed=99"]:
            for _ in range(3):
                assert post(client, edfs["five"], f"{STREAM}?{query}").status_code == 400
        ok = post(client, edfs["five"], f"{STREAM}?speed=20")
        assert ok.status_code == 200 and frames(ok)[-1]["type"] == "end"

    def test_a_response_closed_without_being_iterated_still_releases_everything(
            self, artifact, edfs, tmp_path, monkeypatch):
        """PEP 3333 lets a server close an application's iterable without ever
        iterating it -- a client gone before the first byte. The stream's
        generator then never starts, so its ``finally`` never runs: the slot
        and the spooled upload must be released by closing alone. Neither the
        test client nor waitress takes that path, so drive the WSGI app raw."""
        import tempfile

        from werkzeug.test import EnvironBuilder

        spool = tmp_path / "spool"
        spool.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(spool))
        app = fx.make_app(artifact, threads=2)  # one slot
        slots = app.extensions["bwt_replay_slots"]
        builder = EnvironBuilder(
            path=STREAM, method="POST", query_string="speed=20",
            data={"file": (io.BytesIO(edfs["five"].read_bytes()), "rec.edf")})
        try:
            statuses = []
            body = app.wsgi_app(builder.get_environ(),
                                lambda status, headers, exc_info=None: statuses.append(status))
            assert statuses and statuses[0].startswith("200")
            assert slots._value == 0, "the replay holds its slot until closed"
            assert len(list(spool.iterdir())) == 1, "the upload is spooled"
            body.close()
            assert slots._value == 1, "closing without iterating leaked the slot"
            assert list(spool.iterdir()) == [], "closing without iterating leaked the upload"
            body.close()  # a second close is harmless
            assert slots._value == 1
        finally:
            builder.close()

    def test_a_crowd_of_paced_replays_never_breaks_the_server(self, artifact, edfs):
        import requests

        outcomes, lock = [], threading.Lock()
        with fx.LiveServer(fx.make_app(artifact, threads=4), threads=4) as server:
            def viewer():
                with open(edfs["short10"], "rb") as fh:
                    response = requests.post(f"{server.url}{STREAM}?speed=4",
                                             files={"file": fh}, timeout=180)
                lines = [json.loads(x) for x in response.text.splitlines() if x]
                last = lines[-1] if lines else {}
                with lock:
                    outcomes.append((response.status_code,
                                     last.get("type") or last.get("error")))

            crowd = [threading.Thread(target=viewer) for _ in range(12)]
            for t in crowd:
                t.start()
            time.sleep(0.5)
            assert requests.get(server.url + "/healthz", timeout=5).ok
            for t in crowd:
                t.join(timeout=240)

            assert len(outcomes) == 12
            assert {status for status, _ in outcomes} <= {200, 503}
            assert all(kind == "end" for status, kind in outcomes if status == 200)
            assert all(kind == "service_unavailable"
                       for status, kind in outcomes if status == 503)
            assert sum(status == 200 for status, _ in outcomes) >= 3

            # Afterwards the whole budget is free again: three holds succeed
            # and the fourth is the one refused.
            held = []
            try:
                for _ in range(3):
                    fh = open(edfs["good"], "rb")  # noqa: SIM115 - held open on purpose
                    response = requests.post(f"{server.url}{STREAM}?speed=0.25",
                                             files={"file": fh}, timeout=60,
                                             stream=True)
                    held.append((fh, response))
                    assert response.status_code == 200
                    next(response.iter_lines())
                with open(edfs["five"], "rb") as fh:
                    fourth = requests.post(f"{server.url}{STREAM}?speed=0.25",
                                           files={"file": fh}, timeout=30)
                assert fourth.status_code == 503
                assert requests.get(server.url + "/healthz", timeout=5).ok
            finally:
                for fh, response in held:
                    response.close()
                    fh.close()


# --------------------------------------------------------------------------- #
# Numeric query parameters: exactly parsed, or refused
# --------------------------------------------------------------------------- #


class TestNumericQueryParameters:
    @pytest.mark.parametrize(("raw", "kind", "expected"), [
        ("5", int, 5), ("+7", int, 7), ("-3", int, -3), ("007", int, 7),
        (" 12 ", int, 12), ("0.5", float, 0.5), (".5", float, 0.5),
        ("5.", float, 5.0), ("5E-1", float, 0.5), ("+1e+1", float, 10.0),
        ("-0", float, 0.0), ("3", float, 3.0),
    ])
    def test_plain_numbers_parse_exactly(self, app, raw, kind, expected):
        from bwt.serving.app import _number_arg

        with app.test_request_context("/?x=" + quote(raw)):
            value = _number_arg("x", None, kind)
        assert value == expected and type(value) is kind

    @pytest.mark.parametrize("raw", [
        "", " ", "nan", "NaN", "inf", "-inf", "Infinity", "1_000", "0x10", "1e",
        "e5", "--1", "+-1", "1.2.3", "\u0661", "\uff11", "1,5", "5%", "1 2", ".",
        "+", "-", "1e5.5", "0b1", "\u00bd",
    ])
    @pytest.mark.parametrize("kind", [int, float])
    def test_anything_else_is_refused_naming_the_parameter(self, app, raw, kind):
        from bwt.serving.app import _number_arg

        with app.test_request_context("/?threshold=" + quote(raw)), \
                pytest.raises(ValueError, match=r"^threshold must be"):
            _number_arg("threshold", None, kind)

    @pytest.mark.parametrize("raw", ["2.5", "7.0", "1e2"])
    def test_an_integer_parameter_refuses_a_fraction_or_exponent(self, app, raw):
        from bwt.serving.app import _number_arg

        with app.test_request_context("/?max_windows=" + quote(raw)), \
                pytest.raises(ValueError, match="an integer"):
            _number_arg("max_windows", None, int)

    def test_an_absent_parameter_is_the_default(self, app):
        from bwt.serving.app import _number_arg

        with app.test_request_context("/"):
            assert _number_arg("speed", 0.25, float) == 0.25

    @PROPERTY
    @given(text=st.text(st.characters(blacklist_categories=("Cs",)), max_size=12))
    def test_any_text_is_parsed_exactly_or_refused(self, app, text):
        from bwt.serving.app import _number_arg

        for kind in (int, float):
            with app.test_request_context(query_string={"x": text}):
                try:
                    value = _number_arg("x", None, kind)
                except ValueError as exc:
                    assert str(exc).startswith("x must be")
                    continue
            stripped = text.strip()
            assert stripped.isascii() and stripped
            assert "_" not in stripped and "n" not in stripped.lower()
            assert value == kind(stripped)

    @pytest.mark.parametrize("query", [
        "speed=%2B0", "step=5E-1", "threshold=.95", "max_windows=007",
        "speed=0&step=1.&threshold=0.75&max_windows=+3",
    ])
    def test_valid_spellings_are_served(self, client, edfs, query):
        response = post(client, edfs["five"], f"{STREAM}?{query}")
        assert response.status_code == 200
        assert frames(response)[-1]["type"] == "end"

    @pytest.mark.parametrize("query", ["speed=1e999", "step=1e-999", "max_windows=1e3",
                                       "threshold=9e-1x"])
    def test_numbers_that_parse_but_are_out_of_range_or_malformed_are_400(
            self, client, query):
        response = client.post(f"{STREAM}?{query}", content_type="multipart/form-data")
        assert response.status_code == 400
        assert response.get_json()["error"] == "bad_request"


# --------------------------------------------------------------------------- #
# The stream is held to the same contract as a file decode
# --------------------------------------------------------------------------- #


class TestStreamContract:
    @pytest.mark.parametrize("which", ["one", "below", "exact", "above"])
    def test_truncation_is_announced_exactly_when_windows_are_left(
            self, artifact, edfs, predictor, which):
        n = len(predictor.stream_from_edf(edfs["short10"], step_seconds=0.5))
        cap = {"one": 1, "below": n - 1, "exact": n, "above": n + 1}[which]
        client = fx.make_app(artifact, max_epochs=cap).test_client()
        events = frames(post(client, edfs["short10"], STREAM))
        types = [e["type"] for e in events]
        decoded = min(n, cap)
        assert events[0]["n_windows"] == decoded
        assert types.count("window") == decoded
        assert ("truncated" in types) == (cap < n)
        assert events[-1] == {"type": "end", "n_windows": decoded}
        if cap < n:
            assert types[-2] == "truncated"

    def test_wrong_rate_is_refused_in_the_same_words_on_both_paths(self, client,
                                                                   edfs):
        decode = post(client, edfs["rate128"], "/api/v1/predict")
        assert decode.status_code == 400
        message = decode.get_json()["message"]
        assert "128 Hz" in message and "160 Hz" in message
        assert frames(post(client, edfs["rate128"], STREAM)) == [
            {"type": "error", "message": message}]

    def test_the_stream_reads_the_same_samples_the_file_decode_reads(
            self, predictor, edfs):
        from bwt.data.epochs import read_standardised_raw

        stream = predictor.stream_from_edf(edfs["good"], step_seconds=0.25)
        raw = read_standardised_raw(edfs["good"])
        picks = predictor._align_channels(raw.ch_names)
        expected = raw.get_data(picks=picks) * 1e6
        np.testing.assert_array_equal(stream.data, expected)
        assert stream.window_samples == predictor.card.n_times
        assert stream.step_samples == 40 and stream.sfreq == 160.0

    @pytest.mark.parametrize("step", [0.05, 0.0031, 5.0])
    def test_step_is_never_zero_samples(self, predictor, edfs, step):
        stream = predictor.stream_from_edf(edfs["five"], step_seconds=step)
        assert stream.step_samples >= 1


# --------------------------------------------------------------------------- #
# No signal, and estimator refusals
# --------------------------------------------------------------------------- #


class TestNoSignal:
    def test_flat_recording_is_refused_in_the_same_words_on_every_path(self, client,
                                                                       edfs):
        decode = post(client, edfs["flat"], "/api/v1/predict")
        assert decode.status_code == 400
        message = decode.get_json()["message"]
        assert "flat" in message and "Input X" not in message
        assert frames(post(client, edfs["flat"], f"{STREAM}?band_power=1&raw=1")) == [
            {"type": "error", "message": message}]
        page = post(client, edfs["flat"], "/predict")
        assert page.status_code == 400 and "flat" in page.get_data(as_text=True)

    @pytest.mark.parametrize(("name", "data", "refused"), [
        ("zeros", np.zeros((64, 800)), True),
        ("dc_offset", np.full((64, 800), 50.0), True),
        ("one_live_channel", np.vstack([np.zeros((63, 800)),
                                        np.sin(np.arange(800))[None]]), False),
        ("tiny_noise", np.random.default_rng(0).normal(0, 1e-3, (64, 800)), False),
        ("one_sample", np.zeros((64, 1)), False),
        ("no_samples", np.zeros((64, 0)), False),
    ])
    def test_only_a_wholly_flat_recording_is_refused(self, name, data, refused):
        from bwt.serving.predictor import InputContractError, Predictor

        if refused:
            with pytest.raises(InputContractError, match="flat"):
                Predictor._require_signal(data)
        else:
            Predictor._require_signal(data)

    @PROPERTY
    @given(channel=st.integers(0, 63), amplitude=st.floats(1e-5, 1e4),
           offset=st.floats(-1e4, 1e4))
    def test_any_single_moving_channel_is_enough(self, channel, amplitude, offset):
        from bwt.serving.predictor import Predictor

        data = np.full((64, 200), offset)
        data[channel, 100] += amplitude
        Predictor._require_signal(data)


class TestEstimatorRefusals:
    @pytest.fixture
    def rigged(self, artifact):
        app = fx.make_app(artifact)
        return app, app.extensions["bwt_predictor"]

    INTERNAL = "Input X contains infinity or a value too large for dtype('float64')."

    def test_an_estimator_value_error_becomes_a_contract_error_without_internals(
            self, rigged, edfs, monkeypatch):
        app, predictor = rigged

        def refuse(X):
            raise ValueError(self.INTERNAL)

        monkeypatch.setattr(predictor.model, "predict_proba", refuse)
        client = app.test_client()

        decode = post(client, edfs["good"], "/api/v1/predict")
        assert decode.status_code == 400
        assert decode.get_json()["error"] == "input_contract"
        stream = post(client, edfs["good"], STREAM).get_data(as_text=True)
        form = post(client, edfs["good"], "/predict").get_data(as_text=True)
        for text in (decode.get_data(as_text=True), stream, form):
            assert "Input X" not in text and "dtype" not in text
        events = [json.loads(x) for x in stream.splitlines() if x]
        assert events[0]["type"] == "start" and events[-1]["type"] == "error"

    def test_a_crash_inside_the_estimator_is_still_a_generic_500(self, rigged, edfs,
                                                                 monkeypatch):
        app, predictor = rigged

        def crash(X):
            raise RuntimeError(r"native failure at C:\secret\lib.dll")

        monkeypatch.setattr(predictor.model, "predict_proba", crash)
        client = app.test_client()
        decode = post(client, edfs["good"], "/api/v1/predict")
        assert decode.status_code == 500
        assert decode.get_json()["error"] == "internal_error"
        stream = post(client, edfs["good"], STREAM).get_data(as_text=True)
        assert "secret" not in decode.get_data(as_text=True) + stream
        assert json.loads(stream.splitlines()[-1]) == {
            "type": "error", "message": "the recording could not be decoded"}

    def test_the_inference_lock_is_free_after_a_refusal(self, rigged, edfs,
                                                        monkeypatch):
        app, predictor = rigged
        real = predictor.model.predict_proba
        calls = []

        def once(X):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError(self.INTERNAL)
            return real(X)

        monkeypatch.setattr(predictor.model, "predict_proba", once)
        client = app.test_client()
        assert post(client, edfs["good"], "/api/v1/predict").status_code == 400
        assert not predictor._lock.locked()
        done = []
        worker = threading.Thread(target=lambda: done.append(
            post(app.test_client(), edfs["good"], "/api/v1/predict").status_code))
        worker.start()
        worker.join(timeout=60)
        assert done == [200]


class TestNonFiniteSamples:
    @PROPERTY
    @given(n=st.integers(1, 3), channel=st.integers(0, 63), sample=st.integers(0, 480),
           value=st.sampled_from([math.nan, math.inf, -math.inf, 1e39, -1e300]))
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_one_bad_sample_anywhere_is_refused(self, predictor, n, channel, sample,
                                                value):
        from bwt.serving.predictor import InputContractError

        X = np.zeros((n, 64, 481))
        X[n - 1, channel, sample] = value
        with pytest.raises(InputContractError, match="NaN or infinite"):
            predictor.validate_array(X)
        with pytest.raises(InputContractError):
            predictor.predict_array(X)

    def test_the_largest_finite_float32_is_accepted(self, predictor):
        X = np.zeros((1, 64, 481), np.float32)
        X[0, 0, 0] = np.finfo(np.float32).max
        X[0, 1, 0] = np.finfo(np.float32).min
        assert predictor.validate_array(X) is not None


# --------------------------------------------------------------------------- #
# Damaged files
# --------------------------------------------------------------------------- #


HEADER = 256 * (1 + 64)
DAMAGED = {
    "empty": lambda good: b"",
    "eight_bytes": lambda good: good[:8],
    "half_header": lambda good: good[: HEADER // 2],
    "header_only": lambda good: good[:HEADER],
}


class TestDamagedFiles:
    @pytest.mark.parametrize("shape", sorted(DAMAGED))
    def test_every_damaged_shape_is_a_value_error_that_keeps_its_cause(
            self, tmp_path, edfs, shape):
        from bwt.data.epochs import read_standardised_raw

        path = tmp_path / "damaged.edf"
        path.write_bytes(DAMAGED[shape](edfs["short10"].read_bytes()))
        with pytest.raises(ValueError, match="not a readable EDF") as info:
            read_standardised_raw(path)
        assert info.value.__cause__ is not None

    def test_a_missing_file_is_still_a_missing_file(self, tmp_path):
        from bwt.data.epochs import read_standardised_raw

        with pytest.raises(FileNotFoundError):
            read_standardised_raw(tmp_path / "nowhere.edf")

    @pytest.mark.parametrize("shape", ["eight_bytes", "half_header", "header_only"])
    def test_damaged_uploads_are_client_errors_on_every_path(self, client, edfs,
                                                             shape):
        payload = DAMAGED[shape](edfs["short10"].read_bytes())
        for url in ("/api/v1/predict", "/predict"):
            response = post(client, payload, url)
            assert 400 <= response.status_code < 500, (url, response.status_code)
        assert frames(post(client, payload, STREAM)) == [
            {"type": "error", "message": "the recording could not be decoded"}]


# --------------------------------------------------------------------------- #
# Evidence accumulator guards
# --------------------------------------------------------------------------- #


pair = st.tuples(st.floats(0, 1), st.floats(0, 1))


class TestAccumulatorGuards:
    @PROPERTY
    @given(before=st.lists(pair, max_size=15), after=st.lists(pair, max_size=15),
           slot=st.integers(0, 1),
           bad=st.sampled_from([math.nan, math.inf, -math.inf]))
    def test_a_refused_window_leaves_no_trace(self, before, after, slot, bad):
        from bwt.streaming import EvidenceAccumulator

        def make():
            return EvidenceAccumulator(["x", "y"], threshold=0.99,
                                       max_windows=10_000, leak=0.9)

        seen, clean = make(), make()
        for p in before:
            assert (seen.update(p) is None) == (clean.update(p) is None)
        evidence, count = seen.log_evidence.copy(), seen.n_windows
        window = [0.5, 0.5]
        window[slot] = bad
        with pytest.raises(ValueError, match="finite"):
            seen.update(window)
        assert seen.n_windows == count
        np.testing.assert_array_equal(seen.log_evidence, evidence)
        for p in after:
            assert (seen.update(p) is None) == (clean.update(p) is None)
        np.testing.assert_allclose(seen.posterior, clean.posterior)

    @pytest.mark.parametrize(("max_windows", "min_windows"),
                             [(1, 1), (5, 5), (2, 1), (1, 0), (1, -4), (10_000, 1)])
    def test_every_possible_budget_is_accepted(self, max_windows, min_windows):
        from bwt.streaming import EvidenceAccumulator

        acc = EvidenceAccumulator(["x", "y"], threshold=0.9,
                                  max_windows=max_windows, min_windows=min_windows)
        assert 1 <= acc.min_windows <= acc.max_windows

    @pytest.mark.parametrize("p", [0.6, 0.75, 0.8, 0.9, 0.97, 0.999])
    def test_a_posterior_exactly_at_the_threshold_commits(self, p):
        from bwt.streaming import EvidenceAccumulator

        def make(threshold):
            return EvidenceAccumulator(["x", "y"], threshold=threshold, max_windows=10,
                                       min_windows=1, leak=1.0)

        probe = make(0.5000001)
        probe.update([p, 1 - p])
        reached = float(probe.posterior.max())
        # "Commit when one class reaches the threshold": equal is enough...
        decision = make(reached).update([p, 1 - p])
        assert decision is not None and not decision.timed_out
        assert decision.confidence == reached
        # ...and the next representable threshold above it is not.
        assert make(float(np.nextafter(reached, 1.0))).update([p, 1 - p]) is None

    def test_a_budget_of_one_decides_on_the_first_window(self):
        from bwt.streaming import EvidenceAccumulator

        acc = EvidenceAccumulator(["x", "y"], threshold=0.9, max_windows=1,
                                  min_windows=1)
        decision = acc.update([0.95, 0.05])
        assert decision is not None and decision.label == "x"
        assert not decision.timed_out

    @pytest.mark.parametrize(("max_windows", "min_windows", "expected"),
                             [(1, 2, 1), (5, 2, 2), (40, 2, 2), (3, 10, 3), (1, 1, 1)])
    def test_the_decoder_fits_its_minimum_inside_the_budget(self, max_windows,
                                                           min_windows, expected):
        from types import SimpleNamespace

        from bwt.streaming import StreamingDecoder

        fake = SimpleNamespace(card=SimpleNamespace(classes=["x", "y"]))
        decoder = StreamingDecoder(fake, max_windows=max_windows,
                                   min_windows=min_windows)
        assert decoder.accumulator.min_windows == expected

    @pytest.mark.parametrize("max_windows", [0, -1])
    def test_the_decoder_still_refuses_an_empty_budget(self, max_windows):
        from types import SimpleNamespace

        from bwt.streaming import StreamingDecoder

        fake = SimpleNamespace(card=SimpleNamespace(classes=["x", "y"]))
        with pytest.raises(ValueError, match="max_windows"):
            StreamingDecoder(fake, max_windows=max_windows)

    def test_budget_one_over_http_commits_on_single_windows(self, client, edfs):
        events = frames(post(client, edfs["good"], f"{STREAM}?max_windows=1"))
        decisions = [e["decision"] for e in events if e.get("decision")]
        windows = [e for e in events if e.get("type") == "window"]
        assert len(decisions) == len(windows)
        assert all(d["n_windows"] == 1 for d in decisions)
        for d in decisions:
            assert d["timed_out"] == (d["confidence"] < 0.9 - 1e-4)


# --------------------------------------------------------------------------- #
# Small pure functions
# --------------------------------------------------------------------------- #


class TestInformationTransferRate:
    @pytest.mark.parametrize(("accuracy", "seconds"), [
        (math.nan, 3.0), (math.inf, 3.0), (-math.inf, 3.0), (0.8, math.nan),
        (0.8, math.inf), (math.nan, math.nan),
    ])
    def test_non_finite_inputs_give_zero_never_nan(self, accuracy, seconds):
        from bwt.metrics import information_transfer_rate

        assert information_transfer_rate(accuracy, 4, seconds) == 0.0

    @pytest.mark.parametrize("corrupt", [math.nan, math.inf, -math.inf, "0.8", None,
                                         True, [0.8]])
    def test_a_card_with_a_corrupt_accuracy_serves_valid_json_everywhere(
            self, artifact, corrupt):
        app = fx.make_app(artifact)
        card = app.extensions["bwt_predictor"].card
        for section in ("within_subject", "cross_subject"):
            card.evaluation[section] = {"mean_accuracy": corrupt,
                                        "std_accuracy": corrupt}

        note = app.extensions["bwt_predictor"].performance_note()
        json.dumps(note, allow_nan=False)
        assert note["within_subject_accuracy"] is None
        assert note["cross_subject_accuracy"] is None
        assert note["within_subject_std"] is None
        assert note["itr_bits_per_minute"] == 0.0

        client = app.test_client()
        response = client.get("/api/v1/model")
        assert response.status_code == 200
        json.loads(response.get_data(as_text=True),
                   parse_constant=lambda c: pytest.fail(f"invalid JSON constant {c}"))
        for path in ("/", "/live"):
            page = client.get(path)
            assert page.status_code == 200
            assert "nan%" not in page.get_data(as_text=True).lower()

    def test_a_finite_accuracy_is_passed_through_unchanged(self, predictor):
        note = predictor.performance_note()
        assert note["within_subject_accuracy"] == 0.9
        assert note["cross_subject_accuracy"] == 0.8
        assert note["itr_bits_per_minute"] > 0


class TestSubjectSpecifications:
    @pytest.mark.parametrize(("text", "expected"), [
        (" 5 - 7 ", [5, 6, 7]), ("1-109", list(range(1, 110))), ("3,3,3", [3]),
        ("10-12,1", [1, 10, 11, 12]),
    ])
    def test_tolerant_spellings_still_parse(self, text, expected):
        from bwt.cli import _subject_list

        assert _subject_list(text) == expected

    @pytest.mark.parametrize(("text", "needle"), [
        ("9-5", "backwards"), ("1--3", "backwards"), (",", "selects no subjects"),
        (" , , ", "selects no subjects"),
    ])
    def test_refusals_say_what_is_wrong(self, text, needle):
        from bwt.cli import _subject_list

        with pytest.raises(ValueError, match=needle):
            _subject_list(text)


class TestConfigFromTheEnvironment:
    @pytest.mark.parametrize(("raw", "expected"), [
        ("cross_subject", ("cross_subject",)),
        ("within_subject, cross_subject", ("within_subject", "cross_subject")),
        (" within_subject ,, ", ("within_subject",)),
        ("", ()),
    ])
    def test_tuple_settings_split_on_commas(self, monkeypatch, raw, expected):
        from bwt.config import Config

        monkeypatch.setenv("BWT_TRAIN_PROTOCOLS", raw)
        assert Config().with_env().train.protocols == expected

    def test_threads_can_be_set_and_size_the_replay_budget(self, monkeypatch,
                                                           artifact, edfs):
        from bwt.config import Config
        from bwt.serving.app import create_app
        from bwt.serving.predictor import Predictor

        monkeypatch.setenv("BWT_SERVE_THREADS", "2")
        config = Config().with_env()
        assert config.serve.threads == 2
        client = create_app(config, predictor=Predictor.load(artifact)).test_client()
        holder = post(client, edfs["five"], f"{STREAM}?speed=20")
        try:
            assert post(client, edfs["five"], f"{STREAM}?speed=20").status_code == 503
        finally:
            holder.close()

    def test_the_default_file_documents_every_serve_setting(self):
        from dataclasses import fields

        import yaml

        from bwt.config import ServeConfig
        from bwt.paths import configs_dir

        documented = yaml.safe_load(
            (configs_dir() / "default.yaml").read_text(encoding="utf-8"))["serve"]
        assert {f.name for f in fields(ServeConfig)} == set(documented)
        assert documented["threads"] == ServeConfig().threads


# --------------------------------------------------------------------------- #
# Upload limit on the page and at the server
# --------------------------------------------------------------------------- #


class TestUploadLimit:
    def test_the_page_states_the_configured_limit(self, client, artifact):
        html = client.get("/").get_data(as_text=True)
        assert 'data-max-mb="2"' in html and "max 2 MB" in html
        default = fx.make_app(artifact).test_client().get("/").get_data(as_text=True)
        assert 'data-max-mb="64"' in default and "max 64 MB" in default

    @pytest.mark.parametrize(("url", "html"), [
        ("/predict", True), ("/api/v1/predict", False), (STREAM, False),
    ])
    def test_oversized_uploads_get_the_right_kind_of_answer(self, client, url, html):
        response = post(client, b"\x00" * (3 * 1024 * 1024), url)
        assert response.status_code == 413
        if html:
            assert response.mimetype == "text/html"
            assert "Rejected" in response.get_data(as_text=True)
        else:
            assert response.get_json() == {
                "error": "payload_too_large", "message": "upload exceeds the 2 MB limit"}

    def test_just_under_the_limit_is_not_refused_for_size(self, client):
        response = post(client, b"\x00" * (2 * 1024 * 1024 - 4096), "/api/v1/predict")
        assert response.status_code == 400
        assert response.get_json()["error"] == "bad_request"
