"""The command-line stream and the HTTP stream must be the same decoder.

``bwt stream`` drives :meth:`StreamingDecoder.run`; ``/api/v1/stream`` drives
its own generator in ``bwt.serving.app``. They are separate loops over the same
components, so they can drift -- a reset in one place and not the other, a
different dtype, a different order of accumulate and reset -- and a user would
get different decisions for the same recording depending on how they asked.
Coverage showed no test executed the command-line path at all.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

import _factories as fx

STEP, THRESHOLD, MAX_WINDOWS = 0.5, 0.9, 40


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    return fx.build_artifact(tmp_path_factory.mktemp("cli") / "lr")


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    return fx.write_edf(tmp_path_factory.mktemp("cli_edf") / "rec.edf", duration=40.0)


@pytest.fixture(scope="module")
def predictor(artifact):
    from bwt.serving.predictor import Predictor

    return Predictor.load(artifact)


def untimed(payload):
    """An event or decision without its wall-clock timing, which differs
    between any two runs and is not part of what was decoded."""
    if isinstance(payload, dict):
        return {k: untimed(v) for k, v in payload.items() if k != "elapsed_seconds"}
    if isinstance(payload, list):
        return [untimed(v) for v in payload]
    return payload


def http_windows(artifact, recording, **params):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    client = fx.make_app(artifact).test_client()
    response = client.post(f"/api/v1/stream?{query}",
                           data={"file": (io.BytesIO(recording.read_bytes()), "rec.edf")},
                           content_type="multipart/form-data")
    frames = [json.loads(x) for x in response.get_data(as_text=True).splitlines() if x]
    assert frames[-1]["type"] == "end"
    return [f for f in frames if f["type"] == "window"]


def cli_events(predictor, recording, *, max_windows=MAX_WINDOWS, on_event=None):
    stream = predictor.stream_from_edf(recording, step_seconds=STEP)
    decoder = predictor.streaming_decoder(threshold=THRESHOLD, max_windows=max_windows)
    return decoder.run(stream, on_event=on_event)


class TestParity:
    def test_every_window_and_every_decision_is_identical(self, artifact, recording,
                                                          predictor):
        http = http_windows(artifact, recording, speed=0, step=STEP, threshold=THRESHOLD,
                            max_windows=MAX_WINDOWS)
        cli = [event.to_dict() for event in cli_events(predictor, recording)]
        assert len(cli) == len(http) > 50
        for c, h in zip(cli, http, strict=True):
            assert c["window"] == h["window"]
            assert c["onset_seconds"] == h["onset_seconds"]
            assert c["probabilities"] == h["probabilities"]
            assert c["top_label"] == h["top_label"]
            assert c["posterior"] == h["posterior"]
            assert c.get("decision", {}).get("label") == h.get("decision", {}).get("label")
            if "decision" in h:
                for key in ("confidence", "n_windows", "timed_out", "posterior"):
                    assert c["decision"][key] == h["decision"][key], key
        assert any("decision" in h and not h["decision"]["timed_out"] for h in http)

    def test_a_budget_of_one_is_identical_on_both_paths(self, artifact, recording,
                                                        predictor):
        http = http_windows(artifact, recording, speed=0, step=STEP, threshold=THRESHOLD,
                            max_windows=1)
        cli = [e.to_dict() for e in cli_events(predictor, recording, max_windows=1)]
        assert untimed([c["decision"] for c in cli]) == untimed([h["decision"] for h in http])
        assert all(c["decision"]["n_windows"] == 1 for c in cli)

    def test_the_callback_sees_every_event_in_order(self, predictor, recording):
        seen = []
        events = cli_events(predictor, recording, on_event=seen.append)
        assert seen == events
        assert [e.window for e in events] == list(range(len(events)))

    def test_run_is_repeatable_on_one_decoder(self, predictor, recording):
        stream = predictor.stream_from_edf(recording, step_seconds=STEP)
        decoder = predictor.streaming_decoder(threshold=THRESHOLD, max_windows=MAX_WINDOWS)
        first = untimed([e.to_dict() for e in decoder.run(stream)])
        second = untimed([e.to_dict() for e in decoder.run(stream)])
        assert first == second


class TestCommand:
    def test_the_command_prints_the_same_counts_the_api_computes(self, artifact,
                                                                 recording, capsys):
        from bwt.cli import main

        http = http_windows(artifact, recording, speed=0, step=STEP, threshold=THRESHOLD,
                            max_windows=MAX_WINDOWS)
        committed = sum(1 for h in http if "decision" in h and not h["decision"]["timed_out"])
        timed_out = sum(1 for h in http if "decision" in h and h["decision"]["timed_out"])

        code = main(["stream", str(recording), "--model", str(artifact),
                     "--step", str(STEP), "--threshold", str(THRESHOLD)])
        out = capsys.readouterr().out
        assert code == 0
        assert f"windows  : {len(http)} (step {STEP}s, speed max)" in out
        assert f"decisions: {committed} committed, {timed_out} timed out" in out
        assert out.count(" after ") == committed + timed_out

    def test_the_command_refuses_a_wrong_rate_with_the_contract_error(self, artifact,
                                                                      tmp_path):
        from bwt.cli import cmd_stream
        from bwt.serving.predictor import InputContractError

        path = fx.write_edf(tmp_path / "rate.edf", duration=20.0, sfreq=128.0)
        args = SimpleNamespace(model=str(artifact), file=str(path), speed=0.0, step=STEP,
                               threshold=THRESHOLD, max_windows=MAX_WINDOWS)
        with pytest.raises(InputContractError, match="128 Hz"):
            cmd_stream(args)


class TestEDFStreamFromFile:
    def test_from_edf_reads_what_the_predictor_reads(self, predictor, recording):
        from bwt.data.epochs import read_standardised_raw
        from bwt.streaming import EDFStream

        raw = read_standardised_raw(recording)
        picks = predictor._align_channels(raw.ch_names)
        direct = EDFStream.from_edf(recording, predictor.card, step_seconds=STEP,
                                    picks=picks)
        served = predictor.stream_from_edf(recording, step_seconds=STEP)
        np.testing.assert_array_equal(direct.data, served.data)
        assert (direct.window_samples, direct.step_samples) == \
            (served.window_samples, served.step_samples)
        assert len(direct) == len(served)

    def test_from_edf_refuses_a_wrong_rate(self, predictor, tmp_path):
        from bwt.streaming import EDFStream

        path = fx.write_edf(tmp_path / "rate.edf", duration=20.0, sfreq=128.0)
        with pytest.raises(ValueError, match="128 Hz"):
            EDFStream.from_edf(path, predictor.card)
