"""Adversarial tests for the HTTP layer.

Every test here tries to make the service do something it should not: crash,
emit invalid JSON, leak a temp file or an internal detail, silently accept a
recording that breaks the input contract, lie in its streaming protocol, or stop
answering other users. Assertions state the *correct* behaviour. Where the
service currently gets it wrong, the test is marked ``xfail(strict=True)`` with
the defect spelled out, so the suite stays green while the bug is tracked -- and
turns red the moment a fix lands without the marker being removed.

Recordings and models are synthetic (see ``tests/_factories.py``), so none of
this needs the dataset.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import _factories as fx
from bwt.serving.app import TRACE_CHANNELS
from conftest import EEGBCI_CHANNELS

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    return {
        "lr": fx.build_artifact(root / "lr", pipeline="csp_lda",
                                task="mi_left_right"),
        "four": fx.build_artifact(root / "four", pipeline="riemann_ts_aligned",
                                  task="mi_four_class"),
    }


@pytest.fixture(scope="module")
def edfs(tmp_path_factory):
    """Every recording shape the suite throws at the service."""
    import mne

    d = tmp_path_factory.mktemp("edfs")
    files = {
        "good": fx.write_edf(d / "good.edf", duration=60.0),
        "short10": fx.write_edf(d / "short10.edf", duration=10.0),
        "five": fx.write_edf(d / "five.edf", duration=5.0, cues=[]),
        "no_cues": fx.write_edf(d / "no_cues.edf", duration=40.0, cues=[]),
        "only_rest": fx.write_edf(
            d / "only_rest.edf", duration=30.0,
            cues=[(t, "T0") for t in (0.0, 4.2, 8.4, 12.6)]),
        "rate128": fx.write_edf(d / "rate128.edf", duration=30.0, sfreq=128.0),
        "rate250": fx.write_edf(d / "rate250.edf", duration=30.0, sfreq=250.0),
        "missing_c3": fx.write_edf(
            d / "missing_c3.edf", duration=30.0,
            ch_names=[c for c in EEGBCI_CHANNELS if c != "C3"]),
        "extra": fx.write_edf(
            d / "extra.edf", duration=30.0,
            ch_names=[*EEGBCI_CHANNELS, "X1", "X2"]),
        "too_short": fx.write_edf(d / "too_short.edf", duration=2.0,
                                  cues=[(0.0, "T1")]),
        # Two cues whose 0.5-3.5 s window runs past the end of the file,
        # between two that fit.
        "cue_past_end": fx.write_edf(
            d / "cue_past_end.edf", duration=20.0,
            cues=[(1.0, "T1"), (6.0, "T2"), (17.5, "T1"), (19.0, "T2")]),
        "flat": fx.write_edf(d / "flat.edf", duration=30.0, signal="flat"),
        "huge": fx.write_edf(d / "huge.edf", duration=30.0,
                             signal="noise", amplitude_uv=3000.0),
    }
    # Same samples as "good", channels written in a shuffled order: decoding
    # must be identical, which only holds if alignment reorders by name.
    raw = mne.io.read_raw_edf(files["good"], preload=True, verbose="ERROR")
    order = list(raw.ch_names)
    np.random.default_rng(3).shuffle(order)
    raw.reorder_channels(order)
    files["shuffled"] = d / "shuffled.edf"
    mne.export.export_raw(str(files["shuffled"]), raw, fmt="edf",
                          overwrite=True, verbose="ERROR")
    return files


@pytest.fixture(scope="module")
def app(artifacts):
    app = fx.make_app(artifacts["lr"], max_upload_mb=2)
    app.config.update(TESTING=True)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(scope="module")
def four_client(artifacts):
    app = fx.make_app(artifacts["four"])
    app.config.update(TESTING=True)
    return app.test_client()


def upload(client, path_or_bytes, *, filename="rec.edf", url="/api/v1/predict",
           **kwargs):
    payload = (Path(path_or_bytes).read_bytes()
               if isinstance(path_or_bytes, Path) else path_or_bytes)
    return client.post(
        url, data={"file": (io.BytesIO(payload), filename)},
        content_type="multipart/form-data", **kwargs,
    )


def _reject_constant(name):
    raise ValueError(f"non-standard JSON constant {name}")


def strict_json(text: str | bytes):
    """Parse JSON the way a browser does: NaN and Infinity are errors."""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return json.loads(text, parse_constant=_reject_constant)


def frames(response) -> list[dict]:
    body = response.get_data(as_text=True)
    assert body.endswith("\n"), "every NDJSON frame must be newline-terminated"
    return [strict_json(line) for line in body.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Routing, methods and path handling
# --------------------------------------------------------------------------- #


class TestRouting:
    @pytest.mark.parametrize(("method", "path", "status"), [
        ("get", "/api/v1/predict", 405),
        ("put", "/api/v1/predict", 405),
        ("delete", "/api/v1/predict", 405),
        ("patch", "/api/v1/stream", 405),
        ("get", "/api/v1/stream", 405),
        ("post", "/api/v1/model", 405),
        ("get", "/api/v1/does-not-exist", 404),
        ("get", "/api/v2/predict", 404),
    ])
    def test_api_errors_are_json_with_the_right_status(self, client, method,
                                                       path, status):
        response = getattr(client, method)(path)
        assert response.status_code == status
        assert response.is_json, response.get_data(as_text=True)[:200]
        body = response.get_json()
        assert set(body) >= {"error", "message"}
        assert "Traceback" not in json.dumps(body)

    def test_wrong_method_on_a_non_api_route_is_still_405(self, client):
        assert client.post("/healthz").status_code == 405
        assert client.post("/live").status_code == 405

    def test_head_and_options_are_answered(self, client):
        assert client.head("/healthz").status_code == 200
        allow = client.options("/api/v1/predict").headers.get("Allow", "")
        assert "POST" in allow and "GET" not in allow

    @pytest.mark.parametrize("path", [
        "/static/../pyproject.toml",
        "/static/%2e%2e/pyproject.toml",
        "/static/..%2fpyproject.toml",
        "/static/..%5cpyproject.toml",
        "/static/..\\pyproject.toml",
        "/static/%2e%2e%2f%2e%2e%2fsrc/bwt/config.py",
        "/static//etc/passwd",
        "/static/C:/Windows/win.ini",
    ])
    def test_static_route_cannot_escape_its_directory(self, client, path):
        response = client.get(path)
        if response.status_code == 308:
            # Werkzeug merges duplicate slashes by redirecting; the target must
            # still be inside /static/ and must itself not resolve.
            location = response.headers["Location"]
            assert "/static/" in location
            response = client.get(location)
        assert response.status_code in {400, 404}, path
        assert b"[project]" not in response.data
        assert b"class ServeConfig" not in response.data

    def test_every_page_and_asset_is_served(self, client):
        for path in ("/", "/live", "/static/app.css", "/static/brain3d.js",
                     "/static/three.module.min.js", "/static/cortex.bin",
                     "/api/v1/geometry", "/api/v1/model", "/healthz"):
            assert client.get(path).status_code == 200, path

    def test_metadata_never_discloses_filesystem_paths(self, client):
        for path in ("/healthz", "/api/v1/model", "/api/v1/geometry"):
            text = client.get(path).get_data(as_text=True)
            for marker in ("C:\\\\", "C:/", "/home/", "/Users/", "AppData",
                           "site-packages"):
                assert marker not in text, (path, marker)

    def test_request_ids_are_unique(self, client, edfs):
        ids = {upload(client, edfs["short10"]).get_json()["request_id"]
               for _ in range(4)}
        assert len(ids) == 4


# --------------------------------------------------------------------------- #
# Upload validation
# --------------------------------------------------------------------------- #


class TestUploadValidation:
    @pytest.mark.parametrize("filename", [
        "noextension", "rec.edf.txt", "rec.edfx", "rec.", "rec.bdf",
        "rec.edf ", "rec.gdf", "payload.py", ".edf",
    ])
    def test_non_edf_names_are_rejected_before_reading(self, client, edfs,
                                                       filename):
        response = upload(client, edfs["short10"], filename=filename)
        assert response.status_code == 400
        assert response.get_json()["error"] == "bad_request"
        assert "unsupported file type" in response.get_json()["message"]

    @pytest.mark.parametrize("filename", [
        "REC.EDF", "rec.Edf", "a.b.c.edf", "reçu é 雪.edf", "x" * 1000 + ".edf",
        "../../escape.edf", "..\\..\\escape.edf", "CON.edf", "nul.edf",
        "with space ;rm -rf.edf", "rtl\u202eoverride.edf",
    ])
    def test_any_edf_name_decodes_and_never_touches_that_path(self, client,
                                                              edfs, filename):
        from bwt.paths import repo_root

        response = upload(client, edfs["short10"], filename=filename)
        assert response.status_code == 200, response.get_data(as_text=True)
        assert response.get_json()["n_epochs"] > 0
        assert not (repo_root() / "escape.edf").exists()
        assert not (repo_root().parent / "escape.edf").exists()

    def test_missing_file_field(self, client):
        response = client.post("/api/v1/predict", data={},
                               content_type="multipart/form-data")
        assert response.status_code == 400
        assert "'file'" in response.get_json()["message"]

    def test_file_under_the_wrong_field_name(self, client, edfs):
        response = client.post(
            "/api/v1/predict",
            data={"upload": (io.BytesIO(edfs["short10"].read_bytes()), "r.edf")},
            content_type="multipart/form-data")
        assert response.status_code == 400

    def test_empty_filename(self, client):
        response = upload(client, b"data", filename="")
        assert response.status_code == 400

    def test_zero_byte_edf(self, client):
        response = upload(client, b"")
        assert response.status_code == 400
        assert "empty" in response.get_json()["message"]

    @pytest.mark.parametrize(("body", "content_type"), [
        (b'{"file": "x"}', "application/json"),
        (b"file=x.edf", "application/x-www-form-urlencoded"),
        (b"\x00\x01\x02", "application/octet-stream"),
        (b"--broken\r\nnot really multipart", "multipart/form-data; boundary=broken"),
        (b"", "multipart/form-data"),
    ])
    def test_non_multipart_bodies_are_a_clean_400(self, client, body,
                                                  content_type):
        response = client.post("/api/v1/predict", data=body,
                               content_type=content_type)
        assert response.status_code == 400
        assert response.is_json

    def test_two_files_in_one_field_decodes_the_first(self, client, edfs):
        response = client.post(
            "/api/v1/predict",
            data={"file": [(io.BytesIO(edfs["short10"].read_bytes()), "a.edf"),
                           (io.BytesIO(b"garbage"), "b.edf")]},
            content_type="multipart/form-data")
        assert response.status_code == 200

    def test_just_over_the_size_limit_is_413_json(self, client):
        response = upload(client, b"\x00" * (2 * 1024 * 1024 + 1))
        assert response.status_code == 413
        assert response.get_json()["error"] == "payload_too_large"

    def test_just_under_the_size_limit_is_read_not_refused(self, client):
        # Leave room for multipart framing, which counts against the limit.
        response = upload(client, b"\x00" * (2 * 1024 * 1024 - 4096))
        assert response.status_code == 400
        assert response.get_json()["error"] != "payload_too_large"

    def test_oversized_form_upload_renders_html(self, client):
        response = upload(client, b"\x00" * (3 * 1024 * 1024), url="/predict")
        assert response.status_code == 413
        assert response.mimetype == "text/html"
        html = response.get_data(as_text=True)
        assert "Rejected" in html and "2 MB limit" in html

    XSS = 'x.<img src=x onerror="alert(document.domain)">'

    def test_reflected_filename_is_escaped_in_html(self, client, edfs):
        response = upload(client, edfs["short10"], filename=self.XSS,
                          url="/predict")
        assert response.status_code == 400
        html = response.get_data(as_text=True)
        assert "<img src=x" not in html
        assert "&lt;img src=x" in html

    def test_reflected_filename_is_inert_in_json(self, client, edfs):
        response = upload(client, edfs["short10"], filename=self.XSS)
        assert response.mimetype == "application/json"
        assert response.status_code == 400

    @pytest.mark.parametrize("case", [
        "one_byte", "text", "null_bytes", "header_then_garbage", "bit_flipped",
        "html", "zip_magic",
    ])
    def test_garbage_with_an_edf_name_is_4xx_never_5xx(self, client, edfs, case):
        good = edfs["short10"].read_bytes()
        payload = {
            "one_byte": b"\x00",
            "text": b"hello world " * 400,
            "null_bytes": b"\x00" * 100_000,
            "header_then_garbage": good[:256] + np.random.default_rng(0)
                .integers(0, 256, 8192, dtype=np.uint8).tobytes(),
            "bit_flipped": bytes(b ^ 0xFF for b in good[:20_000]),
            "html": b"<html><script>alert(1)</script></html>" * 50,
            "zip_magic": b"PK\x03\x04" + b"\x00" * 4000,
        }[case]
        response = upload(client, payload)
        assert 400 <= response.status_code < 500, response.get_data(as_text=True)
        text = response.get_data(as_text=True)
        assert "Traceback" not in text
        assert "bwt_upload_" not in text, "temp path disclosed"
        assert "site-packages" not in text

    @pytest.mark.parametrize("url", ["/api/v1/predict", "/predict"])
    def test_header_only_edf_is_a_client_error(self, client, edfs, url):
        header_bytes = 256 * (1 + 64)  # fixed header + one per signal
        response = upload(client, edfs["short10"].read_bytes()[:header_bytes],
                          url=url)
        assert 400 <= response.status_code < 500

    def test_truncated_edf_decodes_only_what_survives(self, client, edfs):
        whole = edfs["good"].read_bytes()
        full = upload(client, whole).get_json()["n_epochs"]
        response = upload(client, whole[: len(whole) // 2])
        assert response.status_code in {200, 400}
        if response.status_code == 200:
            assert 0 < response.get_json()["n_epochs"] <= full


# --------------------------------------------------------------------------- #
# The decode contract, end to end through the HTTP layer
# --------------------------------------------------------------------------- #


class TestDecodeContract:
    def test_response_is_internally_consistent(self, client, edfs):
        response = upload(client, edfs["good"])
        assert response.status_code == 200
        body = strict_json(response.data)
        preds = body["predictions"]
        k = len(body["classes"])

        assert body["n_epochs"] == len(preds) > 0
        assert body["epoching"] == "cue_locked"
        assert [p["index"] for p in preds] == list(range(len(preds)))
        for p in preds:
            assert p["label"] in body["classes"]
            assert set(p["probabilities"]) == set(body["classes"])
            assert all(0.0 <= v <= 1.0 for v in p["probabilities"].values())
            assert math.isclose(sum(p["probabilities"].values()), 1.0,
                                abs_tol=k * 5e-5 + 1e-9)
            assert math.isclose(p["confidence"], max(p["probabilities"].values()),
                                abs_tol=1e-4)
            assert p["probabilities"][p["label"]] == max(p["probabilities"].values())
            assert p["source"] == "cue_locked"

        counts = {c: sum(p["label"] == c for p in preds) for c in body["classes"]}
        assert counts[body["majority_label"]] == max(counts.values())
        assert math.isclose(body["mean_confidence"],
                            np.mean([p["confidence"] for p in preds]), abs_tol=1e-3)

    def test_epochs_are_cut_at_the_cues_and_decoded_correctly(self, client, edfs):
        body = upload(client, edfs["good"]).get_json()
        cues = [c for c in fx.alternating_cues(60.0) if c[1] in {"T1", "T2"}]
        assert [p["onset_seconds"] for p in body["predictions"]] == pytest.approx(
            [onset for onset, _ in cues], abs=1 / 160)
        truth = fx.cue_labels(cues, body["classes"])
        hits = sum(p["label"] == t for p, t in zip(body["predictions"], truth,
                                                   strict=True))
        assert hits / len(truth) >= 0.9, f"{hits}/{len(truth)}"

    def test_decoding_is_deterministic(self, client, edfs):
        bodies = [upload(client, edfs["good"]).get_json() for _ in range(3)]
        for body in bodies:
            body.pop("request_id")
        assert bodies[0] == bodies[1] == bodies[2]

    def test_channel_order_in_the_file_does_not_matter(self, client, edfs):
        a = upload(client, edfs["good"]).get_json()["predictions"]
        b = upload(client, edfs["shuffled"]).get_json()["predictions"]
        assert [p["label"] for p in a] == [p["label"] for p in b]
        assert [p["confidence"] for p in a] == pytest.approx(
            [p["confidence"] for p in b], abs=2e-4)

    def test_extra_channels_are_ignored(self, client, edfs):
        response = upload(client, edfs["extra"])
        assert response.status_code == 200
        assert response.get_json()["n_epochs"] > 0

    @pytest.mark.parametrize("name", ["rate128", "rate250"])
    def test_wrong_sampling_rate_is_refused_not_resampled(self, client, edfs,
                                                          name):
        response = upload(client, edfs[name])
        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] == "input_contract"
        assert "Hz" in body["message"] and "160" in body["message"]

    def test_missing_channel_is_named(self, client, edfs):
        response = upload(client, edfs["missing_c3"])
        assert response.status_code == 400
        assert "C3" in response.get_json()["message"]

    @pytest.mark.parametrize("name", ["no_cues", "only_rest"])
    def test_without_movement_cues_falls_back_and_says_so(self, client, edfs,
                                                          name):
        body = upload(client, edfs[name]).get_json()
        assert body["epoching"] == "sliding_window"
        assert any("No T1/T2 cue" in w for w in body["warnings"])
        onsets = [p["onset_seconds"] for p in body["predictions"]]
        window = 481 / 160
        assert onsets == pytest.approx([i * window for i in range(len(onsets))],
                                       abs=1e-3)
        duration = 40.0 if name == "no_cues" else 30.0
        assert body["n_epochs"] == int(duration * 160) // 481

    def test_cues_whose_window_overruns_the_file_are_dropped(self, client, edfs):
        body = upload(client, edfs["cue_past_end"]).get_json()
        assert [p["onset_seconds"] for p in body["predictions"]] == pytest.approx(
            [1.0, 6.0])

    def test_recording_too_short_for_one_window(self, client, edfs):
        response = upload(client, edfs["too_short"])
        assert response.status_code == 400
        assert "too short" in response.get_json()["message"]

    def test_extreme_amplitude_stays_finite(self, client, edfs):
        response = upload(client, edfs["huge"])
        assert response.status_code == 200
        strict_json(response.data)

    @pytest.mark.parametrize("url", ["/api/v1/predict", "/api/v1/stream?band_power=1"])
    def test_flat_line_recording_never_emits_invalid_json(self, client, edfs, url):
        """A disconnected amplifier writes zeros; nothing may become NaN."""
        response = upload(client, edfs["flat"], url=url)
        assert response.status_code in {200, 400}
        for line in response.get_data(as_text=True).splitlines():
            strict_json(line)

    def test_flat_line_recording_is_explained_in_domain_terms(self, client, edfs):
        message = upload(client, edfs["flat"]).get_json()["message"]
        assert "dtype" not in message and "Input X" not in message
        assert "flat" in message.lower() or "no signal" in message.lower()

    def test_epoch_cap_truncates_and_warns(self, artifacts, edfs):
        app = fx.make_app(artifacts["lr"], max_epochs=3)
        body = upload(app.test_client(), edfs["good"]).get_json()
        assert body["n_epochs"] == 3
        assert any("only the first 3" in w for w in body["warnings"])

    def test_transductive_model_warns_below_its_batch_minimum(self, four_client,
                                                              edfs):
        few = upload(four_client, edfs["short10"]).get_json()
        assert few["n_epochs"] < 8
        assert any("recenters" in w for w in few["warnings"])
        many = upload(four_client, edfs["no_cues"]).get_json()
        assert many["n_epochs"] >= 8
        assert not any("recenters" in w for w in many["warnings"])

    def test_form_result_page_matches_the_api(self, client, edfs):
        api = upload(client, edfs["good"]).get_json()
        html = upload(client, edfs["good"], url="/predict").get_data(as_text=True)
        assert html.count("<tr>") - 1 == api["n_epochs"]  # minus the header row
        for p in api["predictions"]:
            assert f"{p['confidence']:.3f}" in html
        assert "Traceback" not in html

    def test_form_rejection_page_is_html_with_a_way_back(self, client):
        response = upload(client, b"\x00" * 5000, url="/predict")
        assert response.status_code == 400
        assert response.mimetype == "text/html"
        html = response.get_data(as_text=True)
        assert "Rejected" in html and 'href="/"' in html


# --------------------------------------------------------------------------- #
# Temp-file hygiene on every path, including abandoned streams
# --------------------------------------------------------------------------- #


class TestTempFileHygiene:
    @pytest.fixture
    def tmpdir_spy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        return lambda: sorted(p.name for p in tmp_path.glob("bwt_upload_*"))

    def test_nothing_survives_any_request_that_completes(self, client, edfs,
                                                         tmpdir_spy):
        upload(client, edfs["short10"])
        upload(client, edfs["short10"], url="/predict")
        upload(client, b"\x00" * 3000)
        upload(client, b"\x00" * 3000, url="/predict")
        upload(client, b"")
        upload(client, edfs["rate128"])
        upload(client, edfs["short10"], url="/api/v1/stream").get_data()
        upload(client, b"\x00" * 3000, url="/api/v1/stream").get_data()
        upload(client, edfs["short10"], url="/api/v1/stream?threshold=7")
        assert tmpdir_spy() == []

    def test_nothing_survives_a_stream_abandoned_mid_way(self, client, edfs,
                                                         tmpdir_spy):
        response = upload(client, edfs["good"], url="/api/v1/stream",
                          buffered=False)
        stream = iter(response.response)
        next(stream)
        next(stream)
        response.close()
        assert tmpdir_spy() == []

    def test_nothing_survives_a_client_that_never_reads_the_stream(
            self, artifacts, edfs, tmpdir_spy):
        """Upload saved, then the client vanishes before the first byte.

        Only a real WSGI server shows this: the upload is written before the
        NDJSON generator starts, and only the generator deletes it.
        """
        import requests

        with fx.LiveServer(fx.make_app(artifacts["lr"])) as server:
            for _ in range(3):
                with open(edfs["good"], "rb") as fh:
                    response = requests.post(server.url + "/api/v1/stream?speed=1",
                                             files={"file": fh}, timeout=60,
                                             stream=True)
                response.close()
            deadline = time.time() + 20
            while tmpdir_spy() and time.time() < deadline:
                time.sleep(0.25)
        assert tmpdir_spy() == []


# --------------------------------------------------------------------------- #
# The NDJSON streaming protocol
# --------------------------------------------------------------------------- #


def stream(client, path, **params):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return upload(client, path, url=f"/api/v1/stream?{query}")


class TestStreamParameters:
    @pytest.mark.parametrize("query", [
        "speed=0", "speed=20", "step=0.05", "step=5", "threshold=0.5001",
        "threshold=0.9999", "max_windows=1", "max_windows=1000",
        "band_power=true&raw=true",
    ])
    def test_boundary_values_are_accepted(self, client, edfs, query):
        response = upload(client, edfs["five"], url=f"/api/v1/stream?{query}")
        assert response.status_code == 200
        assert frames(response)[-1]["type"] == "end"

    @pytest.mark.parametrize("query", [
        "speed=-0.0001", "speed=20.0001", "speed=nan", "speed=inf",
        "speed=-inf", "step=0.0499", "step=5.0001", "step=0", "step=-1",
        "step=nan", "threshold=0.5", "threshold=1.0", "threshold=1",
        "threshold=0", "threshold=nan", "threshold=-inf", "max_windows=0",
        "max_windows=-3", "max_windows=1001", "max_windows=99999999999999999999",
    ])
    def test_out_of_range_values_are_rejected_before_any_upload_is_read(
            self, client, query):
        response = client.post(f"/api/v1/stream?{query}",
                               content_type="multipart/form-data")
        assert response.status_code == 400
        assert response.get_json()["error"] == "bad_request"

    @pytest.mark.parametrize("query", [
        "threshold=abc", "step=1e", "max_windows=2.5", "speed=fast", "speed=",
        "max_windows=1_0", "threshold=0x1", "step=%EF%BC%91", "max_windows=+",
        "threshold=.", "speed=1e5e5", "max_windows=10abc",
    ])
    def test_unparseable_values_are_rejected_not_defaulted(self, client, edfs,
                                                           query):
        response = upload(client, edfs["five"], url=f"/api/v1/stream?{query}")
        assert response.status_code == 400


class TestStreamProtocol:
    @pytest.fixture(scope="class")
    def run(self, artifacts, edfs):
        app = fx.make_app(artifacts["lr"])
        client = app.test_client()
        response = stream(client, edfs["good"], step=0.5, threshold=0.9,
                          band_power=1, raw=1)
        assert response.status_code == 200
        assert response.mimetype == "application/x-ndjson"
        assert response.headers["Cache-Control"] == "no-store"
        from bwt.serving.predictor import Predictor

        return frames(response), Predictor.load(artifacts["lr"])

    def test_frame_grammar(self, run):
        events, _ = run
        types = [e.get("type") for e in events]
        assert types[0] == "start" and types[-1] == "end"
        assert set(types[1:-1]) == {"window"}
        windows = events[1:-1]
        assert [w["window"] for w in windows] == list(range(len(windows)))
        assert events[0]["n_windows"] == events[-1]["n_windows"] == len(windows)
        assert [w["onset_seconds"] for w in windows] == pytest.approx(
            [0.5 * i for i in range(len(windows))], abs=1e-3)

    def test_every_window_is_a_consistent_distribution(self, run):
        events, predictor = run
        classes = events[0]["classes"]
        assert classes == list(predictor.card.classes)
        tol = len(classes) * 5e-5 + 1e-9
        for w in events[1:-1]:
            for key in ("probabilities", "posterior"):
                assert list(w[key]) == classes
                assert math.isclose(sum(w[key].values()), 1.0, abs_tol=tol)
                assert all(0 <= v <= 1 for v in w[key].values())
            assert w["probabilities"][w["top_label"]] == max(w["probabilities"].values())

    def test_decisions_obey_the_accumulator_rules(self, run):
        events, _ = run
        since = 0
        decided = 0
        for w in events[1:-1]:
            since += 1
            d = w.get("decision")
            if not d:
                continue
            assert d["n_windows"] == since, "the accumulator did not reset"
            if d["timed_out"]:
                assert d["label"] is None and d["n_windows"] == 40
            else:
                decided += 1
                assert d["label"] in events[0]["classes"]
                assert d["n_windows"] >= 2, "committed before min_windows"
                assert d["confidence"] >= 0.9 - 1e-4
                assert d["confidence"] == pytest.approx(max(w["posterior"].values()),
                                                        abs=1e-4)
                assert d["label"] == max(w["posterior"], key=w["posterior"].get)
            since = 0
        assert decided > 0, "a learnable recording produced no decision at all"

    def test_band_power_is_per_channel_and_bounded(self, run):
        events, predictor = run
        for w in events[1:-1]:
            assert len(w["band_power"]) == predictor.card.n_channels
            assert all(0.0 <= v <= 1.0 for v in w["band_power"])

    def test_trace_is_the_recordings_own_samples_without_overlap(self, run, edfs):
        events, predictor = run
        trace = events[0]["trace"]
        assert trace["channels"] == [c for c in TRACE_CHANNELS
                                     if c in predictor.card.ch_names]
        assert trace["sfreq"] == 160.0
        step = trace["samples_per_step"]
        assert step == 80

        source = predictor.stream_from_edf(edfs["good"], step_seconds=0.5)
        picks = [list(predictor.card.ch_names).index(c) for c in trace["channels"]]
        for w in events[1:-1]:
            block = np.asarray(w["raw"]["samples"])
            assert block.shape == (len(picks), step)
            end = round(w["onset_seconds"] * 160) + source.window_samples
            expected = source.data[np.asarray(picks), end - step:end]
            np.testing.assert_allclose(block, expected, atol=0.051)

    def test_stream_agrees_with_the_estimator(self, run, edfs):
        events, predictor = run
        source = predictor.stream_from_edf(edfs["good"], step_seconds=0.5)
        X = np.stack([w.data for w in source]).astype(np.float32)
        proba = predictor.model.predict_proba(X)
        got = np.array([[w["probabilities"][c] for c in events[0]["classes"]]
                        for w in events[1:-1]])
        np.testing.assert_allclose(got, proba, atol=6e-5)

    def test_real_truncation_is_announced(self, artifacts, edfs):
        app = fx.make_app(artifacts["lr"], max_epochs=4)
        events = frames(stream(app.test_client(), edfs["short10"]))
        assert [e.get("type") for e in events][-2:] == ["truncated", "end"]
        assert events[-1]["n_windows"] == 4

    def test_no_false_truncation_at_exactly_the_cap(self, artifacts, edfs):
        from bwt.serving.predictor import Predictor

        n = len(Predictor.load(artifacts["lr"]).stream_from_edf(
            edfs["short10"], step_seconds=0.5))
        app = fx.make_app(artifacts["lr"], max_epochs=n)
        events = frames(stream(app.test_client(), edfs["short10"]))
        assert events[-1]["n_windows"] == n
        assert "truncated" not in [e.get("type") for e in events]

    def test_smallest_accepted_window_budget_can_still_decide(self, client, edfs):
        events = frames(stream(client, edfs["good"], max_windows=1))
        decisions = [w["decision"] for w in events if w.get("decision")]
        assert any(not d["timed_out"] for d in decisions)

    def test_garbage_content_yields_one_generic_error_frame(self, client):
        events = frames(upload(client, b"\x00" * 6000, url="/api/v1/stream"))
        assert events == [{"type": "error",
                           "message": "the recording could not be decoded"}]

    def test_missing_channel_error_names_it(self, client, edfs):
        events = frames(stream(client, edfs["missing_c3"]))
        assert events[-1]["type"] == "error"
        assert "C3" in events[-1]["message"]

    def test_wrong_rate_error_is_actionable_on_the_stream_too(self, client, edfs):
        events = frames(stream(client, edfs["rate128"]))
        assert events[-1]["type"] == "error"
        assert "Hz" in events[-1]["message"]

    def test_internal_failures_do_not_leak_into_frames(self, artifacts, edfs,
                                                       monkeypatch):
        from bwt.serving.predictor import Predictor

        def explode(*_a, **_k):
            raise RuntimeError(r"secret C:\internal\path token=hunter2")

        app = fx.make_app(artifacts["lr"])
        monkeypatch.setattr(Predictor, "stream_from_edf", explode)
        text = stream(app.test_client(), edfs["five"]).get_data(as_text=True)
        assert "hunter2" not in text and "internal" not in text
        assert '"type": "error"' in text

    def test_replay_speed_is_honoured(self, client, edfs):
        started = time.perf_counter()
        events = frames(stream(client, edfs["short10"], speed=10, step=1.0))
        elapsed = time.perf_counter() - started
        last_onset = events[-2]["onset_seconds"]
        assert last_onset >= 5.0
        assert elapsed >= last_onset / 10 * 0.9, elapsed


# --------------------------------------------------------------------------- #
# Real server: concurrency, disconnects, starvation
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def live(artifacts):
    with fx.LiveServer(fx.make_app(artifacts["four"]), threads=8) as server:
        yield server


class TestUnderRealConcurrency:
    def _predict(self, url, path):
        import requests

        with open(path, "rb") as fh:
            return requests.post(url + "/api/v1/predict", files={"file": fh},
                                 timeout=300)

    def test_simultaneous_decodes_on_the_transductive_model(self, live, edfs):
        """The regression that took the whole process down (commit b685915)."""
        results: list = [None] * 8
        barrier = threading.Barrier(8)

        def one(i):
            barrier.wait()
            response = self._predict(live.url, edfs["no_cues"])
            results[i] = (response.status_code, response.json())

        threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=300)

        assert all(r is not None and r[0] == 200 for r in results), results
        for _, body in results:
            body.pop("request_id")
        assert all(body == results[0][1] for _, body in results)
        import requests

        assert requests.get(live.url + "/healthz", timeout=10).ok

    def test_mixed_streams_and_decodes_all_complete(self, live, edfs):
        import requests

        outcomes: list = []
        lock = threading.Lock()

        def decode():
            response = self._predict(live.url, edfs["short10"])
            with lock:
                outcomes.append(("decode", response.status_code))

        def streamer():
            with open(edfs["short10"], "rb") as fh:
                response = requests.post(live.url + "/api/v1/stream?band_power=1",
                                         files={"file": fh}, timeout=300)
            lines = [strict_json(x) for x in response.text.splitlines() if x]
            with lock:
                outcomes.append(("stream", lines[-1]["type"]))

        threads = ([threading.Thread(target=decode) for _ in range(4)]
                   + [threading.Thread(target=streamer) for _ in range(3)])
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=300)
        assert sorted(outcomes) == [("decode", 200)] * 4 + [("stream", "end")] * 3
        assert requests.get(live.url + "/healthz", timeout=10).ok

    def test_client_hanging_up_mid_stream(self, live, edfs):
        import requests

        with open(edfs["good"], "rb") as fh:
            response = requests.post(live.url + "/api/v1/stream?speed=2",
                                     files={"file": fh}, timeout=60, stream=True)
            for i, _ in enumerate(response.iter_lines()):
                if i >= 2:
                    break
            response.close()
        time.sleep(1.0)
        assert requests.get(live.url + "/healthz", timeout=10).ok
        assert self._predict(live.url, edfs["short10"]).status_code == 200

    def test_paced_streams_cannot_starve_other_users(self, artifacts, edfs):
        import requests

        with fx.LiveServer(fx.make_app(artifacts["lr"]), threads=4) as server:
            held = []
            for _ in range(4):
                fh = open(edfs["five"], "rb")  # noqa: SIM115 - held open on purpose
                response = requests.post(server.url + "/api/v1/stream?speed=0.25",
                                         files={"file": fh}, timeout=60,
                                         stream=True)
                next(response.iter_lines())  # the start frame: worker is busy
                held.append((fh, response))
            try:
                assert requests.get(server.url + "/healthz", timeout=3).ok
            finally:
                for fh, response in held:
                    response.close()
                    fh.close()
