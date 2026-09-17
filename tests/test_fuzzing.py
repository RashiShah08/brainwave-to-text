"""Fuzzing the upload and query surface with Hypothesis.

A valid synthetic EDF is corrupted field by field -- every fixed-width header
field of the EDF specification, for the whole file and for each signal -- and
in its data records (bit flips, truncation, trailing garbage). Query strings,
URL paths and raw request bodies are generated freely. For every input the
service must give a clean answer: a 2xx or 4xx, strict JSON, a stream that ends
in an end or error frame, no internals in the text, and no temporary file left
behind.

``BWT_FUZZ_EXAMPLES`` sets the examples per test (default 120).
"""

from __future__ import annotations

import io
import json
import os
import string
import tempfile
from urllib.parse import quote

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import _factories as fx

FUZZ = settings(
    max_examples=int(os.environ.get("BWT_FUZZ_EXAMPLES", "120")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large,
                           HealthCheck.function_scoped_fixture],
)

#: Fixed-width fields of the EDF header, (offset, width). Per-signal fields
#: repeat once per signal starting at byte 256, in this order.
GLOBAL_FIELDS = {
    "version": (0, 8), "patient": (8, 80), "recording": (88, 80),
    "startdate": (168, 8), "starttime": (176, 8), "header_bytes": (184, 8),
    "reserved": (192, 44), "n_records": (236, 8), "record_duration": (244, 8),
    "n_signals": (252, 4),
}
SIGNAL_FIELDS = [("label", 16), ("transducer", 80), ("physical_dimension", 8),
                 ("physical_min", 8), ("physical_max", 8), ("digital_min", 8),
                 ("digital_max", 8), ("prefilter", 80), ("samples_per_record", 8),
                 ("signal_reserved", 32)]
INTERESTING = ["0", "-1", "1", "2", "64", "65", "99999999", "-9999999", "", " ",
               "nan", "inf", "-inf", "1e308", "-1e308", "1e-308", "abc", "0.000001",
               "+5", "0x10", "1.5", "-0", "32767", "-32768", "160", "1/160",
               "\x00\x00\x00\x00", "EDF+D", "EDF+C", "Fp1.", "uV", "mV", "V"]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    return fx.build_artifact(tmp_path_factory.mktemp("fuzz") / "lr")


@pytest.fixture(scope="module")
def seed(tmp_path_factory):
    return fx.write_edf(tmp_path_factory.mktemp("fuzz_seed") / "seed.edf",
                        duration=10.0).read_bytes()


@pytest.fixture(scope="module")
def app(artifact):
    return fx.make_app(artifact, max_upload_mb=8)


@pytest.fixture(autouse=True)
def private_spool(tmp_path, monkeypatch):
    """Uploads spool here, so a leaked temporary file is visible."""
    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spool))
    yield
    assert list(spool.iterdir()) == [], "an upload left a temporary file behind"


def strict_json(text: str):
    return json.loads(text, parse_constant=lambda c: pytest.fail(f"invalid JSON {c}"))


def assert_clean(app, payload: bytes, *, filename="rec.edf"):
    client = app.test_client()
    for url in ("/api/v1/predict", "/predict", "/api/v1/stream?band_power=1&raw=1"):
        response = client.post(url, data={"file": (io.BytesIO(payload), filename)},
                               content_type="multipart/form-data")
        body = response.get_data(as_text=True)
        assert response.status_code < 500, (url, response.status_code, body[:300])
        assert "Traceback" not in body and "site-packages" not in body, url
        if url == "/predict":
            continue
        if "stream" in url and response.status_code == 200:
            lines = [strict_json(x) for x in body.splitlines() if x]
            assert lines, "empty stream"
            assert lines[-1]["type"] in {"end", "error"}, lines[-1]
            for frame in lines:
                assert frame.get("type") in {"start", "window", "truncated", "end", "error"}
        else:
            payload_json = strict_json(body)
            assert payload_json.get("error") != "internal_error", payload_json


def set_field(data: bytes, offset: int, width: int, value: str) -> bytes:
    encoded = value.encode("latin-1", "replace")[:width].ljust(width, b" ")
    return data[:offset] + encoded + data[offset + width:]


field_value = st.one_of(
    st.sampled_from(INTERESTING),
    st.text(alphabet=string.printable, max_size=16),
    st.integers(-10**9, 10**9).map(str),
    st.floats(allow_nan=True, allow_infinity=True).map(repr),
)


class TestHeaderFuzz:
    @FUZZ
    @given(name=st.sampled_from(sorted(GLOBAL_FIELDS)), value=field_value)
    def test_any_global_header_field(self, app, seed, name, value):
        offset, width = GLOBAL_FIELDS[name]
        assert_clean(app, set_field(seed, offset, width, value))

    @FUZZ
    @given(field=st.integers(0, len(SIGNAL_FIELDS) - 1), signal=st.integers(0, 64),
           value=field_value)
    def test_any_per_signal_header_field(self, app, seed, field, signal, value):
        n_signals = int(seed[252:256])
        signal = signal % n_signals
        offset = 256
        for _, width in SIGNAL_FIELDS[:field]:
            offset += width * n_signals
        width = SIGNAL_FIELDS[field][1]
        assert_clean(app, set_field(seed, offset + signal * width, width, value))

    @FUZZ
    @given(edits=st.lists(st.tuples(st.sampled_from(sorted(GLOBAL_FIELDS)), field_value),
                          min_size=2, max_size=6))
    def test_several_fields_at_once(self, app, seed, edits):
        data = seed
        for name, value in edits:
            offset, width = GLOBAL_FIELDS[name]
            data = set_field(data, offset, width, value)
        assert_clean(app, data)


class TestDataFuzz:
    @FUZZ
    @given(flips=st.lists(st.tuples(st.integers(0, 10**9), st.integers(1, 255)),
                          min_size=1, max_size=300))
    def test_bit_flips_anywhere_after_the_header(self, app, seed, flips):
        header = int(seed[184:192])
        data = bytearray(seed)
        for position, mask in flips:
            index = header + position % (len(data) - header)
            data[index] ^= mask
        assert_clean(app, bytes(data))

    @FUZZ
    @given(cut=st.integers(0, 10**9), tail=st.binary(max_size=2048))
    def test_truncated_or_padded_files(self, app, seed, cut, tail):
        assert_clean(app, seed[: cut % (len(seed) + 1)] + tail)

    @FUZZ
    @given(blob=st.binary(max_size=8192))
    def test_arbitrary_bytes_named_edf(self, app, blob):
        assert_clean(app, blob)


QUERY_KEYS = ["speed", "step", "threshold", "max_windows", "band_power", "raw", "x",
              "speed[]", ""]


class TestRequestFuzz:
    @FUZZ
    @given(params=st.lists(st.tuples(st.sampled_from(QUERY_KEYS),
                                     st.text(max_size=24)), max_size=6))
    def test_any_query_string_on_the_stream(self, app, seed, params):
        query = "&".join(f"{quote(k)}={quote(v)}" for k, v in params)
        response = app.test_client().post(
            f"/api/v1/stream?{query}",
            data={"file": (io.BytesIO(seed), "rec.edf")},
            content_type="multipart/form-data")
        assert response.status_code in {200, 400, 503}, response.status_code
        text = response.get_data(as_text=True)
        if response.status_code == 200:
            lines = [strict_json(x) for x in text.splitlines() if x]
            assert lines[-1]["type"] in {"end", "error"}
        else:
            assert strict_json(text)["error"] in {"bad_request", "service_unavailable"}

    @FUZZ
    @given(path=st.text(max_size=80))
    def test_any_url_path(self, app, path):
        for method in ("GET", "POST"):
            response = app.test_client().open("/" + quote(path, safe="/"), method=method)
            assert response.status_code < 500, (path, method, response.status_code)

    @FUZZ
    @given(body=st.binary(max_size=4096),
           content_type=st.sampled_from([
               "multipart/form-data", "multipart/form-data; boundary=x",
               "multipart/form-data; boundary=", "application/json",
               "application/octet-stream", "text/plain; charset=utf-16",
               "application/x-www-form-urlencoded", "", "multipart/mixed; boundary=x"]))
    def test_any_raw_body(self, app, body, content_type):
        for url in ("/api/v1/predict", "/api/v1/stream", "/predict"):
            response = app.test_client().post(url, data=body, content_type=content_type)
            assert response.status_code < 500, (url, content_type, response.status_code)

    @FUZZ
    @given(filename=st.text(max_size=120))
    def test_any_filename(self, app, seed, filename):
        response = app.test_client().post(
            "/api/v1/predict", data={"file": (io.BytesIO(seed), filename + ".edf")},
            content_type="multipart/form-data")
        assert response.status_code in {200, 400}
