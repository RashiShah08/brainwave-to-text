"""The attack surface, from the socket up.

Headers on every kind of response, the content-security policy and its nonce,
log forging, the WSGI server's own limits under raw-socket abuse (oversized
declared bodies, malformed framing, slowloris), multipart abuse, the full
method matrix, cross-origin behaviour, model-artifact integrity, and templates
that must stay safe by construction.

Synthetic model and recordings only (``tests/_factories.py``).
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import re
import shutil
import socket
import threading
import time
from pathlib import Path

import pytest

import _factories as fx

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    return fx.build_artifact(tmp_path_factory.mktemp("sec") / "lr")


@pytest.fixture(scope="module")
def edf(tmp_path_factory):
    return fx.write_edf(tmp_path_factory.mktemp("sec_edf") / "rec.edf", duration=20.0)


@pytest.fixture(scope="module")
def app(artifact):
    return fx.make_app(artifact, max_upload_mb=2)


@pytest.fixture
def client(app):
    return app.test_client()


def upload(client, payload, url="/api/v1/predict", *, filename="rec.edf"):
    data = payload.read_bytes() if isinstance(payload, Path) else payload
    return client.post(url, data={"file": (io.BytesIO(data), filename)},
                       content_type="multipart/form-data")


def raw_exchange(url: str, request: bytes, *, read_for: float = 3.0) -> bytes:
    """Send raw bytes to a live server and return whatever comes back."""
    host, port = url.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=read_for) as sock:
        sock.sendall(request)
        sock.settimeout(read_for)
        chunks, deadline = [], time.time() + read_for
        while time.time() < deadline:
            try:
                chunk = sock.recv(65536)
            except (TimeoutError, OSError):
                break
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)


def status_of(reply: bytes) -> int | None:
    match = re.match(rb"HTTP/1\.[01] (\d{3})", reply)
    return int(match.group(1)) if match else None


@pytest.fixture(scope="module")
def hardened(artifact):
    """The app under waitress with the production server options."""
    from bwt.config import Config
    from bwt.serving.app import waitress_options

    config = Config()
    config.serve.max_upload_mb = 2
    options = waitress_options(config)
    with fx.LiveServer(fx.make_app(artifact, max_upload_mb=2), threads=4,
                       options=options) as live:
        yield live


# --------------------------------------------------------------------------- #
# Security headers, on every kind of response
# --------------------------------------------------------------------------- #


def _responses(client, edf, app):
    yield "index", client.get("/")
    yield "live", client.get("/live")
    yield "healthz", client.get("/healthz")
    yield "model", client.get("/api/v1/model")
    yield "geometry", client.get("/api/v1/geometry")
    yield "static", client.get("/static/app.css")
    yield "404 page", client.get("/nowhere")
    yield "404 api", client.get("/api/v1/nowhere")
    yield "405 api", client.get("/api/v1/predict")
    yield "400 api", upload(client, b"\x00" * 64)
    yield "400 form", upload(client, b"\x00" * 64, "/predict")
    yield "413 api", upload(client, b"\x00" * (3 * 1024 * 1024))
    yield "413 form", upload(client, b"\x00" * (3 * 1024 * 1024), "/predict")
    yield "decode page", upload(client, edf, "/predict")
    yield "stream", upload(client, edf, "/api/v1/stream")


class TestSecurityHeaders:
    def test_every_response_carries_the_full_set(self, client, edf, app):
        for name, response in _responses(client, edf, app):
            headers = response.headers
            assert headers.get("X-Content-Type-Options") == "nosniff", name
            assert headers.get("Referrer-Policy") == "no-referrer", name
            policy = headers.get("Content-Security-Policy", "")
            for directive in ("default-src 'self'", "object-src 'none'",
                              "base-uri 'none'", "frame-ancestors https: http://localhost:5173",
                              "form-action 'self'"):
                assert directive in policy, (name, directive)
            script_src = re.search(r"script-src ([^;]+)", policy).group(1)
            assert "'unsafe-inline'" not in script_src, name
            assert "'unsafe-eval'" not in policy, name
            assert "*" not in policy, name
            assert "Set-Cookie" not in headers, name
            response.close()

    def test_a_crash_still_gets_the_headers(self, artifact, monkeypatch, edf):
        app = fx.make_app(artifact)
        predictor = app.extensions["bwt_predictor"]
        monkeypatch.setattr(predictor, "predict_edf",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        response = upload(app.test_client(), edf)
        assert response.status_code == 500
        assert "frame-ancestors https: http://localhost:5173" in response.headers["Content-Security-Policy"]
        assert response.headers.get("X-Content-Type-Options") == "nosniff"

    @pytest.mark.parametrize("setting", ["'none'", "", "  "])
    def test_framing_can_be_turned_off(self, artifact, setting, edf):
        app = fx.make_app(artifact, frame_ancestors=setting)
        for name, response in _responses(app.test_client(), edf, app):
            assert response.headers.get("X-Frame-Options") == "DENY", name
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"], name
            response.close()

    def test_allowed_sites_never_get_the_deny_header(self, client, edf, app):
        # X-Frame-Options: DENY would override the allow-list in older browsers.
        for name, response in _responses(client, edf, app):
            assert "X-Frame-Options" not in response.headers, name
            response.close()

    @pytest.mark.parametrize("path", ["/", "/live"])
    def test_every_inline_script_carries_this_responses_nonce(self, client, path):
        first, second = client.get(path), client.get(path)
        nonces = []
        for response in (first, second):
            policy = response.headers["Content-Security-Policy"]
            nonce = re.search(r"'nonce-([A-Za-z0-9_-]+)'", policy).group(1)
            assert len(nonce) >= 16
            html = response.get_data(as_text=True)
            tags = re.findall(r"<script\b([^>]*)>", html)
            assert tags, "the page has scripts to protect"
            for attrs in tags:
                if "src=" in attrs:
                    continue
                assert f'nonce="{nonce}"' in attrs, attrs
            nonces.append(nonce)
        assert nonces[0] != nonces[1], "a nonce must never repeat"

    def test_the_decode_result_page_scripts_carry_the_nonce(self, client, edf):
        response = upload(client, edf, "/predict")
        nonce = re.search(r"'nonce-([A-Za-z0-9_-]+)'",
                          response.headers["Content-Security-Policy"]).group(1)
        for attrs in re.findall(r"<script\b([^>]*)>", response.get_data(as_text=True)):
            assert "src=" in attrs or f'nonce="{nonce}"' in attrs

    @pytest.mark.parametrize("path", ["/", "/live"])
    def test_no_inline_handlers_or_javascript_urls(self, client, path):
        html = client.get(path).get_data(as_text=True)
        assert not re.search(r"<[^>]+\son[a-z]+\s*=", html, re.IGNORECASE)
        assert "javascript:" not in html.lower()

    def test_api_answers_are_never_cached_but_assets_may_be(self, client, edf):
        for response in (client.get("/api/v1/model"), client.get("/api/v1/geometry"),
                         upload(client, edf), client.get("/api/v1/nope")):
            assert response.headers.get("Cache-Control") == "no-store"
        assert client.get("/static/app.css").headers.get("Cache-Control") != "no-store"

    @pytest.mark.parametrize("origin", ["https://evil.example", "null",
                                        "http://127.0.0.1.evil.example"])
    def test_no_cross_origin_access_is_ever_granted(self, client, origin):
        for response in (
            client.get("/api/v1/model", headers={"Origin": origin}),
            client.open("/api/v1/predict", method="OPTIONS", headers={
                "Origin": origin, "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type"}),
        ):
            assert not any(k.lower().startswith("access-control-")
                           for k, _ in response.headers.items())

    def test_static_assets_have_honest_content_types(self, client):
        assert client.get("/static/app.css").mimetype == "text/css"
        assert client.get("/static/brain3d.js").mimetype in {
            "text/javascript", "application/javascript"}
        assert client.get("/static/cortex.bin").mimetype == "application/octet-stream"


# --------------------------------------------------------------------------- #
# Log forging
# --------------------------------------------------------------------------- #


FORGING = ["%0a", "%0d", "%0d%0a", "%1b%5b31m", "%00", "%e2%80%a8", "%e2%80%a9",
           "%c2%85", "%7f", "%0a%0a%0a"]


class TestLogInjection:
    @pytest.mark.parametrize("payload", FORGING)
    def test_a_path_cannot_forge_a_log_line(self, client, caplog, payload):
        caplog.set_level(logging.INFO, logger="bwt.serving.app")
        client.get(f"/x{payload}12:00:00 CRITICAL bwt.serving.app FORGED")
        messages = [r.getMessage() for r in caplog.records if "FORGED" in r.getMessage()]
        assert len(messages) == 1
        message = messages[0]
        for bad in ("\n", "\r", "\x1b", "\x00", "\u2028", "\u2029", "\x85", "\x7f"):
            assert bad not in message, (payload, message)

    def test_an_enormous_path_is_bounded_in_the_log(self, client, caplog):
        caplog.set_level(logging.INFO, logger="bwt.serving.app")
        client.get("/" + "A" * 20_000)
        message = next(r.getMessage() for r in caplog.records if "AAAA" in r.getMessage())
        assert len(message) < 600

    def test_ordinary_paths_are_logged_unchanged(self, client, caplog):
        caplog.set_level(logging.INFO, logger="bwt.serving.app")
        client.get("/static/app.css?v=1")
        assert any("GET /static/app.css -> 200" in r.getMessage() for r in caplog.records)

    def test_the_json_log_format_is_one_valid_object_per_line(self):
        from bwt.logging_utils import _JsonFormatter

        record = logging.LogRecord("bwt", logging.INFO, __file__, 1,
                                   "evil %s", ("line\nbreak\u2028\x00",), None)
        line = _JsonFormatter().format(record)
        assert "\n" not in line and "\u2028" not in line
        assert json.loads(line)["msg"] == "evil line\nbreak\u2028\x00"


# --------------------------------------------------------------------------- #
# The WSGI server's own limits, from raw sockets
# --------------------------------------------------------------------------- #


class TestServerLimits:
    def test_production_options_hold_the_body_to_the_upload_limit(self):
        from bwt.config import Config
        from bwt.serving.app import waitress_options

        config = Config()
        options = waitress_options(config)
        limit = config.serve.max_upload_mb * 1024 * 1024
        assert limit <= options["max_request_body_size"] <= limit + 1024 * 1024
        assert options["threads"] == config.serve.threads
        assert options["clear_untrusted_proxy_headers"] is True
        assert options["ident"] and "waitress" not in options["ident"]

    def test_both_entry_points_use_the_production_options(self, monkeypatch):
        from types import SimpleNamespace

        import waitress

        import bwt.cli as cli
        import bwt.serving.app as app_module

        seen = []
        monkeypatch.setattr(waitress, "serve", lambda app, **kw: seen.append(kw))
        monkeypatch.setattr(app_module, "create_app", lambda config=None: object())
        app_module.main()
        cli.cmd_serve(SimpleNamespace(model=None, host=None, port=None, dev=False))
        assert len(seen) == 2
        for kwargs in seen:
            assert "max_request_body_size" in kwargs
            assert kwargs["max_request_body_size"] < 1024 ** 3

    def test_an_oversized_declared_body_is_refused_before_it_is_sent(self, hardened):
        started = time.time()
        reply = raw_exchange(hardened.url, (
            b"POST /api/v1/predict HTTP/1.1\r\nHost: x\r\n"
            b"Content-Type: multipart/form-data; boundary=b\r\n"
            b"Content-Length: 209715200\r\n\r\n"), read_for=4.0)
        assert status_of(reply) == 413, reply[:200]
        assert time.time() - started < 4.0

    def test_an_oversized_chunked_body_is_cut_off(self, hardened):
        chunk = b"\x00" * 65536
        body = b"".join(b"%x\r\n" % len(chunk) + chunk + b"\r\n" for _ in range(60))
        reply = raw_exchange(hardened.url, (
            b"POST /api/v1/predict HTTP/1.1\r\nHost: x\r\n"
            b"Content-Type: multipart/form-data; boundary=b\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n") + body + b"0\r\n\r\n", read_for=6.0)
        status = status_of(reply)
        assert status is None or 400 <= status < 500, reply[:200]
        assert _healthy(hardened)

    @pytest.mark.parametrize("name,request_bytes", [
        ("duplicate content-length", b"POST /api/v1/predict HTTP/1.1\r\nHost: x\r\n"
         b"Content-Length: 5\r\nContent-Length: 50\r\n\r\nhello"),
        ("negative content-length", b"POST /api/v1/predict HTTP/1.1\r\nHost: x\r\n"
         b"Content-Length: -5\r\n\r\n"),
        ("bad chunk size", b"POST /api/v1/predict HTTP/1.1\r\nHost: x\r\n"
         b"Transfer-Encoding: chunked\r\n\r\nzz\r\nhello\r\n0\r\n\r\n"),
        ("unknown transfer coding", b"POST /api/v1/predict HTTP/1.1\r\nHost: x\r\n"
         b"Transfer-Encoding: gzip, chunked\r\n\r\n0\r\n\r\n"),
        ("bare LF lines", b"GET /healthz HTTP/1.1\nHost: x\n\n"),
        ("no version", b"GET /healthz\r\n\r\n"),
        ("binary garbage", bytes(range(256)) * 4),
        ("absolute URI to another host", b"GET http://evil.example/healthz HTTP/1.1\r\n"
         b"Host: x\r\n\r\n"),
    ])
    def test_malformed_framing_never_gets_a_server_error(self, hardened, name,
                                                         request_bytes):
        reply = raw_exchange(hardened.url, request_bytes, read_for=3.0)
        status = status_of(reply)
        # 501 is the correct answer to a transfer coding the server does not
        # implement (RFC 9112 section 6.1), and it closes the connection; it is
        # a refusal, not a fault. Every other case must stay below 500.
        allowed = {501} if name == "unknown transfer coding" else set()
        assert status is None or status < 500 or status in allowed, (name, reply[:200])
        if status == 501:
            assert b"Connection: close" in reply
        assert b"Traceback" not in reply
        assert _healthy(hardened)

    def test_slow_clients_cannot_starve_everyone_else(self, hardened):
        host, port = hardened.url.removeprefix("http://").split(":")
        stalled = []
        try:
            for _ in range(60):
                sock = socket.create_connection((host, int(port)), timeout=5)
                sock.sendall(b"GET /healthz HTTP/1.1\r\nHost: x\r\n")
                stalled.append(sock)
            for _ in range(6):
                for sock in stalled:
                    with contextlib.suppress(OSError):
                        sock.sendall(b"X-a: b\r\n")
                started = time.time()
                assert _healthy(hardened)
                assert time.time() - started < 3.0
                time.sleep(0.3)
        finally:
            for sock in stalled:
                sock.close()

    def test_a_header_flood_is_refused_cleanly(self, hardened):
        reply = raw_exchange(hardened.url, b"GET /healthz HTTP/1.1\r\nHost: x\r\n"
                             + b"".join(b"X-%d: %s\r\n" % (i, b"a" * 200)
                                        for i in range(2000)) + b"\r\n")
        assert status_of(reply) in {400, 431}
        assert _healthy(hardened)

    def test_proxy_headers_from_an_untrusted_client_are_dropped(self, hardened):
        import requests

        response = requests.get(hardened.url + "/", headers={
            "X-Forwarded-Proto": "https", "X-Forwarded-Host": "evil.example",
            "X-Forwarded-For": "1.2.3.4", "Host": "evil.example"}, timeout=10)
        assert response.ok and "evil.example" not in response.text
        assert response.headers.get("Server", "").lower() != "waitress"


def _healthy(server) -> bool:
    import requests

    try:
        return requests.get(server.url + "/healthz", timeout=5).ok
    except requests.RequestException:
        return False


# --------------------------------------------------------------------------- #
# Multipart abuse
# --------------------------------------------------------------------------- #


class TestMultipartAbuse:
    def test_too_many_parts_is_named_as_such(self, client):
        fields = {f"f{i}": "x" for i in range(5000)}
        fields["file"] = (io.BytesIO(b"\x00" * 100), "a.edf")
        response = client.post("/api/v1/predict", data=fields,
                               content_type="multipart/form-data")
        assert response.status_code == 413
        assert response.get_json()["message"] == "the request has too many form fields"

    @pytest.mark.parametrize("filename", [
        "a\r\nX-Injected: 1.edf", 'quote".edf', "nul\x00.edf", "..\\..\\evil.edf",
        "/etc/passwd.edf", "C:\\Windows\\win.edf", "\u202egpj.edf", "脑电图🧠.edf",
        "a" * 5000 + ".edf", ".edf", "rec.EDF", "rec.edf.exe", "rec.edf ",
        "rec.edf\x00.txt",
    ])
    def test_hostile_filenames_are_decoded_or_refused_never_crash(self, client, edf,
                                                                  filename):
        response = upload(client, edf, filename=filename)
        assert response.status_code in {200, 400}, response.get_data(as_text=True)[:200]
        assert "X-Injected" not in response.headers
        body = response.get_data(as_text=True)
        assert "Traceback" not in body and "X-Injected: 1" not in body

    def test_the_file_sent_as_a_text_field_is_refused(self, client):
        response = client.post("/api/v1/predict", data={"file": "not a file"},
                               content_type="multipart/form-data")
        assert response.status_code == 400

    @pytest.mark.parametrize("content_type,body", [
        ("multipart/form-data", b"--x\r\n\r\n"),
        ("multipart/form-data; boundary=" + "b" * 2000, b"--" + b"b" * 2000 + b"--"),
        ("multipart/form-data; boundary=x", b"--x\r\nContent-Disposition: form-data"),
        ("application/json", json.dumps({"file": "x"}).encode()),
        ("application/x-www-form-urlencoded", b"file=abc"),
        ("text/plain", b"\x00" * 100),
    ])
    def test_malformed_bodies_are_client_errors(self, client, content_type, body):
        response = client.post("/api/v1/predict", data=body, content_type=content_type)
        assert 400 <= response.status_code < 500
        assert response.mimetype == "application/json"

    def test_a_second_file_part_does_not_replace_the_first(self, client, edf):
        response = client.post("/api/v1/predict", data={
            "file": [(io.BytesIO(edf.read_bytes()), "rec.edf"),
                     (io.BytesIO(b"\x00" * 64), "junk.edf")]},
            content_type="multipart/form-data")
        assert response.status_code == 200


# --------------------------------------------------------------------------- #
# Every route, every method
# --------------------------------------------------------------------------- #


ROUTES = ["/", "/live", "/healthz", "/predict", "/api/v1/model", "/api/v1/geometry",
          "/api/v1/predict", "/api/v1/stream", "/static/app.css", "/nope",
          "/api/v1/nope", "/api/v2/model", "/API/V1/MODEL", "/api/v1/model/"]
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE",
           "PROPFIND", "CONNECT"]


class TestMethodMatrix:
    @pytest.mark.parametrize("method", METHODS)
    def test_no_route_and_method_is_a_server_error(self, client, method):
        for route in ROUTES:
            response = client.open(route, method=method)
            assert response.status_code < 500, (method, route, response.status_code)
            if (route.startswith("/api/") and response.status_code >= 400
                    and method != "HEAD"):
                assert response.mimetype == "application/json", (method, route)
            if method == "HEAD":
                assert response.get_data() == b""
            response.close()

    def test_trace_is_never_echoed(self, client):
        response = client.open("/", method="TRACE", headers={"X-Secret": "s3cr3t"})
        assert response.status_code == 405
        assert "s3cr3t" not in response.get_data(as_text=True)


# --------------------------------------------------------------------------- #
# Model artifact integrity
# --------------------------------------------------------------------------- #


class TestArtifactIntegrity:
    @pytest.fixture
    def copy(self, artifact, tmp_path):
        target = tmp_path / "artifact"
        shutil.copytree(artifact, target)
        return target

    def test_the_card_records_the_weights_checksum(self, copy):
        import hashlib

        from bwt.artifacts import CARD_FILE, PIPELINE_FILE

        card = json.loads((copy / CARD_FILE).read_text(encoding="utf-8"))
        expected = hashlib.sha256((copy / PIPELINE_FILE).read_bytes()).hexdigest()
        assert card["pipeline_sha256"] == expected

    @pytest.mark.parametrize("tamper", ["flip_a_byte", "truncate", "append",
                                        "swap_for_another_pickle"])
    def test_tampered_weights_are_refused_before_they_are_unpickled(
            self, copy, monkeypatch, tamper, tmp_path):
        import joblib

        import bwt.artifacts as artifacts

        weights = copy / artifacts.PIPELINE_FILE
        data = bytearray(weights.read_bytes())
        if tamper == "flip_a_byte":
            data[len(data) // 2] ^= 0xFF
        elif tamper == "truncate":
            data = data[:-10]
        elif tamper == "append":
            data += b"\x00"
        else:
            other = tmp_path / "other.joblib"
            joblib.dump({"not": "the model"}, other)
            data = bytearray(other.read_bytes())
        weights.write_bytes(bytes(data))

        def never(*_a, **_k):
            raise AssertionError("tampered weights were unpickled")

        monkeypatch.setattr(artifacts.joblib, "load", never)
        with pytest.raises(ValueError, match="does not match the checksum"):
            artifacts.load_artifact(copy)

    @pytest.mark.parametrize("recorded", ["", None, 12345, "   "])
    def test_a_card_without_a_usable_checksum_loads_with_a_warning(
            self, copy, caplog, recorded):
        from bwt.artifacts import CARD_FILE, load_artifact

        card_file = copy / CARD_FILE
        card = json.loads(card_file.read_text(encoding="utf-8"))
        card["pipeline_sha256"] = recorded
        card_file.write_text(json.dumps(card), encoding="utf-8")
        caplog.set_level(logging.WARNING, logger="bwt.artifacts")
        model, _ = load_artifact(copy)
        assert hasattr(model, "predict_proba")
        assert any("no pipeline checksum" in r.getMessage() for r in caplog.records)

    def test_an_uppercase_checksum_is_the_same_checksum(self, copy):
        from bwt.artifacts import CARD_FILE, load_artifact

        card_file = copy / CARD_FILE
        card = json.loads(card_file.read_text(encoding="utf-8"))
        card["pipeline_sha256"] = card["pipeline_sha256"].upper()
        card_file.write_text(json.dumps(card), encoding="utf-8")
        load_artifact(copy)


# --------------------------------------------------------------------------- #
# Templates safe by construction
# --------------------------------------------------------------------------- #


class TestTemplates:
    def test_safe_is_only_ever_applied_to_the_literal_arrow_table(self):
        for template in (REPO / "templates").glob("*.html"):
            text = template.read_text(encoding="utf-8")
            for expression in re.findall(r"\{\{(.*?)\}\}", text, re.DOTALL):
                if re.search(r"\|\s*safe\b", expression):
                    assert "'left': '&larr;'" in expression, (template.name, expression)

    def test_autoescaping_is_on_for_every_template(self, app):
        for template in (REPO / "templates").glob("*.html"):
            assert app.jinja_env.select_autoescape(template.name) \
                if hasattr(app.jinja_env, "select_autoescape") \
                else app.select_jinja_autoescape(template.name)

    def test_no_markup_objects_are_built_from_data(self):
        for source in (REPO / "src").rglob("*.py"):
            assert "Markup(" not in source.read_text(encoding="utf-8"), source

    def test_hostile_card_text_is_inert_on_every_page(self, artifact):
        app = fx.make_app(artifact)
        card = app.extensions["bwt_predictor"].card
        evil = '</script><script>window.__pwned=1</script><img src=x onerror=alert(1)>'
        card.task_description = evil
        card.pipeline = evil
        card.created_utc = evil
        client = app.test_client()
        for path in ("/", "/live"):
            html = client.get(path).get_data(as_text=True)
            assert "<script>window.__pwned" not in html, path
            assert "<img src=x onerror" not in html, path

    def test_hostile_class_names_cannot_close_the_script_they_are_embedded_in(
            self, artifact):
        app = fx.make_app(artifact)
        card = app.extensions["bwt_predictor"].card
        card.classes = ['</script><script>window.__pwned=1</script>', "b'\"<>&"]
        response = app.test_client().get("/live")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "</script><script>window.__pwned" not in html
        scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", html, re.DOTALL)
        assert any("MODEL_CLASSES" in body for body in scripts), \
            "the class list must stay inside its own script"


class TestConcurrentHeaders:
    def test_nonces_do_not_leak_between_simultaneous_requests(self, artifact):
        import requests

        seen, lock = [], threading.Lock()
        with fx.LiveServer(fx.make_app(artifact), threads=8) as live:
            def fetch():
                for _ in range(10):
                    response = requests.get(live.url + "/", timeout=30)
                    nonce = re.search(r"'nonce-([A-Za-z0-9_-]+)'",
                                      response.headers["Content-Security-Policy"]).group(1)
                    assert response.text.count(f'nonce="{nonce}"') >= 3
                    with lock:
                        seen.append(nonce)

            workers = [threading.Thread(target=fetch) for _ in range(8)]
            for w in workers:
                w.start()
            for w in workers:
                w.join(timeout=120)
        assert len(seen) == 80 and len(set(seen)) == 80
