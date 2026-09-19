"""Security regressions for the outward-facing surface.

The v1 service shipped `app.run(debug=True)` — the Werkzeug debugger is a
remote code execution primitive — took uploads with no size cap, and wrote them
to a path derived from the client-supplied filename. Each of those is pinned
here so it cannot come back quietly.

Some checks read the source rather than exercising behaviour. That is
deliberate: "there is no `debug=True` anywhere" is a property of the code, and
asserting it at runtime would only prove the one path the test happened to take.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

from bwt.config import Config
from bwt.paths import repo_root
from bwt.serving.app import create_app
from bwt.serving.predictor import Predictor

SRC = repo_root() / "src" / "bwt"


def _python_sources() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]


def _source_text() -> dict[Path, str]:
    return {p: p.read_text(encoding="utf-8") for p in _python_sources()}


@pytest.fixture
def predictor(trained_artifact):
    return Predictor.load(trained_artifact)


@pytest.fixture
def client(predictor):
    config = Config()
    config.serve.max_upload_mb = 2
    app = create_app(config, predictor=predictor)
    app.config.update(TESTING=True)
    return app.test_client()


class TestNoDangerousConstructs:
    def test_debug_is_never_enabled_in_source(self):
        """Only a docstring may mention debug=True, describing the old bug."""
        offenders = []
        for path, text in _source_text().items():
            for number, line in enumerate(text.splitlines(), start=1):
                if re.search(r"debug\s*=\s*True", line):
                    # Allow the historical reference in prose.
                    if "``app.run(debug=True)``" in line:
                        continue
                    offenders.append(f"{path.name}:{number}")
        assert not offenders, f"debug=True found at {offenders}"

    def test_no_shell_true(self):
        offenders = [p.name for p, t in _source_text().items() if "shell=True" in t]
        assert not offenders

    def test_no_eval_or_exec(self):
        pattern = re.compile(r"(?<![\w.])(eval|exec)\s*\(")
        offenders = [p.name for p, t in _source_text().items() if pattern.search(t)]
        assert not offenders

    def test_serving_never_unpickles_request_data(self):
        """Model weights are unpickled; request bodies must never be."""
        serving = (SRC / "serving").rglob("*.py")
        for path in serving:
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            assert "pickle.load" not in text
            assert "yaml.load(" not in text  # unsafe loader


class TestAppConfiguration:
    def test_app_is_not_in_debug_mode(self, predictor):
        assert create_app(Config(), predictor=predictor).debug is False

    def test_upload_limit_is_applied(self, predictor):
        config = Config()
        config.serve.max_upload_mb = 7
        app = create_app(config, predictor=predictor)
        assert app.config["MAX_CONTENT_LENGTH"] == 7 * 1024 * 1024

    def test_exceptions_are_not_propagated_to_the_client(self, predictor):
        app = create_app(Config(), predictor=predictor)
        assert app.config["PROPAGATE_EXCEPTIONS"] is False


class TestUploadHandling:
    def test_oversized_upload_rejected_with_413(self, client):
        payload = b"0" * (3 * 1024 * 1024)
        response = client.post(
            "/api/v1/predict",
            data={"file": (io.BytesIO(payload), "big.edf")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 413

    @pytest.mark.parametrize("filename", [
        "../../escape.edf",
        "..\\..\\escape.edf",
        "/etc/passwd.edf",
        "C:\\Windows\\System32\\evil.edf",
    ])
    def test_traversal_filenames_write_nothing_outside_temp(self, client, filename):
        marker = repo_root() / "escape.edf"
        assert not marker.exists()
        client.post(
            "/api/v1/predict",
            data={"file": (io.BytesIO(b"junk"), filename)},
            content_type="multipart/form-data",
        )
        assert not marker.exists()

    def test_non_edf_extension_rejected(self, client):
        response = client.post(
            "/api/v1/predict",
            data={"file": (io.BytesIO(b"whatever"), "payload.py")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_no_temp_files_are_left_behind(self, client, tmp_path, monkeypatch):
        import tempfile

        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        client.post(
            "/api/v1/predict",
            data={"file": (io.BytesIO(b"not an edf"), "x.edf")},
            content_type="multipart/form-data",
        )
        assert list(tmp_path.glob("bwt_upload_*")) == []


class TestErrorDisclosure:
    def test_internal_errors_do_not_leak_details(self, predictor, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("secret: /srv/keys/private.pem")

        monkeypatch.setattr(Predictor, "predict_edf", explode)
        app = create_app(Config(), predictor=predictor)
        app.config.update(TESTING=False)
        response = app.test_client().post(
            "/api/v1/predict",
            data={"file": (io.BytesIO(b"x" * 64), "a.edf")},
            content_type="multipart/form-data",
        )
        body = response.get_data(as_text=True)
        assert response.status_code == 500
        assert "secret" not in body
        assert "private.pem" not in body
        assert "Traceback" not in body

    def test_malformed_edf_does_not_leak_filesystem_paths(self, client):
        response = client.post(
            "/api/v1/predict",
            data={"file": (io.BytesIO(b"definitely not an EDF"), "bad.edf")},
            content_type="multipart/form-data",
        )
        body = response.get_data(as_text=True)
        assert "Traceback" not in body
        assert "/home/" not in body
        assert not re.search(r"[A-Z]:\\\\Users", body)


class TestStreamParameterValidation:
    @pytest.mark.parametrize("query,expected", [
        ("speed=-1", 400),
        ("speed=1000", 400),
        ("step=0", 400),
        ("step=99", 400),
        ("threshold=0.1", 400),
        ("threshold=1.5", 400),
    ])
    def test_out_of_range_parameters_are_rejected(self, client, query, expected):
        response = client.post(
            f"/api/v1/stream?{query}",
            data={"file": (io.BytesIO(b"x" * 64), "a.edf")},
            content_type="multipart/form-data",
        )
        assert response.status_code == expected
