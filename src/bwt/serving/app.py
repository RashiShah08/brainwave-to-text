"""Flask application.

Differences from the version this replaces, all of them deliberate:

* ``debug`` is never enabled. The Werkzeug debugger is a remote code execution
  primitive and the previous ``app.run(debug=True)`` shipped it.
* Uploads are size-capped and written to a private temporary file that is
  removed in a ``finally`` block, so a failed prediction cannot leak files.
* The uploaded filename never influences the path written to.
* The model is loaded once at application start and its identity, classes, and
  measured accuracy are exposed at ``/api/v1/model`` -- a prediction is not
  interpretable without them.
* Errors return structured JSON with a real status code instead of a bare
  string.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

from flask import Flask, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from bwt import __version__
from bwt.config import Config
from bwt.logging_utils import get_logger
from bwt.paths import repo_root
from bwt.serving.predictor import InputContractError, Predictor

log = get_logger(__name__)

ALLOWED_SUFFIXES = {".edf"}


def create_app(config: Config | None = None, predictor: Predictor | None = None) -> Flask:
    """Application factory.

    Passing ``predictor`` explicitly is what lets the test suite exercise the
    HTTP layer without a trained model on disk.
    """
    config = config or Config.load()

    app = Flask(
        __name__,
        template_folder=str(repo_root() / "templates"),
        static_folder=str(repo_root() / "static"),
    )
    app.config.update(
        MAX_CONTENT_LENGTH=config.serve.max_upload_mb * 1024 * 1024,
        JSON_SORT_KEYS=False,
        PROPAGATE_EXCEPTIONS=False,
    )

    if predictor is None:
        predictor = Predictor.load(
            config.serve.model,
            strict_versions=config.serve.strict_versions,
            max_epochs=config.serve.max_epochs_per_request,
        )
    app.extensions["bwt_predictor"] = predictor
    app.extensions["bwt_config"] = config

    _register_routes(app)
    _register_error_handlers(app)

    log.info(
        "service ready: model=%s task=%s classes=%s",
        predictor.name, predictor.card.task, predictor.card.classes,
    )
    return app


def _predictor() -> Predictor:
    """The predictor bound to the active application."""
    from flask import current_app

    return current_app.extensions["bwt_predictor"]


def _register_routes(app: Flask) -> None:

    @app.before_request
    def _start_timer() -> None:
        g.started = time.time()
        g.request_id = uuid.uuid4().hex[:12]

    @app.after_request
    def _log_request(response):
        if request.path != "/healthz":
            log.info(
                "%s %s -> %s (%.0f ms) [%s]",
                request.method, request.path, response.status_code,
                (time.time() - getattr(g, "started", time.time())) * 1000,
                getattr(g, "request_id", "-"),
            )
        return response

    # -- health and metadata ---------------------------------------------- #

    @app.get("/healthz")
    def healthz():
        """Liveness and readiness. Reports the loaded model, not just 'ok'."""
        predictor = _predictor()
        return jsonify(
            status="ok",
            version=__version__,
            model=predictor.name,
            task=predictor.card.task,
            classes=predictor.card.classes,
        )

    @app.get("/api/v1/model")
    def model_info():
        predictor = _predictor()
        return jsonify(
            model=predictor.name,
            card=predictor.card.to_dict(),
            performance=predictor.performance_note(),
            input_contract={
                "format": "EDF",
                "sfreq_hz": predictor.card.sfreq,
                "n_channels": predictor.card.n_channels,
                "channels": predictor.card.ch_names,
                "epoch_samples": predictor.card.n_times,
                "window_seconds": [predictor.card.tmin, predictor.card.tmax],
                "units": predictor.card.units,
            },
        )

    # -- UI ---------------------------------------------------------------- #

    @app.get("/")
    def index():
        predictor = _predictor()
        return render_template(
            "upload.html",
            model=predictor.card,
            performance=predictor.performance_note(),
        )

    @app.post("/predict")
    def predict_form():
        predictor = _predictor()
        try:
            batch = _handle_upload(predictor)
        except InputContractError as exc:
            return render_template("result.html", error=str(exc),
                                   model=predictor.card), 400
        except ValueError as exc:
            return render_template("result.html", error=str(exc),
                                   model=predictor.card), 400
        return render_template(
            "result.html",
            batch=batch,
            result=batch.to_dict(),
            model=predictor.card,
            performance=predictor.performance_note(),
        )

    # -- JSON API ----------------------------------------------------------- #

    @app.post("/api/v1/predict")
    def predict_api():
        predictor = _predictor()
        batch = _handle_upload(predictor)
        return jsonify(
            request_id=getattr(g, "request_id", None),
            **batch.to_dict(),
            performance=predictor.performance_note(),
        )


def _handle_upload(predictor: Predictor):
    """Validate and consume the uploaded file, then predict.

    The file is written to a private temp path -- never a path derived from the
    client-supplied filename -- and always removed.
    """
    if "file" not in request.files:
        raise ValueError("no file part in the request; send multipart field 'file'")

    upload = request.files["file"]
    if not upload.filename:
        raise ValueError("no file selected")

    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(
            f"unsupported file type {suffix or '(none)'}; "
            f"expected one of {sorted(ALLOWED_SUFFIXES)}"
        )

    handle, temp_path = tempfile.mkstemp(suffix=".edf", prefix="bwt_upload_")
    os.close(handle)
    temp_path = Path(temp_path)
    try:
        upload.save(temp_path)
        if temp_path.stat().st_size == 0:
            raise ValueError("uploaded file is empty")
        return predictor.predict_edf(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _register_error_handlers(app: Flask) -> None:

    def _wants_json() -> bool:
        return request.path.startswith("/api/") or request.is_json

    @app.errorhandler(InputContractError)
    def _contract(exc: InputContractError):
        log.warning("input contract violation: %s", exc)
        return jsonify(error="input_contract", message=str(exc)), 400

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(exc: RequestEntityTooLarge):
        limit = app.config["MAX_CONTENT_LENGTH"] / 1024 / 1024
        return jsonify(
            error="payload_too_large",
            message=f"upload exceeds the {limit:.0f} MB limit",
        ), 413

    @app.errorhandler(ValueError)
    def _bad_request(exc: ValueError):
        return jsonify(error="bad_request", message=str(exc)), 400

    @app.errorhandler(HTTPException)
    def _http(exc: HTTPException):
        if _wants_json():
            return jsonify(error=exc.name.lower().replace(" ", "_"),
                           message=exc.description), exc.code
        return exc

    @app.errorhandler(Exception)
    def _unexpected(exc: Exception):
        # Log the detail, return a generic message: internal exception text can
        # disclose filesystem layout and library internals.
        log.exception("unhandled error [%s]", getattr(g, "request_id", "-"))
        return jsonify(
            error="internal_error",
            message="the request could not be processed",
            request_id=getattr(g, "request_id", None),
        ), 500


def main() -> None:  # pragma: no cover - process entry point
    """Run the app under waitress, a real WSGI server."""
    from waitress import serve

    config = Config.load()
    app = create_app(config)
    log.info("listening on http://%s:%d", config.serve.host, config.serve.port)
    serve(app, host=config.serve.host, port=config.serve.port, threads=4)


__all__ = ["create_app", "main"]
