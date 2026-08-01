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
from collections.abc import Sequence
from pathlib import Path

from flask import Flask, current_app, g, jsonify, render_template, request
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

    @app.get("/api/v1/geometry")
    def geometry():
        """3D electrode positions for the loaded model's channel set.

        Derived from the montage at request time rather than a static file, so
        the visualiser can never draw a head that disagrees with the model.
        """
        from bwt.serving.geometry import electrode_geometry

        predictor = _predictor()
        return jsonify(electrode_geometry(predictor.card.ch_names))

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

    # -- live streaming ------------------------------------------------------ #

    @app.get("/live")
    def live():
        predictor = _predictor()
        return render_template(
            "live.html",
            model=predictor.card,
            performance=predictor.performance_note(),
        )

    @app.post("/api/v1/stream")
    def stream_api():
        """Decode a recording window by window, streaming NDJSON as it goes.

        One JSON object per line rather than a single response body, so the
        browser can render each decoded window as it arrives. Newline-delimited
        JSON is used in preference to server-sent events because the recording
        arrives by POST and ``EventSource`` only issues GETs.
        """
        import json as _json

        predictor = _predictor()
        config = current_app.extensions["bwt_config"]

        speed = request.args.get("speed", default=0.0, type=float)
        step = request.args.get("step", default=0.5, type=float)
        threshold = request.args.get("threshold", default=0.9, type=float)
        max_windows = request.args.get("max_windows", default=40, type=int)
        want_power = request.args.get("band_power", default="0") in {"1", "true"}
        want_raw = request.args.get("raw", default="0") in {"1", "true"}

        if not 0.0 <= speed <= 20.0:
            raise ValueError("speed must be between 0 and 20")
        if not 0.05 <= step <= 5.0:
            raise ValueError("step must be between 0.05 and 5 seconds")
        if not 0.5 < threshold < 1.0:
            raise ValueError("threshold must be between 0.5 and 1.0")

        path = _save_upload()

        def generate():
            try:
                stream = predictor.stream_from_edf(
                    path, speed=speed, step_seconds=step
                )
                decoder = predictor.streaming_decoder(
                    threshold=threshold, max_windows=max_windows
                )
                total = len(stream)
                picks = _trace_picks(predictor.card.ch_names) if want_raw else []
                start_payload = {
                    "type": "start",
                    "classes": list(predictor.card.classes),
                    "n_windows": min(total, config.serve.max_epochs_per_request),
                    "window_seconds": predictor.card.tmax - predictor.card.tmin,
                    "step_seconds": step,
                }
                if picks:
                    start_payload["trace"] = {
                        "channels": [predictor.card.ch_names[i] for i in picks],
                        "sfreq": float(predictor.card.sfreq),
                        "samples_per_step": int(stream.step_samples),
                    }
                yield _json.dumps(start_payload) + "\n"

                emitted = 0
                for event in _iter_events(decoder, stream,
                                          with_band_power=want_power,
                                          trace_picks=picks,
                                          trace_samples=stream.step_samples if picks else 0):
                    yield _json.dumps({"type": "window", **event.to_dict()}) + "\n"
                    emitted += 1
                    if emitted >= config.serve.max_epochs_per_request:
                        yield _json.dumps({
                            "type": "truncated",
                            "message": f"stopped after {emitted} windows",
                        }) + "\n"
                        break
                yield _json.dumps({"type": "end", "n_windows": emitted}) + "\n"
            except InputContractError as exc:
                yield _json.dumps({"type": "error", "message": str(exc)}) + "\n"
            except Exception:
                log.exception("streaming failed")
                yield _json.dumps({
                    "type": "error",
                    "message": "the recording could not be decoded",
                }) + "\n"
            finally:
                path.unlink(missing_ok=True)

        return current_app.response_class(
            generate(), mimetype="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )


#: Channels sent as waveforms when a caller asks for them: a left / midline /
#: right sweep across the motor strip, so the trace block reads as a montage
#: rather than an arbitrary subset. Missing names are skipped.
TRACE_CHANNELS = ("FC5", "C5", "CP5", "FCz", "Cz", "CPz", "FC6", "C6", "CP6")


def _trace_picks(channels: Sequence[str]) -> list[int]:
    """Indices of TRACE_CHANNELS in a model's channel list, else a spread."""
    lookup = {name.upper(): i for i, name in enumerate(channels)}
    picks = [lookup[n.upper()] for n in TRACE_CHANNELS if n.upper() in lookup]
    if picks:
        return picks
    step = max(1, len(channels) // 9)
    return list(range(0, len(channels), step))[:9]


def _iter_events(decoder, stream, *, with_band_power: bool = False,
                 trace_picks: Sequence[int] | None = None,
                 trace_samples: int = 0):
    """Yield streaming events one at a time.

    ``StreamingDecoder.run`` collects everything before returning, which defeats
    the point of a stream, so the generator form is built here from the same
    components rather than duplicating the decode logic.

    ``with_band_power`` adds per-electrode mu/beta power for the 3D visualiser.
    It is computed alongside the decoder purely for display and never feeds back
    into a prediction.
    """
    import numpy as np

    from bwt.streaming import BandPowerNormaliser, StreamEvent, channel_band_power

    classes = decoder.classes
    decoder.accumulator.reset()

    normalise = (
        BandPowerNormaliser(decoder.predictor.card.n_channels)
        if with_band_power else None
    )

    for window in stream:
        probabilities = decoder.predictor.model.predict_proba(
            window.data[None, ...].astype(np.float32)
        )[0]
        decision = decoder.accumulator.update(probabilities)
        event = StreamEvent(
            window=window.index,
            onset_seconds=window.onset_seconds,
            probabilities={c: float(p) for c, p in zip(classes, probabilities, strict=True)},
            top_label=classes[int(np.argmax(probabilities))],
            posterior=decoder.accumulator.posterior_dict(),
            decision=decision,
        )

        if normalise is not None:
            power = channel_band_power(window.data, decoder.predictor.card.sfreq)
            event.band_power = [float(v) for v in normalise(power)]

        if trace_picks and trace_samples > 0:
            # Only the samples this step advanced by. Windows overlap, so
            # sending the whole window would draw the same signal repeatedly.
            block = window.data[np.asarray(trace_picks), -trace_samples:]
            event.raw = {
                "samples": [[round(float(v), 1) for v in row] for row in block],
            }

        if decision is not None:
            decoder.accumulator.reset()
        yield event


def _save_upload() -> Path:
    """Validate the uploaded file and write it to a private temp path.

    The destination never derives from the client-supplied filename. Callers own
    the returned path and must delete it.
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
    upload.save(temp_path)
    if temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        raise ValueError("uploaded file is empty")
    return temp_path


def _handle_upload(predictor: Predictor):
    """Validate and consume the uploaded file, then predict."""
    temp_path = _save_upload()
    try:
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
