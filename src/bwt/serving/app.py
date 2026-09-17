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
import re
import secrets
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from flask import Flask, current_app, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge, ServiceUnavailable

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
    # A paced replay sleeps inside a server worker for as long as the replay
    # lasts. Paced streams therefore get a budget one smaller than the worker
    # pool: however many people watch a slow replay, one worker is always left
    # for everyone else, /healthz included.
    app.extensions["bwt_replay_slots"] = threading.BoundedSemaphore(
        max(1, config.serve.threads - 1)
    )

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
        g.csp_nonce = secrets.token_urlsafe(18)

    @app.context_processor
    def _inject_nonce() -> dict:
        # Every inline <script> carries this, and only this response's policy
        # allows it: an injected script without the nonce does not run.
        return {"csp_nonce": getattr(g, "csp_nonce", "")}

    @app.after_request
    def _harden_and_log(response):
        _apply_security_headers(response)
        if request.path != "/healthz":
            # The path is client text: logged raw, a %0A in it forges a
            # separate, legitimate-looking log line.
            log.info(
                "%s %s -> %s (%.0f ms) [%s]",
                request.method, _printable(request.path), response.status_code,
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
            # The card is data read from disk: a corrupt or NaN figure in it
            # must reach the client as null, not as an invalid JSON constant.
            card=_json_safe(predictor.card.to_dict()),
            performance=predictor.performance_note(),
            cue_positions=predictor.cue_positions,
            cue_axes=predictor.cue_axes,
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
            cue_positions=predictor.cue_positions,
            max_upload_mb=current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
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
            cue_positions=predictor.cue_positions,
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
            cue_positions=predictor.cue_positions,
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

        speed = _number_arg("speed", 0.0, float)
        step = _number_arg("step", 0.5, float)
        threshold = _number_arg("threshold", 0.9, float)
        max_windows = _number_arg("max_windows", 40, int)
        want_power = request.args.get("band_power", default="0") in {"1", "true"}
        want_raw = request.args.get("raw", default="0") in {"1", "true"}

        if not 0.0 <= speed <= 20.0:
            raise ValueError("speed must be between 0 and 20")
        if not 0.05 <= step <= 5.0:
            raise ValueError("step must be between 0.05 and 5 seconds")
        if not 0.5 < threshold < 1.0:
            raise ValueError("threshold must be between 0.5 and 1.0")
        # Unvalidated, max_windows=0 or a negative made the accumulator give up
        # on the very first window, so every window in the recording emitted a
        # timed-out decision -- a stream of hundreds of non-answers, served with
        # a 200 as though it meant something.
        if not 1 <= max_windows <= 1000:
            raise ValueError("max_windows must be between 1 and 1000")

        cap = config.serve.max_epochs_per_request
        release = _claim_replay_slot(speed)
        try:
            path = _save_upload()
        except BaseException:
            release()
            raise

        def cleanup() -> None:
            path.unlink(missing_ok=True)
            release()

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
                    "cue_positions": predictor.cue_positions,
                    "cue_axes": predictor.cue_axes,
                    "n_windows": min(total, cap),
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
                    if emitted >= cap:
                        # Only when something was actually left undecoded: a
                        # recording of exactly `cap` windows was decoded whole.
                        if emitted < total:
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
                cleanup()

        response = current_app.response_class(
            generate(), mimetype="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )
        # A client that hangs up before the first frame means the generator
        # never starts, so its `finally` never runs. Closing the response always
        # happens, and cleanup is idempotent.
        response.call_on_close(cleanup)
        return response


_INT_ARG = re.compile(r"[+-]?[0-9]+")
_FLOAT_ARG = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _number_arg(name: str, default, kind: type):
    """A numeric query parameter, refusing anything that is not plainly a number.

    ``request.args.get(type=...)`` falls back to the default on a conversion
    error, so ``threshold=abc`` used to be served with threshold 0.9 and a 200.
    Python's own parsers are too lenient for a URL as well (``1_000``, ``nan``,
    non-ASCII digits), hence the explicit grammar.
    """
    raw = request.args.get(name)
    if raw is None:
        return default
    text = raw.strip()
    pattern = _INT_ARG if kind is int else _FLOAT_ARG
    if not pattern.fullmatch(text):
        noun = "an integer" if kind is int else "a number"
        raise ValueError(f"{name} must be {noun}, got {raw[:40]!r}")
    return kind(text)


def _claim_replay_slot(speed: float) -> Callable[[], None]:
    """Reserve a slot for a paced replay, or refuse with 503.

    Returns an idempotent release. Unthrottled streams are not budgeted: they
    hold a worker only for as long as the decode itself takes.
    """
    if speed <= 0:
        return lambda: None
    slots = current_app.extensions["bwt_replay_slots"]
    if not slots.acquire(blocking=False):
        raise ServiceUnavailable(
            "too many paced replays are running; try again shortly, or replay "
            "unthrottled (speed=0)"
        )
    lock = threading.Lock()
    held = [True]

    def release() -> None:
        with lock:
            if held[0]:
                held[0] = False
                slots.release()

    return release


#: Channels sent as waveforms when a caller asks for them: a left / midline /
#: right sweep across the motor strip, so the trace block reads as a montage
#: rather than an arbitrary subset. Missing names are skipped.
TRACE_CHANNELS = ("FC5", "C5", "CP5", "FCz", "Cz", "CPz", "FC6", "C6", "CP6")


#: Scripts only from this origin or carrying the per-response nonce; no eval,
#: no plugins, no framing, no <base> rewriting, forms post only back here.
#: Inline style attributes are allowed: they cannot execute anything.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self' 'nonce-{nonce}'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "worker-src 'self' blob:; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'; "
    "form-action 'self'"
)


def _apply_security_headers(response) -> None:
    nonce = getattr(g, "csp_nonce", None) or secrets.token_urlsafe(18)
    headers = response.headers
    headers.setdefault("Content-Security-Policy",
                       CONTENT_SECURITY_POLICY.format(nonce=nonce))
    headers.setdefault("X-Content-Type-Options", "nosniff")
    headers.setdefault("X-Frame-Options", "DENY")
    headers.setdefault("Referrer-Policy", "no-referrer")
    headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    if request.path.startswith("/api/"):
        # Decoded recordings are about a person; no intermediary keeps them.
        headers.setdefault("Cache-Control", "no-store")


def _printable(text: str, limit: int = 300) -> str:
    """``text`` made safe for one log line: control and line-break characters
    escaped, and the length bounded."""
    out = []
    for ch in text[:limit]:
        code = ord(ch)
        if code < 0x20 or 0x7F <= code < 0xA0 or code in (0x2028, 0x2029):
            out.append(f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}")
        else:
            out.append(ch)
    return "".join(out) + ("..." if len(text) > limit else "")


def _json_safe(value):
    """``value`` with every NaN or infinite float replaced by ``None``.

    Python's encoder writes those as ``NaN``/``Infinity``, which no strict JSON
    parser -- including the browser's ``JSON.parse`` -- accepts.
    """
    import math

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


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
        probabilities = decoder.predictor.predict_proba(
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
        declared = request.content_length
        if declared is not None and declared <= app.config["MAX_CONTENT_LENGTH"]:
            # Within the size limit, so Werkzeug refused it for its number of
            # multipart parts; blaming the size would send the user the wrong way.
            message = "the request has too many form fields"
        else:
            message = f"upload exceeds the {limit:.0f} MB limit"
        if not _wants_json():
            # Sent from the decode form: the browser should land on a page that
            # says what happened, not on a JSON document.
            return render_template(
                "result.html", error=message + ".",
                model=app.extensions["bwt_predictor"].card,
            ), 413
        return jsonify(error="payload_too_large", message=message), 413

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


def waitress_options(config: Config) -> dict:
    """Keyword arguments for waitress, derived from the serve configuration.

    Waitress buffers a whole request body before the application sees it, with
    a default cap of 1 GB, so the application's upload limit alone let a client
    make the server spool a gigabyte before being told no. Held to the upload
    limit plus room for multipart framing, an oversized upload is refused as
    soon as its headers declare it.
    """
    return {
        "host": config.serve.host,
        "port": config.serve.port,
        "threads": config.serve.threads,
        "max_request_body_size": config.serve.max_upload_mb * 1024 * 1024 + 64 * 1024,
        "clear_untrusted_proxy_headers": True,
        "ident": "bwt",
    }


def main() -> None:  # pragma: no cover - process entry point
    """Run the app under waitress, a real WSGI server."""
    from waitress import serve

    config = Config.load()
    app = create_app(config)
    log.info("listening on http://%s:%d", config.serve.host, config.serve.port)
    serve(app, **waitress_options(config))


__all__ = ["create_app", "main", "waitress_options"]
