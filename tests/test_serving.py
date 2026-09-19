"""Tests for the inference path and the HTTP service.

The centrepiece is :class:`TestTrainServeParity`. The previous version of this
project had a hand-written feature extractor in the web app that had drifted
away from the training one, so the served model silently computed different
features from the ones it was fitted on. The parity test makes that class of
bug impossible to reintroduce unnoticed.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from bwt.config import Config
from bwt.serving.app import create_app
from bwt.serving.predictor import InputContractError, Predictor


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


class TestTrainServeParity:
    def test_serving_reproduces_training_predictions_exactly(
        self, trained_artifact, synthetic_bundle, predictor
    ):
        from bwt.artifacts import load_artifact

        model, _ = load_artifact(trained_artifact)
        expected = model.predict(synthetic_bundle.X)

        served = predictor.predict_array(synthetic_bundle.X)
        actual = np.array(
            [synthetic_bundle.classes.index(p.label) for p in served]
        )
        np.testing.assert_array_equal(actual, expected)

    def test_probabilities_match_the_underlying_model(
        self, trained_artifact, synthetic_bundle, predictor
    ):
        from bwt.artifacts import load_artifact

        model, _ = load_artifact(trained_artifact)
        expected = model.predict_proba(synthetic_bundle.X[:10])
        served = predictor.predict_array(synthetic_bundle.X[:10])
        for row, prediction in zip(expected, served, strict=True):
            for index, class_name in enumerate(synthetic_bundle.classes):
                assert prediction.probabilities[class_name] == pytest.approx(
                    row[index], rel=1e-9
                )

    def test_no_duplicate_feature_extraction_code_path(self):
        """The predictor must not implement its own feature extraction."""
        import inspect

        from bwt.serving import predictor as module

        source = inspect.getsource(module)
        for banned in ("welch", "skew(", "kurtosis(", "np.var(", "butter("):
            assert banned not in source, (
                f"{banned!r} appears in the serving module; feature extraction "
                "belongs inside the fitted pipeline only"
            )


class TestInputContract:
    def test_rejects_wrong_channel_count(self, predictor):
        with pytest.raises(InputContractError, match="channels"):
            predictor.validate_array(np.zeros((2, 32, 481), dtype=np.float32))

    def test_rejects_wrong_epoch_length(self, predictor):
        with pytest.raises(InputContractError, match="samples per epoch"):
            predictor.validate_array(np.zeros((2, 64, 256), dtype=np.float32))

    def test_accepts_single_epoch_as_2d(self, predictor):
        out = predictor.validate_array(np.zeros((64, 481), dtype=np.float32))
        assert out.shape == (1, 64, 481)

    def test_rejects_four_dimensional_input(self, predictor):
        with pytest.raises(InputContractError, match="trials, channels, times"):
            predictor.validate_array(np.zeros((2, 2, 64, 481), dtype=np.float32))

    def test_missing_channels_reported_by_name(self, predictor):
        with pytest.raises(InputContractError, match="missing"):
            predictor._align_channels(["C3", "C4", "Cz"])

    def test_channel_alignment_reorders_rather_than_assumes(self, predictor):
        expected = list(predictor.card.ch_names)
        shuffled = list(reversed(expected))
        picks = predictor._align_channels(shuffled)
        assert [shuffled[i] for i in picks] == expected


@pytest.mark.slow
class TestEdfIngestion:
    def test_reads_a_synthetic_edf_with_cues(self, predictor, synthetic_edf):
        X, onsets, mode, _ = predictor.epochs_from_edf(synthetic_edf)
        assert mode == "cue_locked"
        assert X.shape[1:] == (predictor.card.n_channels, predictor.card.n_times)
        assert len(onsets) == len(X)

    def test_predicts_movements_and_nothing_else(self, predictor, synthetic_edf):
        batch = predictor.predict_edf(synthetic_edf)
        assert batch.n > 0
        assert batch.epoching == "cue_locked"
        assert {p.label for p in batch.predictions} <= set(batch.classes)
        # The output is which movement was imagined. There is no text, and no
        # speller, and the response must not offer either.
        payload = batch.to_dict()
        assert "text" not in payload
        assert "speller" not in payload

    def test_batch_serialises_to_json(self, predictor, synthetic_edf):
        payload = json.loads(json.dumps(predictor.predict_edf(synthetic_edf).to_dict()))
        assert payload["n_epochs"] > 0
        assert "majority_label" in payload


class TestHealthAndMetadata:
    def test_healthz_reports_the_loaded_model(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["status"] == "ok"
        assert payload["classes"] == ["left_fist", "right_fist"]

    def test_model_endpoint_publishes_the_input_contract(self, client):
        payload = client.get("/api/v1/model").get_json()
        contract = payload["input_contract"]
        assert contract["sfreq_hz"] == 160.0
        assert contract["n_channels"] == 64
        assert contract["epoch_samples"] == 481
        assert len(contract["channels"]) == 64

    def test_model_endpoint_reports_measured_accuracy(self, client):
        payload = client.get("/api/v1/model").get_json()
        assert payload["performance"]["chance_level"] == 0.5
        assert payload["card"]["classes"] == ["left_fist", "right_fist"]

    def test_index_page_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert "left_fist" in body

    def test_index_page_carries_the_scope_disclaimer(self, client):
        # Collapse whitespace: the sentence is line-wrapped in the template and
        # the claim matters, not its formatting.
        body = " ".join(client.get("/").get_data(as_text=True).split())
        assert "does not read words, inner speech, or intent" in body

    def test_index_page_shows_both_accuracy_figures(self, client):
        body = " ".join(client.get("/").get_data(as_text=True).split())
        assert "Within-subject" in body
        assert "Cross-subject" in body
        assert "Chance" in body


class TestUploadHandling:
    def test_missing_file_is_a_400(self, client):
        response = client.post("/api/v1/predict", data={})
        assert response.status_code == 400
        assert response.get_json()["error"] == "bad_request"

    def test_empty_filename_is_a_400(self, client):
        response = client.post(
            "/api/v1/predict",
            data={"file": (io.BytesIO(b"x"), "")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_wrong_extension_is_rejected(self, client):
        response = client.post(
            "/api/v1/predict",
            data={"file": (io.BytesIO(b"whatever"), "notes.txt")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert "unsupported file type" in response.get_json()["message"]

    def test_oversized_upload_is_rejected(self, client):
        payload = b"0" * (3 * 1024 * 1024)  # limit is 2 MB in the fixture
        response = client.post(
            "/api/v1/predict",
            data={"file": (io.BytesIO(payload), "big.edf")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 413

    def test_corrupt_edf_does_not_leak_internals(self, client):
        response = client.post(
            "/api/v1/predict",
            data={"file": (io.BytesIO(b"not an edf at all"), "bad.edf")},
            content_type="multipart/form-data",
        )
        assert response.status_code in (400, 500)
        body = response.get_data(as_text=True)
        assert "Traceback" not in body
        assert "C:\\" not in body and "/home/" not in body

    def test_upload_is_not_written_under_a_client_controlled_path(self, client):
        """A traversal filename must not create anything outside the temp dir."""
        from bwt.paths import repo_root

        marker = repo_root() / "pwned.edf"
        assert not marker.exists()
        client.post(
            "/api/v1/predict",
            data={"file": (io.BytesIO(b"junk"), "../../pwned.edf")},
            content_type="multipart/form-data",
        )
        assert not marker.exists()


class TestSecurity:
    def test_debug_mode_is_never_enabled(self, predictor):
        app = create_app(Config(), predictor=predictor)
        assert app.debug is False

    def test_upload_limit_is_configured(self, predictor):
        config = Config()
        config.serve.max_upload_mb = 8
        app = create_app(config, predictor=predictor)
        assert app.config["MAX_CONTENT_LENGTH"] == 8 * 1024 * 1024

    def test_unhandled_errors_return_generic_json(self, predictor, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("secret internal detail /etc/passwd")

        monkeypatch.setattr(Predictor, "predict_edf", explode)
        app = create_app(Config(), predictor=predictor)
        app.config.update(TESTING=False)
        client = app.test_client()
        response = client.post(
            "/api/v1/predict",
            data={"file": (io.BytesIO(b"x" * 100), "a.edf")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 500
        body = response.get_data(as_text=True)
        assert "secret internal detail" not in body
        assert response.get_json()["error"] == "internal_error"


@pytest.mark.slow
class TestRealDataEpochingParity:
    """The serving epocher must reproduce the training epocher exactly.

    Both paths cut windows out of the same EDF. If they disagree by even a
    sample the model is being fed something it was not trained on -- which is
    precisely the drift that made the previous version's web app unreliable.
    """

    def _predictor_for(self, real_data_root, tmp_path):
        from bwt.artifacts import ModelCard, save_artifact
        from bwt.data.epochs import load_subject_epochs
        from bwt.pipelines import build_pipeline

        bundle = load_subject_epochs(1, "mi_left_right", root=real_data_root)
        model = build_pipeline("csp_lda", sfreq=bundle.sfreq, n_classes=2)
        model.fit(bundle.X, bundle.y)
        card = ModelCard(
            name="parity", task=bundle.task, task_description="parity check",
            pipeline="csp_lda", classes=list(bundle.classes), sfreq=bundle.sfreq,
            n_channels=bundle.n_channels, n_times=bundle.n_times,
            ch_names=list(bundle.ch_names), tmin=bundle.tmin, tmax=bundle.tmax,
            chance_level=0.5,
        )
        path = save_artifact(model, card, path=tmp_path / "parity")
        return Predictor.load(path), bundle

    def test_serving_epochs_match_training_epochs(self, real_data_root, tmp_path):
        from bwt.data.epochs import epochs_from_raw, read_standardised_raw
        from bwt.data.physionet import edf_path, get_task

        predictor, bundle = self._predictor_for(real_data_root, tmp_path)
        path = edf_path(1, 4, real_data_root)

        raw = read_standardised_raw(path)
        training = epochs_from_raw(raw, get_task("mi_left_right"), run=4,
                                   tmin=bundle.tmin, tmax=bundle.tmax)
        expected = training.get_data(copy=True) * 1e6

        served, _onsets, mode, _ = predictor.epochs_from_edf(path)

        assert mode == "cue_locked"
        assert served.shape == expected.shape
        np.testing.assert_allclose(served, expected.astype(np.float32),
                                   rtol=1e-5, atol=1e-4)

    def test_channel_order_matches_the_card(self, real_data_root, tmp_path):
        from bwt.data.physionet import edf_path

        predictor, _ = self._predictor_for(real_data_root, tmp_path)
        raw = __import__("bwt.data.epochs", fromlist=["x"]).read_standardised_raw(
            edf_path(1, 4, real_data_root)
        )
        picks = predictor._align_channels(raw.ch_names)
        assert [raw.ch_names[i] for i in picks] == list(predictor.card.ch_names)

    def test_predictions_agree_across_both_paths(self, real_data_root, tmp_path):
        from bwt.artifacts import load_artifact
        from bwt.data.epochs import epochs_from_raw, read_standardised_raw
        from bwt.data.physionet import edf_path, get_task

        predictor, bundle = self._predictor_for(real_data_root, tmp_path)
        path = edf_path(1, 4, real_data_root)

        raw = read_standardised_raw(path)
        training = epochs_from_raw(raw, get_task("mi_left_right"), run=4,
                                   tmin=bundle.tmin, tmax=bundle.tmax)
        X = (training.get_data(copy=True) * 1e6).astype(np.float32)

        model, card = load_artifact(tmp_path / "parity")
        direct = model.predict(X)
        served = predictor.predict_edf(path)
        via_service = np.array(
            [card.classes.index(p.label) for p in served.predictions]
        )
        np.testing.assert_array_equal(via_service, direct)


class TestElectrodeGeometry:
    """The 3D scene must be driven by the model's real montage."""

    def test_geometry_endpoint_returns_every_channel(self, client):
        payload = client.get("/api/v1/geometry").get_json()
        assert payload["n_channels"] == 64
        assert payload["unplaced"] == []
        assert len(payload["electrodes"]) == 64

    def test_geometry_matches_the_model_channel_order(self, client):
        model = client.get("/api/v1/model").get_json()
        geometry = client.get("/api/v1/geometry").get_json()
        assert [e["name"] for e in geometry["electrodes"]] == \
            model["input_contract"]["channels"]

    def test_left_and_right_electrodes_are_on_opposite_sides(self, client):
        by_name = {e["name"]: e for e in
                   client.get("/api/v1/geometry").get_json()["electrodes"]}
        # A mirrored scene would render the contralateral highlight backwards.
        assert by_name["C3"]["x"] < 0 < by_name["C4"]["x"]
        assert by_name["FC5"]["x"] < 0 < by_name["FC6"]["x"]

    def test_motor_regions_are_labelled(self, client):
        by_name = {e["name"]: e for e in
                   client.get("/api/v1/geometry").get_json()["electrodes"]}
        assert by_name["C3"]["region"] == "left_motor"
        assert by_name["C4"]["region"] == "right_motor"
        assert by_name["Cz"]["region"] == "midline_motor"
        assert by_name["Oz"]["region"] == "other"

    def test_coordinates_are_normalised_for_the_scene(self, client):
        electrodes = client.get("/api/v1/geometry").get_json()["electrodes"]
        for e in electrodes:
            for axis in ("x", "y", "z"):
                assert -1.5 <= e[axis] <= 1.5, f"{e['name']} {axis} out of range"


class TestBandPowerStream:
    """Band power is a display quantity, opt-in and never fed back to the model."""

    def test_absent_unless_requested(self, predictor, synthetic_bundle):
        from bwt.serving.app import _iter_events

        decoder = predictor.streaming_decoder()
        stream = _FakeStream(synthetic_bundle.X[:3])
        events = list(_iter_events(decoder, stream, with_band_power=False))
        assert all(e.band_power is None for e in events)

    def test_present_and_bounded_when_requested(self, predictor, synthetic_bundle):
        from bwt.serving.app import _iter_events

        decoder = predictor.streaming_decoder()
        stream = _FakeStream(synthetic_bundle.X[:6])
        events = list(_iter_events(decoder, stream, with_band_power=True))
        assert all(e.band_power is not None for e in events)
        for event in events:
            assert len(event.band_power) == predictor.card.n_channels
            assert all(0.0 <= v <= 1.0 for v in event.band_power)

    def test_does_not_change_the_predictions(self, predictor, synthetic_bundle):
        """Turning the visualiser on must not alter a single decode."""
        from bwt.serving.app import _iter_events

        def labels(flag):
            decoder = predictor.streaming_decoder()
            stream = _FakeStream(synthetic_bundle.X[:6])
            return [e.top_label for e in
                    _iter_events(decoder, stream, with_band_power=flag)]

        assert labels(False) == labels(True)


class _FakeStream:
    """Minimal stand-in for EDFStream over pre-cut epochs."""

    def __init__(self, epochs):
        self.epochs = epochs

    def __len__(self):
        return len(self.epochs)

    def __iter__(self):
        from bwt.streaming import StreamWindow

        for i, data in enumerate(self.epochs):
            yield StreamWindow(index=i, start_sample=i * 100,
                               onset_seconds=i * 0.5, data=data)
