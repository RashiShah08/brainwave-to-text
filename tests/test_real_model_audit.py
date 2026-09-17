"""Audit of the model that is actually served, against the real recordings.

This replaces the ad-hoc ``scripts/audit_model.py`` and
``scripts/verify_no_hardcoding.py``, which hardcoded a developer's absolute
paths and needed a running server. Everything here runs in-process against
the artifact named in ``configs/default.yaml`` and skips itself when that
artifact or the dataset is absent, so it is marked ``slow``.

It asks the questions a synthetic model cannot answer: are the labels the
recording protocol's labels, is every served number the estimator's number, is
the input really microvolts, and does the model degenerate on real data.
"""

from __future__ import annotations

import io
import json
import math
from collections import Counter

import numpy as np
import pytest

pytestmark = pytest.mark.slow

RECORDING = ("S003", "S003R04.edf")


@pytest.fixture(scope="module")
def served():
    from bwt.config import Config
    from bwt.serving.predictor import Predictor

    config = Config.load()
    try:
        predictor = Predictor.load(config.serve.model,
                                   max_epochs=config.serve.max_epochs_per_request)
    except FileNotFoundError as exc:
        pytest.skip(f"served artifact not present: {exc}")
    return config, predictor


@pytest.fixture(scope="module")
def recording(real_data_root_module):
    path = real_data_root_module.joinpath(*RECORDING)
    if not path.is_file():
        pytest.skip(f"{path} not present")
    return path


@pytest.fixture(scope="module")
def real_data_root_module():
    from bwt.paths import raw_data_dir

    root = raw_data_dir()
    if not (root / "S001").is_dir():
        pytest.skip("real dataset not present")
    return root


@pytest.fixture(scope="module")
def client(served):
    from bwt.serving.app import create_app

    config, predictor = served
    app = create_app(config, predictor=predictor)
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture(scope="module")
def bundle(served, real_data_root_module):
    from bwt.data import load_bundle

    _, predictor = served
    card = predictor.card
    return load_bundle(card.task, subjects=[1, 2, 3, 42, 77], tmin=card.tmin,
                       tmax=card.tmax, root=real_data_root_module, n_jobs=2)


class TestLabels:
    def test_card_class_order_is_the_task_class_order(self, served):
        from bwt.data.physionet import TASKS

        _, predictor = served
        assert list(predictor.card.classes) == list(TASKS[predictor.card.task].classes)

    def test_bundle_labels_equal_the_annotations_under_the_run_protocol(
            self, served, bundle, real_data_root_module):
        import mne

        from bwt.data.physionet import TASKS, edf_path, run_spec

        task = TASKS[served[1].card.task]
        checked = 0
        for subject in bundle.subjects:
            for run in task.runs:
                path = edf_path(subject, run, real_data_root_module)
                if not path.is_file():
                    continue
                raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
                spec = run_spec(run)
                expected = [
                    task.classes.index(task.label_map[spec.annotation_map[d]])
                    for d in raw.annotations.description
                    if d in spec.annotation_map
                    and spec.annotation_map[d] in task.label_map
                ]
                got = bundle.y[(bundle.groups == subject) & (bundle.runs == run)]
                assert got.tolist() == expected, (subject, run)
                checked += 1
        assert checked >= 10

    def test_no_excluded_subject_was_trained_on(self, served):
        from bwt.data.physionet import EXCLUDED_SUBJECTS

        assert not set(served[1].card.train_subjects) & EXCLUDED_SUBJECTS


class TestSignalContract:
    def test_epochs_are_microvolts_of_the_declared_length(self, served, bundle):
        card = served[1].card
        assert bundle.X.shape[1:] == (card.n_channels, card.n_times)
        assert card.n_times == round((card.tmax - card.tmin) * card.sfreq) + 1
        assert np.isfinite(bundle.X).all()
        median = float(np.median(np.abs(bundle.X)))
        assert 1.0 < median < 500.0, f"median |x| = {median} -- not microvolts"

    def test_no_duplicate_trials(self, bundle):
        flat = bundle.X.reshape(len(bundle.X), -1)[:, ::97]
        assert len({row.tobytes() for row in flat}) == len(flat)

    def test_shuffled_channel_order_is_restored_by_name(self, served):
        _, predictor = served
        names = list(predictor.card.ch_names)
        shuffled = names.copy()
        np.random.default_rng(0).shuffle(shuffled)
        picks = predictor._align_channels(shuffled)
        assert [shuffled[i] for i in picks] == names


class TestServedNumbersAreTheEstimators:
    def test_decode_endpoint_equals_the_estimator(self, served, client, recording):
        _, predictor = served
        X, _, _, _ = predictor.epochs_from_edf(recording)
        proba = predictor.model.predict_proba(predictor.validate_array(X))

        response = client.post("/api/v1/predict", data={
            "file": (io.BytesIO(recording.read_bytes()), recording.name)},
            content_type="multipart/form-data")
        body = response.get_json()
        classes = list(predictor.card.classes)
        served_proba = np.array([[p["probabilities"][c] for c in classes]
                                 for p in body["predictions"]])

        assert body["n_epochs"] == len(proba)
        assert [p["label"] for p in body["predictions"]] == \
            [classes[i] for i in proba.argmax(1)]
        np.testing.assert_allclose(served_proba, proba, atol=6e-5)
        assert np.allclose(proba.sum(1), 1.0, atol=1e-9)

    def test_metadata_endpoints_equal_the_card_and_the_montage(self, served,
                                                               client):
        import mne

        card = served[1].card
        model = client.get("/api/v1/model").get_json()
        assert model["card"]["classes"] == list(card.classes)
        assert model["input_contract"]["n_channels"] == card.n_channels

        geometry = client.get("/api/v1/geometry").get_json()
        names = [e["name"] for e in geometry["electrodes"]]
        assert names == list(card.ch_names) and geometry["unplaced"] == []
        positions = mne.channels.make_standard_montage("standard_1005") \
            .get_positions()["ch_pos"]
        lookup = {k.upper(): v for k, v in positions.items()}
        raw = np.array([lookup[n.upper()] for n in names])
        centred = raw - raw.mean(0)
        centred /= np.abs(centred).max()
        expected = np.stack([centred[:, 0], centred[:, 2], -centred[:, 1]], 1)
        served_xyz = np.array([[e["x"], e["y"], e["z"]] for e in geometry["electrodes"]])
        np.testing.assert_allclose(served_xyz, expected, atol=1.5e-4)

    def test_stream_trace_is_the_files_own_samples(self, served, client, recording):
        import mne

        response = client.post("/api/v1/stream?raw=1&band_power=1&speed=0&step=0.5",
                               data={"file": (io.BytesIO(recording.read_bytes()),
                                              recording.name)},
                               content_type="multipart/form-data")
        events = [json.loads(line) for line in response.get_data(as_text=True)
                  .splitlines() if line]
        assert events[0]["type"] == "start" and events[-1]["type"] in {"end", "truncated"}
        windows = [e for e in events if e.get("type") == "window"]
        trace = events[0]["trace"]
        samples = np.concatenate([np.asarray(w["raw"]["samples"]) for w in windows], 1)

        edf = mne.io.read_raw_edf(recording, preload=True, verbose="ERROR")
        edf.rename_channels({c: c.strip(".").upper() for c in edf.ch_names})
        data = edf.get_data(picks=[c.upper() for c in trace["channels"]]) * 1e6
        first = round(windows[0]["onset_seconds"] * trace["sfreq"]) \
            + served[1].card.n_times - trace["samples_per_step"]
        np.testing.assert_allclose(samples[:, :400], data[:, first:first + 400],
                                   atol=0.051)

        posterior = np.array([list(w["posterior"].values()) for w in windows])
        assert np.allclose(posterior.sum(1), 1.0, atol=len(trace["channels"]) * 5e-5)
        for w in windows:
            d = w.get("decision")
            if d and not d["timed_out"]:
                assert d["confidence"] >= 0.9 - 1e-4
                assert math.isclose(d["confidence"], max(w["posterior"].values()),
                                    abs_tol=1e-4)


class TestBehaviourOnRealData:
    def test_determinism(self, served, recording):
        _, predictor = served
        X, onsets, mode, _ = predictor.epochs_from_edf(recording)
        a = predictor.predict_array(X, onsets=onsets, source=mode)
        b = predictor.predict_array(X, onsets=onsets, source=mode)
        assert [p.probabilities for p in a] == [p.probabilities for p in b]

    def test_model_does_not_collapse_onto_one_class(self, served, bundle):
        _, predictor = served
        predicted = predictor.model.predict(predictor.validate_array(bundle.X))
        share = Counter(predicted.tolist())
        assert len(share) == len(predictor.card.classes)
        assert min(share.values()) / len(predicted) > 0.10, share

    def test_transductive_recentring_is_stable_across_batch_sizes(self, served,
                                                                  recording):
        _, predictor = served
        if not predictor.card.requires_batch_recentering:
            pytest.skip("served model is not transductive")
        X, onsets, mode, _ = predictor.epochs_from_edf(recording)
        full = predictor.predict_array(X, onsets=onsets, source=mode)
        half = predictor.predict_array(X[:8], onsets=onsets[:8], source=mode)
        agree = sum(a.label == b.label for a, b in zip(full[:8], half, strict=True))
        assert agree >= 6, f"{agree}/8 -- recentring moves with batch size"
