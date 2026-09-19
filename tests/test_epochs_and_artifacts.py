"""Tests for the epoch container, caching, and model artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bwt import ARTIFACT_SCHEMA_VERSION
from bwt.artifacts import (
    ModelCard,
    list_artifacts,
    load_artifact,
    resolve_artifact,
    save_artifact,
)
from bwt.data.epochs import EpochBundle, concat_bundles


class TestEpochBundle:
    def test_rejects_mismatched_lengths(self, synthetic_bundle):
        with pytest.raises(ValueError, match="rows"):
            EpochBundle(
                X=synthetic_bundle.X,
                y=synthetic_bundle.y[:-1],
                groups=synthetic_bundle.groups,
                runs=synthetic_bundle.runs,
                classes=synthetic_bundle.classes,
                ch_names=synthetic_bundle.ch_names,
                sfreq=160.0, tmin=0.5, tmax=3.5, task="t",
            )

    def test_rejects_channel_name_count_mismatch(self, synthetic_bundle):
        with pytest.raises(ValueError, match="channels"):
            EpochBundle(
                X=synthetic_bundle.X,
                y=synthetic_bundle.y,
                groups=synthetic_bundle.groups,
                runs=synthetic_bundle.runs,
                classes=synthetic_bundle.classes,
                ch_names=("A", "B"),
                sfreq=160.0, tmin=0.5, tmax=3.5, task="t",
            )

    def test_subject_subsetting(self, synthetic_bundle):
        one = synthetic_bundle.for_subject(2)
        assert set(np.unique(one.groups)) == {2}
        assert one.classes == synthetic_bundle.classes
        assert one.n_trials < synthetic_bundle.n_trials

    def test_round_trip_through_disk(self, synthetic_bundle, tmp_path):
        path = tmp_path / "bundle.npz"
        synthetic_bundle.save(path)
        assert path.is_file(), "save must land on the requested filename"

        loaded = EpochBundle.load(path)
        np.testing.assert_array_equal(loaded.X, synthetic_bundle.X)
        np.testing.assert_array_equal(loaded.y, synthetic_bundle.y)
        np.testing.assert_array_equal(loaded.groups, synthetic_bundle.groups)
        assert loaded.ch_names == synthetic_bundle.ch_names
        assert loaded.classes == synthetic_bundle.classes
        assert loaded.sfreq == synthetic_bundle.sfreq

    def test_save_leaves_no_temp_file(self, synthetic_bundle, tmp_path):
        synthetic_bundle.save(tmp_path / "b.npz")
        assert list(tmp_path.glob("*.tmp*")) == []

    def test_metadata_is_json_serialisable(self, synthetic_bundle):
        payload = json.loads(json.dumps(synthetic_bundle.metadata()))
        assert payload["n_subjects"] == 6
        assert payload["units"] == "uV"

    def test_class_counts_sum_to_trials(self, synthetic_bundle):
        assert sum(synthetic_bundle.class_counts().values()) == synthetic_bundle.n_trials


class TestConcat:
    def test_refuses_mismatched_channels(self, synthetic_bundle):
        other = EpochBundle(
            X=synthetic_bundle.X[:4],
            y=synthetic_bundle.y[:4],
            groups=synthetic_bundle.groups[:4],
            runs=synthetic_bundle.runs[:4],
            classes=synthetic_bundle.classes,
            ch_names=tuple(reversed(synthetic_bundle.ch_names)),
            sfreq=160.0, tmin=0.5, tmax=3.5, task="t",
        )
        with pytest.raises(ValueError, match="channel names differ"):
            concat_bundles([synthetic_bundle, other])

    def test_refuses_mismatched_epoch_length(self, synthetic_bundle):
        other = EpochBundle(
            X=synthetic_bundle.X[:4, :, :100],
            y=synthetic_bundle.y[:4],
            groups=synthetic_bundle.groups[:4],
            runs=synthetic_bundle.runs[:4],
            classes=synthetic_bundle.classes,
            ch_names=synthetic_bundle.ch_names,
            sfreq=160.0, tmin=0.5, tmax=3.5, task="t",
        )
        with pytest.raises(ValueError, match="epoch length differs"):
            concat_bundles([synthetic_bundle, other])


class TestArtifacts:
    def test_save_then_load(self, trained_artifact, synthetic_bundle):
        model, card = load_artifact(trained_artifact)
        assert card.classes == list(synthetic_bundle.classes)
        assert card.n_channels == synthetic_bundle.n_channels
        predictions = model.predict(synthetic_bundle.X[:5])
        assert len(predictions) == 5

    def test_card_records_the_input_contract(self, trained_artifact):
        _, card = load_artifact(trained_artifact)
        assert card.sfreq == 160.0
        assert card.n_times == 481
        assert len(card.ch_names) == card.n_channels
        assert card.units == "uV"

    def test_refuses_artifact_without_a_card(self, trained_artifact):
        (trained_artifact / "model_card.json").unlink()
        with pytest.raises(FileNotFoundError, match="undocumented"):
            load_artifact(trained_artifact)

    def test_refuses_incompatible_schema_version(self, trained_artifact):
        card_file = trained_artifact / "model_card.json"
        payload = json.loads(card_file.read_text())
        payload["schema_version"] = ARTIFACT_SCHEMA_VERSION + 99
        card_file.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="schema version"):
            load_artifact(trained_artifact)

    def test_missing_directory_gives_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="bwt train"):
            load_artifact(tmp_path / "absent")

    def test_headline_states_chance_level(self, trained_artifact):
        _, card = load_artifact(trained_artifact)
        assert "chance" in card.headline()

    def test_headline_when_unevaluated(self):
        card = ModelCard(
            name="x", task="t", task_description="d", pipeline="p",
            classes=["a", "b"], sfreq=160.0, n_channels=64, n_times=481,
            ch_names=["c"] * 64, tmin=0.5, tmax=3.5,
        )
        assert card.headline() == "no evaluation recorded"

    def test_listing_and_resolution(self, trained_artifact):
        root = trained_artifact.parent
        found = list_artifacts(root)
        assert len(found) == 1
        assert resolve_artifact(None, root) == trained_artifact
        assert resolve_artifact(trained_artifact.name, root) == trained_artifact

    def test_resolution_failure_is_explicit(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_artifact("nonexistent", tmp_path)
