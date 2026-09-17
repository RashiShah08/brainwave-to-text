"""Every subject in the real database, through the served model.

One recording per subject (run 4, left/right imagery) goes through the file
decode and the stream exactly as an upload would. A synthetic suite cannot find
the recording that is subtly different from the rest -- a stray sampling rate,
an odd annotation, a channel label the standardiser does not know -- and this
one looks at all 109. Marked ``slow``; skips without the dataset or the model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.slow

RUN = 4


@pytest.fixture(scope="module")
def served():
    from bwt.config import Config
    from bwt.serving.predictor import Predictor

    config = Config.load()
    try:
        return Predictor.load(config.serve.model,
                              max_epochs=config.serve.max_epochs_per_request)
    except FileNotFoundError as exc:
        pytest.skip(f"served artifact not present: {exc}")


@pytest.fixture(scope="module")
def root():
    from bwt.paths import raw_data_dir

    path = raw_data_dir()
    if not (path / "S001").is_dir():
        pytest.skip("real dataset not present")
    return path


@pytest.mark.parametrize("subject", range(1, 110))
def test_every_subject_decodes_or_is_refused_for_a_stated_reason(served, root,
                                                                 subject):
    from bwt.data.physionet import EXCLUDED_SUBJECTS, edf_path
    from bwt.serving.predictor import InputContractError

    path = edf_path(subject, RUN, root)
    if not path.is_file():
        pytest.skip(f"{path.name} not present")

    try:
        batch = served.predict_edf(path)
    except InputContractError as exc:
        # Only a subject the project already excludes may be refused, and the
        # refusal must say why in terms a user can act on.
        assert subject in EXCLUDED_SUBJECTS, f"S{subject:03d} refused: {exc}"
        assert "Hz" in str(exc) or "channel" in str(exc)
        return

    classes = list(served.card.classes)
    proba = np.array([[p.probabilities[c] for c in classes] for p in batch.predictions])
    assert batch.epoching == "cue_locked", f"S{subject:03d} lost its cues"
    assert 10 <= batch.n <= 30, f"S{subject:03d}: {batch.n} trials"
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert all(math.isfinite(p.onset_seconds) and p.onset_seconds >= 0
               for p in batch.predictions)
    onsets = [p.onset_seconds for p in batch.predictions]
    assert onsets == sorted(onsets) and len(set(onsets)) == len(onsets)

    stream = served.stream_from_edf(path, step_seconds=1.0)
    assert len(stream) > 0
    first = next(iter(stream))
    single = served.predict_proba(served.validate_array(first.data[None]))
    assert np.isfinite(single).all()
