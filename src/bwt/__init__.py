"""Brainwave-to-Text: an EEG motor-imagery decoder and mental-command speller.

The package is organised so that the *exact same* fitted scikit-learn pipeline
object is used for training and for serving. Anything that touches the signal
lives inside that pipeline, which removes the class of bug where the training
and inference feature extractors drift apart.

Public surface:
    bwt.data      -- dataset discovery and epoching
    bwt.pipelines -- model registry (feature extraction + classifier)
    bwt.evaluation-- subject-aware cross-validation protocols
    bwt.artifacts -- versioned model persistence with a model card
    bwt.decoding  -- mental-command -> text speller
    bwt.serving   -- Flask application
"""

__version__ = "3.0.0"

# Schema version for persisted artifacts. Bump whenever the meaning of a saved
# pipeline's input changes (channel order, sfreq, epoch window semantics) so
# that stale artifacts are rejected loudly instead of silently mispredicting.
ARTIFACT_SCHEMA_VERSION = 2

__all__ = ["ARTIFACT_SCHEMA_VERSION", "__version__"]
