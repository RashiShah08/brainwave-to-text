"""An EEG motor-imagery decoder.

Decodes *which imagined movement* a recording contains, and nothing else.
Scalp EEG carries no language: it cannot recover words, inner speech, or
intent, and no method applied to it can. Earlier versions bolted a character
tree onto the classifier so that decoded commands selected letters; that is a
selection interface rather than decoding, it invited exactly the reading this
package is careful to deny, and it has been removed.


The package is organised so that the *exact same* fitted scikit-learn pipeline
object is used for training and for serving. Anything that touches the signal
lives inside that pipeline, which removes the class of bug where the training
and inference feature extractors drift apart.

Public surface:
    bwt.data      -- dataset discovery and epoching
    bwt.pipelines -- model registry (feature extraction + classifier)
    bwt.evaluation-- subject-aware cross-validation protocols
    bwt.artifacts -- versioned model persistence with a model card
    bwt.metrics   -- information transfer rate
    bwt.serving   -- Flask application
"""

__version__ = "3.0.0"

# Schema version for persisted artifacts. Bump whenever the meaning of a saved
# pipeline's input changes (channel order, sfreq, epoch window semantics) so
# that stale artifacts are rejected loudly instead of silently mispredicting.
ARTIFACT_SCHEMA_VERSION = 2

__all__ = ["ARTIFACT_SCHEMA_VERSION", "__version__"]
