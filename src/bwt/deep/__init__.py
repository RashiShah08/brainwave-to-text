"""Neural network decoders.

These are the published convolutional architectures for EEG decoding, wrapped
so that they are ordinary scikit-learn classifiers. That wrapping is the whole
point: :mod:`bwt.evaluation`, :mod:`bwt.artifacts` and the serving layer treat a
deep model exactly like CSP+LDA, so the same subject-grouped cross-validation
and the same train/serve parity guarantees apply without a parallel code path.

Importing this module does not require PyTorch. The dependency is only pulled in
when a network is actually constructed, so a deployment that serves a classical
pipeline does not need a 2 GB CUDA install.
"""

from bwt.deep.estimator import TorchClassifier, torch_available

__all__ = ["TorchClassifier", "torch_available"]
