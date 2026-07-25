"""A scikit-learn classifier that happens to be a neural network.

Wrapping the networks this way means :mod:`bwt.evaluation` needs no knowledge of
PyTorch: a deep model is cross-validated, persisted, and served through exactly
the same code as CSP+LDA. It also means the leakage guards apply automatically,
which matters more for deep models than classical ones -- they have enough
capacity to memorise a subject outright.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from bwt.logging_utils import get_logger

log = get_logger(__name__)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_device(device: str | None = None) -> str:
    """Pick a device, honouring an explicit request but falling back safely."""
    import torch

    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class TorchClassifier(BaseEstimator, ClassifierMixin):
    """Train one of :mod:`bwt.deep.modules` as an sklearn classifier.

    Input is the project's standard ``(trials, channels, times)`` array in
    microvolts. Normalisation statistics are computed on the training set only
    and stored on the estimator, so ``transform``-time behaviour is fixed at fit
    time rather than depending on the batch.

    Parameters
    ----------
    architecture
        ``eegnet``, ``shallownet``, or ``conformer``.
    validation_fraction
        Portion of the *training* data held out to choose the stopping epoch.
        This is not test data -- the outer cross-validation still holds out
        whole subjects -- but it does mean the epoch count is tuned on subjects
        the model has seen.
    """

    def __init__(
        self,
        architecture: str = "eegnet",
        sfreq: float = 160.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 300,
        patience: int = 40,
        validation_fraction: float = 0.15,
        dropout: float | None = None,
        label_smoothing: float = 0.0,
        device: str | None = "auto",
        random_state: int = 42,
        class_weight: str | None = "balanced",
        verbose: bool = False,
        module_kwargs: dict[str, Any] | None = None,
    ):
        self.architecture = architecture
        self.sfreq = sfreq
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.device = device
        self.random_state = random_state
        self.class_weight = class_weight
        self.verbose = verbose
        self.module_kwargs = module_kwargs

    # -- helpers ---------------------------------------------------------- #

    def _seed_everything(self) -> None:
        import torch

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _check_input(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"expected (trials, channels, times), got shape {X.shape}"
            )
        return X

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def _build(self, n_channels: int, n_times: int, n_classes: int):
        from bwt.deep.modules import build_module

        kwargs = dict(self.module_kwargs or {})
        if self.dropout is not None:
            kwargs["dropout"] = self.dropout
        return build_module(
            self.architecture,
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            sfreq=self.sfreq,
            **kwargs,
        )

    # -- sklearn API ------------------------------------------------------- #

    def fit(self, X, y, sample_weight=None):
        import torch

        X = self._check_input(X)
        y = np.asarray(y).astype(np.int64)
        self._seed_everything()

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError("need at least two classes to fit")
        remap = {c: i for i, c in enumerate(self.classes_)}
        y_indexed = np.array([remap[v] for v in y], dtype=np.int64)

        # Per-channel statistics from the training set only.
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-6
        Xn = self._normalise(X)

        self.device_ = resolve_device(self.device)
        self.n_channels_ = X.shape[1]
        self.n_times_ = X.shape[2]

        stratify = y_indexed if np.bincount(y_indexed).min() >= 2 else None
        if self.validation_fraction and len(X) > 20 and stratify is not None:
            X_tr, X_va, y_tr, y_va = train_test_split(
                Xn, y_indexed, test_size=self.validation_fraction,
                stratify=stratify, random_state=self.random_state,
            )
        else:
            X_tr, y_tr, X_va, y_va = Xn, y_indexed, None, None

        model = self._build(self.n_channels_, self.n_times_,
                            self.n_classes_).to(self.device_)

        weight = None
        if self.class_weight == "balanced":
            counts = np.bincount(y_tr, minlength=self.n_classes_).astype(float)
            counts[counts == 0] = 1.0
            weight = torch.tensor(
                (len(y_tr) / (self.n_classes_ * counts)), dtype=torch.float32,
                device=self.device_,
            )
        criterion = torch.nn.CrossEntropyLoss(
            weight=weight, label_smoothing=self.label_smoothing
        )
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.max_epochs
        )

        # The whole training set lives on the device for the duration of the
        # fit. An EEG study is small enough to fit comfortably in VRAM (a
        # 4,000-trial 64x481 set is under 500 MB), and keeping it resident
        # removes the per-batch host-to-device copy that otherwise dominates
        # wall time for networks this small.
        Xtr_t = torch.from_numpy(X_tr).unsqueeze(1).to(self.device_)
        ytr_t = torch.from_numpy(y_tr).to(self.device_)
        if X_va is not None:
            Xva_t = torch.from_numpy(X_va).unsqueeze(1).to(self.device_)
            yva_t = torch.from_numpy(y_va).to(self.device_)

        n_train = len(Xtr_t)
        batch_size = min(self.batch_size, max(2, n_train))
        generator = torch.Generator(device=self.device_)
        generator.manual_seed(self.random_state)

        best_score, best_state, best_epoch, stale = -np.inf, None, 0, 0
        self.history_: list[dict] = []

        def _snapshot():
            return {k: v.detach().clone() for k, v in model.state_dict().items()}

        for epoch in range(self.max_epochs):
            model.train()
            order = torch.randperm(n_train, device=self.device_,
                                   generator=generator)
            total = 0.0
            for start in range(0, n_train, batch_size):
                idx = order[start:start + batch_size]
                if len(idx) < 2:  # BatchNorm needs more than one sample
                    continue
                optimiser.zero_grad(set_to_none=True)
                loss = criterion(model(Xtr_t[idx]), ytr_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()
                total += float(loss.detach()) * len(idx)
            scheduler.step()
            train_loss = total / max(1, n_train)

            if X_va is None:
                best_state = _snapshot()
                best_epoch = epoch
                continue

            model.eval()
            with torch.no_grad():
                logits = model(Xva_t)
                val_loss = float(criterion(logits, yva_t))
                val_acc = float((logits.argmax(1) == yva_t).float().mean())
            self.history_.append(
                {"epoch": epoch, "train_loss": train_loss,
                 "val_loss": val_loss, "val_acc": val_acc}
            )

            # Accuracy first, loss as the tie-break.
            score = val_acc - 1e-4 * val_loss
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = _snapshot()
            else:
                stale += 1
                if stale >= self.patience:
                    break

            if self.verbose and epoch % 20 == 0:
                log.info("  epoch %3d train=%.4f val=%.4f acc=%.4f",
                         epoch, train_loss, val_loss, val_acc)

        del Xtr_t, ytr_t
        if X_va is not None:
            del Xva_t, yva_t
        if self.device_.startswith("cuda"):
            torch.cuda.empty_cache()

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.module_ = model
        self.best_epoch_ = best_epoch
        self.n_parameters_ = sum(p.numel() for p in model.parameters())
        return self

    def _forward(self, X) -> np.ndarray:
        import torch

        X = self._check_input(X)
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                f"model expects {self.n_channels_} channels, got {X.shape[1]}"
            )
        if X.shape[2] != self.n_times_:
            raise ValueError(
                f"model expects {self.n_times_} samples, got {X.shape[2]}"
            )
        Xn = self._normalise(X)

        outputs = []
        self.module_.eval()
        with torch.no_grad():
            for start in range(0, len(Xn), 256):
                batch = torch.from_numpy(Xn[start:start + 256]).unsqueeze(1)
                batch = batch.to(self.device_)
                outputs.append(
                    torch.softmax(self.module_(batch), dim=1).cpu().numpy()
                )
        return np.concatenate(outputs, axis=0)

    def predict_proba(self, X) -> np.ndarray:
        return self._forward(X)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self._forward(X).argmax(axis=1)]

    # -- transfer learning -------------------------------------------------- #

    def clone_for_finetuning(self, freeze_features: bool = False) -> "TorchClassifier":
        """Copy this fitted model so it can be adapted to a new subject.

        Used by :mod:`bwt.calibration`. With ``freeze_features`` only the final
        classification layer is trainable, which is the right choice when the
        calibration set is very small.
        """
        if not hasattr(self, "module_"):
            raise RuntimeError("fit the model before cloning it for fine-tuning")

        clone = TorchClassifier(**self.get_params())
        clone.classes_ = self.classes_.copy()
        clone.n_classes_ = self.n_classes_
        clone.mean_ = self.mean_.copy()
        clone.std_ = self.std_.copy()
        clone.device_ = self.device_
        clone.n_channels_ = self.n_channels_
        clone.n_times_ = self.n_times_
        clone.module_ = copy.deepcopy(self.module_)
        clone.n_parameters_ = self.n_parameters_
        clone.best_epoch_ = getattr(self, "best_epoch_", 0)

        if freeze_features:
            for name, param in clone.module_.named_parameters():
                param.requires_grad = ("classifier" in name or "head" in name)
        return clone

    def partial_fit(self, X, y, epochs: int = 30, lr: float | None = None):
        """Continue training an already-fitted model on new data.

        This is the fine-tuning step for per-user calibration: the network keeps
        everything it learned from the population and only adjusts to the new
        subject. Normalisation statistics are *not* recomputed -- they belong to
        the pretrained model.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if not hasattr(self, "module_"):
            raise RuntimeError("call fit() before partial_fit()")

        X = self._check_input(X)
        y = np.asarray(y).astype(np.int64)
        remap = {c: i for i, c in enumerate(self.classes_)}
        unknown = set(np.unique(y)) - set(remap)
        if unknown:
            raise ValueError(f"unseen classes in calibration data: {unknown}")
        y_indexed = np.array([remap[v] for v in y], dtype=np.int64)

        Xn = self._normalise(X)
        trainable = [p for p in self.module_.parameters() if p.requires_grad]
        optimiser = torch.optim.AdamW(
            trainable, lr=lr if lr is not None else self.lr / 10,
            weight_decay=self.weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xn).unsqueeze(1),
                          torch.from_numpy(y_indexed)),
            batch_size=min(self.batch_size, max(2, len(Xn))), shuffle=True,
        )

        self.module_.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimiser.zero_grad(set_to_none=True)
                loss = criterion(self.module_(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                optimiser.step()
        self.module_.eval()
        return self

    # -- persistence -------------------------------------------------------- #

    def __getstate__(self):
        """Persist weights on CPU so an artifact trained on GPU loads anywhere."""
        state = self.__dict__.copy()
        module = state.pop("module_", None)
        if module is not None:
            state["_module_state"] = {
                k: v.cpu() for k, v in module.state_dict().items()
            }
        state.pop("device_", None)
        return state

    def __setstate__(self, state):
        module_state = state.pop("_module_state", None)
        self.__dict__.update(state)
        if module_state is not None:
            self.device_ = resolve_device(self.device)
            module = self._build(self.n_channels_, self.n_times_,
                                 self.n_classes_)
            module.load_state_dict(module_state)
            self.module_ = module.to(self.device_).eval()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.three_d_array = True
        tags.input_tags.two_d_array = False
        return tags


__all__ = ["TorchClassifier", "resolve_device", "torch_available"]
