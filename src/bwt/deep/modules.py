"""PyTorch architectures for EEG decoding.

All three take input shaped ``(batch, 1, n_channels, n_times)`` and return
class logits. Kernel and pooling sizes in the published papers are quoted for
the sampling rate those papers used, so every architecture here rescales them
from its ``sfreq`` argument rather than hardcoding numbers that are silently
wrong at 160 Hz.

References
----------
EEGNet
    Lawhern et al. (2018), "EEGNet: a compact convolutional neural network for
    EEG-based brain-computer interfaces", J. Neural Eng. 15(5).
ShallowConvNet
    Schirrmeister et al. (2017), "Deep learning with convolutional neural
    networks for EEG decoding and visualization", Hum. Brain Mapp. 38(11).
EEGConformer
    Song et al. (2023), "EEG Conformer: Convolutional Transformer for EEG
    Decoding and Visualization", IEEE TNSRE 31.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _scaled(reference_samples: int, reference_sfreq: float, sfreq: float,
            minimum: int = 1) -> int:
    """Rescale a kernel length quoted at one sampling rate to another."""
    return max(minimum, int(round(reference_samples * sfreq / reference_sfreq)))


class _MaxNormConv2d(nn.Conv2d):
    """Conv2d with a max-norm constraint on its weights.

    EEGNet's depthwise spatial convolution relies on this constraint; without it
    the spatial filters grow without bound and the network overfits badly on the
    small trial counts typical of EEG.
    """

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        with torch.no_grad():
            # vector_norm, not Tensor.norm: the latter routes a 3-tuple `dim`
            # to matrix_norm, which only accepts two dimensions.
            norm = torch.linalg.vector_norm(
                self.weight, dim=(1, 2, 3), keepdim=True
            ).clamp(min=1e-8)
            self.weight.mul_(norm.clamp(max=self.max_norm) / norm)
        return super().forward(x)


class _MaxNormLinear(nn.Linear):
    def __init__(self, *args, max_norm: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        with torch.no_grad():
            norm = torch.linalg.vector_norm(
                self.weight, dim=1, keepdim=True
            ).clamp(min=1e-8)
            self.weight.mul_(norm.clamp(max=self.max_norm) / norm)
        return super().forward(x)


class EEGNet(nn.Module):
    """EEGNet v4.

    A deliberately tiny network (a few thousand parameters). Its depthwise
    convolution across channels learns spatial filters that play the same role
    as CSP, and the separable convolution that follows learns the temporal
    pattern. The small parameter count is what makes it viable on the few
    thousand trials an EEG study produces.
    """

    def __init__(self, n_channels: int, n_times: int, n_classes: int,
                 sfreq: float = 160.0, F1: int = 8, D: int = 2,
                 F2: int | None = None, dropout: float = 0.25):
        super().__init__()
        F2 = F2 or F1 * D
        # The paper's temporal kernel spans half a second.
        kernel_length = _scaled(64, 128.0, sfreq)
        separable_length = _scaled(16, 128.0, sfreq)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            _MaxNormConv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False,
                           max_norm=1.0),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, separable_length),
                      padding=(0, separable_length // 2), groups=F1 * D,
                      bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            n_features = self.block2(self.block1(dummy)).flatten(1).shape[1]
        self.classifier = _MaxNormLinear(n_features, n_classes, max_norm=0.25)

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)).flatten(1))


class _Square(nn.Module):
    def forward(self, x):
        return x * x


class _SafeLog(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(nn.Module):
    """ShallowConvNet.

    Explicitly designed to mirror the FBCSP pipeline: a temporal convolution, a
    spatial convolution, then square -> average-pool -> log, which is exactly
    the log-band-power computation CSP feeds to its classifier. It usually beats
    deeper networks on motor imagery precisely because it encodes that prior.
    """

    def __init__(self, n_channels: int, n_times: int, n_classes: int,
                 sfreq: float = 160.0, n_filters_time: int = 40,
                 n_filters_spatial: int = 40, dropout: float = 0.5):
        super().__init__()
        filter_time = _scaled(25, 250.0, sfreq)
        pool_size = _scaled(75, 250.0, sfreq)
        pool_stride = _scaled(15, 250.0, sfreq)

        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time), bias=False),
            nn.Conv2d(n_filters_time, n_filters_spatial, (n_channels, 1),
                      bias=False),
            nn.BatchNorm2d(n_filters_spatial),
            _Square(),
            nn.AvgPool2d((1, pool_size), stride=(1, pool_stride)),
            _SafeLog(),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            n_features = self.features(dummy).flatten(1).shape[1]
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class EEGConformer(nn.Module):
    """EEG-Conformer: shallow convolutional tokeniser + transformer encoder.

    The convolutional front end is ShallowConvNet's, which produces a short
    sequence of band-power-like tokens; self-attention then models the
    relationships between those time windows. More capacity than the other two,
    and correspondingly more prone to overfitting on small trial counts.
    """

    def __init__(self, n_channels: int, n_times: int, n_classes: int,
                 sfreq: float = 160.0, d_model: int = 40, n_heads: int = 4,
                 depth: int = 4, dropout: float = 0.3):
        super().__init__()
        filter_time = _scaled(25, 250.0, sfreq)
        pool_size = _scaled(75, 250.0, sfreq)
        pool_stride = _scaled(15, 250.0, sfreq)

        self.tokeniser = nn.Sequential(
            nn.Conv2d(1, d_model, (1, filter_time), bias=False),
            nn.Conv2d(d_model, d_model, (n_channels, 1), bias=False),
            nn.BatchNorm2d(d_model),
            _Square(),
            nn.AvgPool2d((1, pool_size), stride=(1, pool_stride)),
            _SafeLog(),
            nn.Dropout(dropout),
        )
        self.positional = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            n_tokens = self.tokeniser(dummy).squeeze(2).shape[-1]
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
        self.n_tokens = n_tokens

    def forward(self, x):
        tokens = self.tokeniser(x).squeeze(2).transpose(1, 2)  # (B, T, d_model)
        tokens = self.positional(tokens)
        encoded = self.encoder(tokens)
        return self.head(encoded.mean(dim=1))


ARCHITECTURES = {
    "eegnet": EEGNet,
    "shallownet": ShallowConvNet,
    "conformer": EEGConformer,
}


def build_module(name: str, **kwargs) -> nn.Module:
    try:
        factory = ARCHITECTURES[name]
    except KeyError:
        raise ValueError(
            f"unknown architecture {name!r}; available: "
            f"{', '.join(sorted(ARCHITECTURES))}"
        ) from None
    return factory(**kwargs)


__all__ = [
    "ARCHITECTURES",
    "EEGConformer",
    "EEGNet",
    "ShallowConvNet",
    "build_module",
]
