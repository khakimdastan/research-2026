"""Lightweight 1D CNN architectures.

Both networks accept input of shape `(B, 1, window_size)` and use global
average pooling, so they generalise to any window length.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """Compact baseline 1D CNN — three conv blocks + GAP + linear head."""

    def __init__(self, n_classes: int, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=2, padding=32),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=32, stride=1, padding=16),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).squeeze(-1)
        return self.classifier(x)


class _DSConv1D(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.depth = nn.Conv1d(c_in, c_in, kernel_size=kernel,
                                stride=stride, padding=padding,
                                groups=c_in, bias=False)
        self.point = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)

    def forward(self, x):
        return self.point(self.depth(x))


class DSCNN1D(nn.Module):
    """Depthwise-separable 1D CNN — same depth as `CNN1D` but ~4x fewer params."""

    def __init__(self, n_classes: int, in_channels: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=2, padding=32, bias=False),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        self.block1 = nn.Sequential(
            _DSConv1D(16, 32, kernel=32, stride=1, padding=16),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        self.block2 = nn.Sequential(
            _DSConv1D(32, 64, kernel=16, stride=1, padding=8),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x).squeeze(-1)
        return self.classifier(x)


class CNN1D_FFT(nn.Module):
    """1D CNN that consumes the magnitude spectrum of the input window.

    The Fourier transform layer is non-trainable and acts as a fixed
    frequency-domain front-end; everything after it is a small CNN.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.cnn = CNN1D(n_classes, in_channels=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) -> magnitude spectrum (B, 1, T//2 + 1)
        spec = torch.fft.rfft(x, dim=-1).abs()
        spec = spec / (spec.amax(dim=-1, keepdim=True) + 1e-8)
        return self.cnn(spec)


class WDCNN(nn.Module):
    """Wide-Deep CNN (Zhang et al. 2017).

    The defining feature is a very wide first-layer kernel (size 64, stride 16)
    designed to act as an automatic noise-robust feature extractor on raw
    vibration signals. Subsequent layers use small kernels (size 3) and
    progressive downsampling. The original paper used `~50k` parameters and
    is the most cited 1D-CNN baseline in bearing fault diagnosis literature.
    """

    def __init__(self, n_classes: int, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).squeeze(-1)
        return self.classifier(x)


class _ActivationEnsemble(nn.Module):
    """Average of three lightweight non-linearities (HardTanh, ReLU6, HardSwish).

    Used as the activation in LSR-Net to provide a parameter-free 'ensemble'
    that empirically improves robustness under noise.
    """

    def __init__(self):
        super().__init__()
        self.ht  = nn.Hardtanh()
        self.r6  = nn.ReLU6()
        self.hsw = nn.Hardswish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.ht(x) + self.r6(x) + self.hsw(x)) / 3.0


class _LSRBlock(nn.Module):
    """LSR-Net block: depthwise conv + grouped pointwise expansion + ensemble act."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 5,
                 expansion: int = 2, groups: int = 2):
        super().__init__()
        c_mid = c_in * expansion
        self.dw     = nn.Conv1d(c_in, c_in, kernel_size=kernel,
                                 padding=kernel // 2, groups=c_in, bias=False)
        self.pw_exp = nn.Conv1d(c_in,  c_mid, kernel_size=1, groups=groups, bias=False)
        self.pw_red = nn.Conv1d(c_mid, c_out, kernel_size=1, groups=groups, bias=False)
        self.bn1    = nn.BatchNorm1d(c_mid)
        self.bn2    = nn.BatchNorm1d(c_out)
        self.act    = _ActivationEnsemble()

    def forward(self, x):
        x = self.dw(x)
        x = self.act(self.bn1(self.pw_exp(x)))
        x = self.bn2(self.pw_red(x))
        return x


class LSRNet(nn.Module):
    """LSR-Net: Lightweight, Strong-Robustness 1D Network.

    Implements the key ingredients reported in the LSR-Net paper: group
    pointwise convolutions for parameter efficiency, and a parameter-free
    activation ensemble (HardTanh + ReLU6 + HardSwish averaged) for noise
    robustness on edge devices.
    """

    def __init__(self, n_classes: int, in_channels: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(16),
            _ActivationEnsemble(),
            nn.MaxPool1d(2),
        )
        self.block1 = _LSRBlock(16, 32, kernel=5,  expansion=2, groups=2)
        self.pool1  = nn.MaxPool1d(2)
        self.block2 = _LSRBlock(32, 64, kernel=5,  expansion=2, groups=2)
        self.pool2  = nn.MaxPool1d(2)
        self.block3 = _LSRBlock(64, 64, kernel=3,  expansion=2, groups=2)
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


def build_cnn(name: str, n_classes: int) -> nn.Module:
    name = name.lower()
    if name == "cnn1d":
        return CNN1D(n_classes)
    if name == "dscnn1d":
        return DSCNN1D(n_classes)
    if name == "cnn1d_fft":
        return CNN1D_FFT(n_classes)
    if name == "wdcnn":
        return WDCNN(n_classes)
    if name == "lsrnet":
        return LSRNet(n_classes)
    raise ValueError(f"Unknown CNN model '{name}'")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total / (1024 ** 2)
