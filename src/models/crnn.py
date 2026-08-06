"""Causal separable-conv + GRU VAD (MarbleNet / MagicNet inspired).

Design rationale (see report Method section):
  - 1D depthwise-separable convolutions over time, with the mel bins as input
    channels -> strong parameter efficiency (MarbleNet-style).
  - Left-only (causal) padding + a unidirectional GRU -> streaming with state
    carry-over and no look-ahead (MagicNet-style).
  - Per-frame sigmoid head -> frame-level speech probability.

Input : x [B, T, n_mels]   (log-mel features from features.py)
Output: logits [B, T]      (apply sigmoid for probabilities)

Causality: every conv pads on the left only and the GRU is unidirectional, so
output t depends on inputs <= t. BatchNorm uses running statistics at inference
(model.eval()), so streaming inference is causal; verify with the causality
test in eval mode.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise 1D conv with causal (left) padding."""
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation, groups=channels)

    def forward(self, x):                      # x: [B, C, T]
        return self.conv(F.pad(x, (self.pad, 0)))


class SepBlock(nn.Module):
    """Causal depthwise-separable block: depthwise -> pointwise -> BN -> ReLU.

    Residual when in/out channels match (helps gradient flow, negligible cost).
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5,
                 dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.depthwise = CausalDepthwiseConv1d(in_ch, kernel_size, dilation)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.residual = (in_ch == out_ch)

    def forward(self, x):                      # x: [B, C, T]
        y = self.pointwise(self.depthwise(x))
        y = self.drop(F.relu(self.bn(y)))
        return y + x if self.residual else y


class SepConvGRUVAD(nn.Module):
    """Lightweight causal VAD: separable conv stack + unidirectional GRU."""
    def __init__(self, n_mels: int = 64, channels: int = 64, n_blocks: int = 4,
                 kernel_size: int = 5, gru_hidden: int = 64, gru_layers: int = 1,
                 dropout: float = 0.1,
                 dilations: tuple = (1, 2, 4, 8), causal: bool = True):
        super().__init__()
        assert causal, "this model is designed to be causal"
        self.stem = nn.Conv1d(n_mels, channels, kernel_size=1)
        dils = (dilations * n_blocks)[:n_blocks]
        self.blocks = nn.ModuleList(
            SepBlock(channels, channels, kernel_size, d, dropout) for d in dils)
        self.gru = nn.GRU(channels, gru_hidden, num_layers=gru_layers,
                          batch_first=True, bidirectional=False)
        self.head = nn.Linear(gru_hidden, 1)

    def forward(self, x, h=None):              # x: [B, T, n_mels]
        z = self.stem(x.transpose(1, 2))       # [B, C, T]
        for blk in self.blocks:
            z = blk(z)
        z = z.transpose(1, 2)                  # [B, T, C]
        z, h = self.gru(z, h)                  # unidirectional
        logits = self.head(z).squeeze(-1)      # [B, T]
        return logits

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Backwards-compatible alias used by src/train.py:build_model
CausalCRNN = SepConvGRUVAD
