"""Causal TCN VAD: dilated causal convolutions -> per-frame sigmoid."""
from __future__ import annotations
import torch.nn as nn


class CausalTCN(nn.Module):
    def __init__(self, n_mels=64, channels=(64, 64, 64, 64),
                 kernel_size=3, causal=True):
        super().__init__()
        # TODO: stacked dilated causal conv blocks (dilations 1,2,4,8) + head
        raise NotImplementedError

    def forward(self, x):            # x: [B, T, n_mels]
        raise NotImplementedError
