"""Reference baselines — NOT the submitted solution.

Energy-based (webrtcvad / auditok) and pretrained neural (Silero, torch.hub).
Used only to position our trained model on the same axes.
"""
from __future__ import annotations
import numpy as np


def energy_vad(wav: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame decisions from webrtcvad or auditok."""
    raise NotImplementedError  # TODO


def silero_vad(wav: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame speech probabilities from Silero (torch.hub)."""
    raise NotImplementedError  # TODO
