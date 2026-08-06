"""Plots for the report — every figure gets title, axis labels, legend."""
from __future__ import annotations
import numpy as np


def plot_mask_overlay(wav, sr, ref_mask, pred_mask, probs, out_path):
    """Waveform + spectrogram + ground-truth vs predicted mask + prob trace."""
    raise NotImplementedError  # TODO


def plot_curves(history, out_path):
    """Train/val loss and val metric over steps."""
    raise NotImplementedError  # TODO


def plot_f1_vs_snr(results, out_path):
    """Headline robustness plot."""
    raise NotImplementedError  # TODO


def plot_det(det_by_system, out_path):
    """DET curve for our model vs baselines, with the chosen operating point."""
    raise NotImplementedError  # TODO
