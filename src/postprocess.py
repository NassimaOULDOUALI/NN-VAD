"""VAD post-processing: smoothing, min-duration, merge, hangover.

Each is a pure function on a per-frame decision/probability sequence so it can
be ablated independently (see report Table: post-processing ablation).
"""
from __future__ import annotations
import numpy as np


def median_smooth(probs: np.ndarray, win_frames: int) -> np.ndarray:
    raise NotImplementedError  # TODO


def apply_min_speech(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Drop speech runs shorter than min_frames."""
    raise NotImplementedError  # TODO


def merge_short_silence(mask: np.ndarray, min_gap_frames: int) -> np.ndarray:
    """Merge speech runs separated by fewer than min_gap_frames."""
    raise NotImplementedError  # TODO


def apply_hangover(mask: np.ndarray, hangover_frames: int) -> np.ndarray:
    """Extend each speech run end by hangover_frames."""
    raise NotImplementedError  # TODO


def mask_to_segments(mask: np.ndarray, hop_s: float):
    """Convert a {0,1} frame mask to a list of (start_s, end_s) segments."""
    raise NotImplementedError  # TODO
