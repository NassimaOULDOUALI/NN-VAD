"""Frame-level and segment-level VAD metrics."""
from __future__ import annotations
import numpy as np


def frame_metrics(probs: np.ndarray, target: np.ndarray, threshold: float):
    """Return dict: precision, recall, f1, accuracy at a given threshold."""
    raise NotImplementedError  # TODO


def roc_auc(probs: np.ndarray, target: np.ndarray):
    """Return (fpr, tpr, auroc) and (precision, recall, auprc)."""
    raise NotImplementedError  # TODO: sklearn


def det_points(probs: np.ndarray, target: np.ndarray):
    """Return (false_alarm_rate, miss_rate) arrays for the DET curve."""
    raise NotImplementedError  # TODO


def select_threshold(val_probs, val_target, criterion="max_f1") -> float:
    """Choose the operating threshold on validation (never on test)."""
    raise NotImplementedError  # TODO


def segment_detection_error_rate(pred_segments, ref_segments, collar_s: float):
    """Segment-level DER = (false alarm + missed detection) / speech duration.

    Prefer pyannote.metrics.detection.DetectionErrorRate.
    """
    raise NotImplementedError  # TODO
