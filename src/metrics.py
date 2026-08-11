"""Frame-level and segment/time-level VAD metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


EPS = 1e-12


def _as_1d(x, dtype=None) -> np.ndarray:
    return np.asarray(x, dtype=dtype).reshape(-1)


def frame_metrics(
    probs: np.ndarray,
    target: np.ndarray,
    threshold: float,
):
    """Compute frame-level binary VAD metrics."""
    probs = _as_1d(probs, np.float64)
    target = _as_1d(target, np.int64)

    if probs.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: probs={probs.shape}, target={target.shape}"
        )

    pred = (probs >= threshold).astype(np.int64)

    precision, recall, f1, _ = precision_recall_fscore_support(
        target,
        pred,
        average="binary",
        zero_division=0,
    )

    accuracy = accuracy_score(target, pred)

    speech = target == 1
    nonspeech = target == 0

    miss = np.logical_and(speech, pred == 0).sum()
    false_alarm = np.logical_and(nonspeech, pred == 1).sum()

    miss_rate = float(miss / max(int(speech.sum()), 1))
    false_alarm_rate = float(
        false_alarm / max(int(nonspeech.sum()), 1)
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "miss_rate": miss_rate,
        "false_alarm_rate": false_alarm_rate,
    }


def roc_auc(
    probs: np.ndarray,
    target: np.ndarray,
):
    """Return ROC and precision-recall curves plus AUROC/AUPRC."""
    probs = _as_1d(probs, np.float64)
    target = _as_1d(target, np.int64)

    if probs.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: probs={probs.shape}, target={target.shape}"
        )

    if len(np.unique(target)) < 2:
        empty = np.asarray([], dtype=np.float64)

        return (
            empty,
            empty,
            float("nan"),
        ), (
            empty,
            empty,
            float("nan"),
        )

    fpr, tpr, _ = roc_curve(target, probs)
    auroc = roc_auc_score(target, probs)

    precision, recall, _ = precision_recall_curve(
        target,
        probs,
    )
    auprc = average_precision_score(target, probs)

    return (
        fpr,
        tpr,
        float(auroc),
    ), (
        precision,
        recall,
        float(auprc),
    )


def det_points(
    probs: np.ndarray,
    target: np.ndarray,
):
    """Return false-alarm and miss-rate arrays for a DET curve."""
    probs = _as_1d(probs, np.float64)
    target = _as_1d(target, np.int64)

    if len(np.unique(target)) < 2:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty

    fpr, tpr, _ = roc_curve(target, probs)

    false_alarm_rate = fpr
    miss_rate = 1.0 - tpr

    return false_alarm_rate, miss_rate


def select_threshold(
    val_probs,
    val_target,
    criterion: str = "max_f1",
) -> float:
    """Choose the operating threshold using validation data only."""
    probs = _as_1d(val_probs, np.float64)
    target = _as_1d(val_target, np.int64)

    if probs.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: probs={probs.shape}, target={target.shape}"
        )

    if criterion != "max_f1":
        raise ValueError(
            f"Unsupported threshold criterion: {criterion}. "
            "Supported criterion: max_f1"
        )

    if len(probs) == 0:
        raise ValueError("Cannot select a threshold from an empty validation set.")

    if len(np.unique(target)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(
        target,
        probs,
    )

    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]

    f1 = (
        2.0
        * precision
        * recall
        / np.maximum(precision + recall, EPS)
    )

    best_idx = int(np.nanargmax(f1))
    threshold = float(thresholds[best_idx])

    return float(np.clip(threshold, 0.0, 1.0))


def _merge_intervals(intervals):
    intervals = sorted(
        (float(s), float(e))
        for s, e in intervals
        if float(e) > float(s)
    )

    if not intervals:
        return []

    merged = [[intervals[0][0], intervals[0][1]]]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(
                merged[-1][1],
                end,
            )
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def _contains(intervals, t: float) -> bool:
    for start, end in intervals:
        if start <= t < end:
            return True
    return False


def segment_detection_error_rate(
    pred_segments,
    ref_segments,
    collar_s: float,
):
    """Time-weighted binary speech detection error rate.

    DER = (missed speech time + false alarm time) / reference speech time.

    A collar around reference speech boundaries is excluded from scoring.
    """
    pred = _merge_intervals(pred_segments)
    ref = _merge_intervals(ref_segments)

    if not ref:
        return float("nan")

    max_time = max(
        [end for _, end in ref]
        + [end for _, end in pred]
        + [0.0]
    )

    exclusions = []

    if collar_s > 0:
        for start, end in ref:
            exclusions.append(
                (
                    max(0.0, start - collar_s),
                    min(max_time, start + collar_s),
                )
            )
            exclusions.append(
                (
                    max(0.0, end - collar_s),
                    min(max_time, end + collar_s),
                )
            )

        exclusions = _merge_intervals(exclusions)

    boundaries = {0.0, max_time}

    for intervals in (pred, ref, exclusions):
        for start, end in intervals:
            boundaries.add(start)
            boundaries.add(end)

    boundaries = sorted(boundaries)

    miss_duration = 0.0
    false_alarm_duration = 0.0
    reference_duration = 0.0

    for left, right in zip(
        boundaries[:-1],
        boundaries[1:],
    ):
        if right <= left:
            continue

        midpoint = 0.5 * (left + right)

        if _contains(exclusions, midpoint):
            continue

        duration = right - left

        is_ref = _contains(ref, midpoint)
        is_pred = _contains(pred, midpoint)

        if is_ref:
            reference_duration += duration

        if is_ref and not is_pred:
            miss_duration += duration

        elif is_pred and not is_ref:
            false_alarm_duration += duration

    if reference_duration <= EPS:
        return float("nan")

    return float(
        (miss_duration + false_alarm_duration)
        / reference_duration
    )
