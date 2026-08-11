"""Post-processing for frame-level VAD predictions.

All functions are pure operations on one utterance at a time.
"""
from __future__ import annotations

import numpy as np


def _runs(mask: np.ndarray):
    """Return binary runs as (value, start, end), with end exclusive."""
    x = np.asarray(
        mask,
        dtype=np.uint8,
    ).reshape(-1)

    if len(x) == 0:
        return []

    changes = (
        np.flatnonzero(
            np.diff(x) != 0
        )
        + 1
    )

    starts = np.r_[
        0,
        changes,
    ]

    ends = np.r_[
        changes,
        len(x),
    ]

    return [
        (
            int(x[start]),
            int(start),
            int(end),
        )
        for start, end
        in zip(starts, ends)
    ]


def median_smooth(
    probs: np.ndarray,
    win_frames: int,
) -> np.ndarray:
    """Median-filter a probability sequence."""
    x = np.asarray(
        probs,
        dtype=np.float32,
    ).reshape(-1)

    if len(x) == 0:
        return x.copy()

    if win_frames <= 1:
        return x.copy()

    # Use an odd median window.
    if win_frames % 2 == 0:
        win_frames += 1

    pad = (
        win_frames // 2
    )

    padded = np.pad(
        x,
        (pad, pad),
        mode="edge",
    )

    windows = (
        np.lib.stride_tricks
        .sliding_window_view(
            padded,
            win_frames,
        )
    )

    return np.median(
        windows,
        axis=-1,
    ).astype(
        np.float32
    )


def apply_min_speech(
    mask: np.ndarray,
    min_frames: int,
) -> np.ndarray:
    """Remove speech runs shorter than min_frames."""
    out = np.asarray(
        mask,
        dtype=np.uint8,
    ).reshape(-1).copy()

    if min_frames <= 1:
        return out

    for value, start, end in _runs(out):
        if (
            value == 1
            and end - start < min_frames
        ):
            out[start:end] = 0

    return out


def merge_short_silence(
    mask: np.ndarray,
    min_gap_frames: int,
) -> np.ndarray:
    """Fill internal silence gaps shorter than min_gap_frames."""
    out = np.asarray(
        mask,
        dtype=np.uint8,
    ).reshape(-1).copy()

    if min_gap_frames <= 0:
        return out

    runs = _runs(out)

    for index, (
        value,
        start,
        end,
    ) in enumerate(runs):

        if value != 0:
            continue

        left_is_speech = (
            index > 0
            and runs[index - 1][0] == 1
        )

        right_is_speech = (
            index + 1 < len(runs)
            and runs[index + 1][0] == 1
        )

        if (
            left_is_speech
            and right_is_speech
            and end - start < min_gap_frames
        ):
            out[start:end] = 1

    return out


def apply_hangover(
    mask: np.ndarray,
    hangover_frames: int,
) -> np.ndarray:
    """Extend the end of each original speech run."""
    out = np.asarray(
        mask,
        dtype=np.uint8,
    ).reshape(-1).copy()

    if hangover_frames <= 0:
        return out

    speech_runs = [
        (start, end)
        for value, start, end in _runs(out)
        if value == 1
    ]

    n = len(out)

    for _, end in speech_runs:
        out[
            end:min(
                n,
                end + hangover_frames,
            )
        ] = 1

    return out


def mask_to_segments(
    mask: np.ndarray,
    hop_s: float,
):
    """Convert a binary frame sequence to [(start_s, end_s), ...]."""
    segments = []

    for value, start, end in _runs(mask):
        if value == 1:
            segments.append(
                (
                    float(
                        start * hop_s
                    ),
                    float(
                        end * hop_s
                    ),
                )
            )

    return segments


def full_postprocess(
    probs: np.ndarray,
    threshold: float,
    smooth_frames: int,
    min_speech_frames: int,
    min_silence_frames: int,
    hangover_frames: int,
) -> np.ndarray:
    """Apply the complete configured VAD post-processing pipeline."""
    smoothed = median_smooth(
        probs,
        smooth_frames,
    )

    mask = (
        smoothed >= threshold
    ).astype(
        np.uint8
    )

    mask = apply_min_speech(
        mask,
        min_speech_frames,
    )

    mask = merge_short_silence(
        mask,
        min_silence_frames,
    )

    mask = apply_hangover(
        mask,
        hangover_frames,
    )

    return mask
