"""Reference VAD baselines.

Baselines
---------
1. Energy:
   simple log-RMS frame energy. The threshold MUST be selected on validation
   data, never on test data.

2. Silero VAD:
   pretrained neural VAD using the official silero-vad Python package.
   We use its default operating point/post-processing.

3. WebRTC VAD:
   optional classical baseline using 10-ms PCM frames.

None of these baselines is used to train the submitted CRNN.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import torch


# =====================================================================
# Energy baseline
# =====================================================================

def energy_scores(
    wav: np.ndarray,
    sr: int,
    n_frames: Optional[int] = None,
    hop_ms: float = 10.0,
    win_ms: float = 25.0,
) -> np.ndarray:
    """Return one log-RMS energy score per VAD frame.

    Frames are centred on the same 10-ms label grid used by dataset.py:
        centre_i = (i + 0.5) * hop

    The scores are continuous, so a global operating threshold can be
    selected using validation data.
    """
    x = np.asarray(wav, dtype=np.float32).reshape(-1)

    if sr <= 0:
        raise ValueError(f"Invalid sample rate: {sr}")

    hop = int(round(sr * hop_ms / 1000.0))
    win = int(round(sr * win_ms / 1000.0))

    if hop <= 0 or win <= 0:
        raise ValueError(
            f"Invalid framing: hop={hop}, win={win}"
        )

    if n_frames is None:
        n_frames = int(round(len(x) / hop))

    if n_frames <= 0:
        return np.empty(0, dtype=np.float32)

    # Half-window padding allows centred energy computation at boundaries.
    half = win // 2

    if len(x) > 1:
        padded = np.pad(
            x,
            (half + hop, half + hop),
            mode="reflect",
        )
    else:
        padded = np.pad(
            x,
            (half + hop, half + hop),
            mode="constant",
        )

    sq = padded.astype(np.float64) ** 2

    cumsum = np.concatenate(
        [
            np.zeros(1, dtype=np.float64),
            np.cumsum(sq),
        ]
    )

    centres = (
        (np.arange(n_frames, dtype=np.float64) + 0.5)
        * hop
    )

    centres = np.round(centres).astype(np.int64)

    # padded index corresponding to original sample 0
    offset = half + hop

    starts = (
        centres
        + offset
        - half
    )

    starts = np.clip(
        starts,
        0,
        len(padded) - win,
    )

    ends = starts + win

    sums = (
        cumsum[ends]
        - cumsum[starts]
    )

    rms = np.sqrt(
        sums / max(win, 1)
        + 1e-12
    )

    # Natural log is sufficient: only score ordering matters.
    return np.log(
        rms + 1e-8
    ).astype(np.float32)


def energy_vad(
    wav: np.ndarray,
    sr: int,
    threshold: float,
    n_frames: Optional[int] = None,
    hop_ms: float = 10.0,
    win_ms: float = 25.0,
) -> np.ndarray:
    """Binary energy VAD using a threshold selected on validation."""
    scores = energy_scores(
        wav,
        sr,
        n_frames=n_frames,
        hop_ms=hop_ms,
        win_ms=win_ms,
    )

    return (
        scores >= threshold
    ).astype(np.uint8)


# =====================================================================
# Silero VAD
# =====================================================================

@lru_cache(maxsize=1)
def load_silero():
    """Load the Silero VAD model once per evaluation process."""
    try:
        from silero_vad import load_silero_vad
    except ImportError as exc:
        raise RuntimeError(
            "silero-vad is not installed. "
            "Install it with: pip install silero-vad"
        ) from exc

    model = load_silero_vad()
    return model


def silero_vad(
    wav: np.ndarray,
    sr: int,
    n_frames: Optional[int] = None,
    hop_s: float = 0.010,
    model=None,
) -> np.ndarray:
    """Run official Silero VAD and rasterise its speech timestamps.

    Uses Silero's default threshold and default segment post-processing.
    """
    if sr not in (8000, 16000):
        raise ValueError(
            f"Silero baseline expects 8 or 16 kHz; received {sr}."
        )

    try:
        from silero_vad import get_speech_timestamps
    except ImportError as exc:
        raise RuntimeError(
            "silero-vad is not installed."
        ) from exc

    if model is None:
        model = load_silero()

    x = torch.as_tensor(
        np.asarray(wav, dtype=np.float32)
    ).reshape(-1)

    timestamps = get_speech_timestamps(
        x,
        model,
        sampling_rate=sr,
        return_seconds=False,
    )

    if n_frames is None:
        n_frames = int(
            round(
                len(x)
                / (sr * hop_s)
            )
        )

    mask = np.zeros(
        n_frames,
        dtype=np.uint8,
    )

    centres = (
        np.arange(n_frames, dtype=np.float64)
        + 0.5
    ) * hop_s

    for segment in timestamps:
        start_s = (
            float(segment["start"])
            / sr
        )

        end_s = (
            float(segment["end"])
            / sr
        )

        lo = np.searchsorted(
            centres,
            start_s,
            side="left",
        )

        hi = np.searchsorted(
            centres,
            end_s,
            side="right",
        )

        mask[lo:hi] = 1

    return mask


# =====================================================================
# WebRTC VAD
# =====================================================================

def webrtc_vad(
    wav: np.ndarray,
    sr: int,
    mode: int = 2,
    frame_ms: int = 10,
    n_frames: Optional[int] = None,
) -> np.ndarray:
    """Run WebRTC VAD on 10-ms PCM frames.

    Parameters
    ----------
    mode:
        Aggressiveness in {0, 1, 2, 3}.
    """
    try:
        import webrtcvad
    except ImportError as exc:
        raise RuntimeError(
            "webrtcvad is not installed."
        ) from exc

    if sr not in (
        8000,
        16000,
        32000,
        48000,
    ):
        raise ValueError(
            f"Unsupported WebRTC sample rate: {sr}"
        )

    if frame_ms not in (
        10,
        20,
        30,
    ):
        raise ValueError(
            "WebRTC frame_ms must be 10, 20, or 30."
        )

    vad = webrtcvad.Vad(
        int(mode)
    )

    x = np.asarray(
        wav,
        dtype=np.float32,
    ).reshape(-1)

    x = np.clip(
        x,
        -1.0,
        1.0,
    )

    pcm = (
        x * 32767.0
    ).astype(
        np.int16
    )

    frame_samples = int(
        sr * frame_ms / 1000
    )

    if n_frames is None:
        n_frames = int(
            round(
                len(x)
                / frame_samples
            )
        )

    out = np.zeros(
        n_frames,
        dtype=np.uint8,
    )

    for i in range(n_frames):
        start = (
            i * frame_samples
        )

        end = (
            start
            + frame_samples
        )

        frame = pcm[
            start:end
        ]

        if len(frame) < frame_samples:
            frame = np.pad(
                frame,
                (
                    0,
                    frame_samples
                    - len(frame),
                ),
            )

        out[i] = int(
            vad.is_speech(
                frame.tobytes(),
                sr,
            )
        )

    return out
