"""VAD post-processing: smoothing, min-duration, silence-merge, hangover.

Each stage is a pure function over a per-frame sequence so it can be toggled
independently for the ablation (report Table: post-processing ablation).

Latency (product-relevant — quantify this in the memo)
------------------------------------------------------
* Centred median: needs (win-1)/2 future frames  -> look-ahead = win/2 (ms).
  Use ``median_smooth(..., causal=True)`` for a trailing window with ZERO
  look-ahead (weaker smoothing) when latency is the binding constraint.
* Silence-merge / min-speech-duration: as offline array ops they add no
  look-ahead, but *online* they cost confirmation latency equal to their window
  (you must wait min_silence to be sure a gap isn't a real offset, and
  min_speech to confirm an onset). Report this, don't hide it.
* Hangover: only *extends* past a detected offset -> ZERO look-ahead. It is the
  cheapest way to stop clipping word tails, which is why it is standard.

``postprocess_pipeline`` composes the stages in the conventional order
(smooth -> threshold -> close short silences -> open short speech -> hangover)
and ``lookahead_ms`` reports the look-ahead that configuration implies.
"""
from __future__ import annotations

import numpy as np

Segment = tuple[float, float]


# ----------------------------------------------------------------- run helpers
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True as ``[start, end)`` (end exclusive)."""
    m = np.asarray(mask).astype(bool)
    if m.size == 0:
        return []
    d = np.diff(m.astype(np.int8))
    starts = (np.flatnonzero(d == 1) + 1).tolist()
    ends = (np.flatnonzero(d == -1) + 1).tolist()
    if m[0]:
        starts = [0] + starts
    if m[-1]:
        ends = ends + [m.size]
    return list(zip(starts, ends))


# ---------------------------------------------------------------------- stages
def median_smooth(probs: np.ndarray, win_frames: int, causal: bool = False) -> np.ndarray:
    """Median filter on the *probability* sequence.

    ``causal=True`` uses a trailing window (zero look-ahead); otherwise a centred
    odd-length window (look-ahead = (win-1)/2 frames).
    """
    p = np.asarray(probs, dtype=float).ravel()
    if win_frames <= 1:
        return p.copy()
    if causal:
        out = np.empty_like(p)
        for i in range(p.size):
            out[i] = np.median(p[max(0, i - win_frames + 1):i + 1])
        return out
    from scipy.signal import medfilt
    k = win_frames + 1 if win_frames % 2 == 0 else win_frames  # medfilt needs odd
    return medfilt(p, kernel_size=k)


def apply_min_speech(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Drop speech runs shorter than ``min_frames`` (kills brief false positives)."""
    out = np.asarray(mask).astype(bool).copy()
    if min_frames <= 1:
        return out
    for s, e in _runs(out):
        if e - s < min_frames:
            out[s:e] = False
    return out


def merge_short_silence(mask: np.ndarray, min_gap_frames: int) -> np.ndarray:
    """Merge speech runs separated by fewer than ``min_gap_frames`` (fills micro-gaps)."""
    out = np.asarray(mask).astype(bool).copy()
    if min_gap_frames <= 0:
        return out
    runs = _runs(out)
    for (s0, e0), (s1, e1) in zip(runs, runs[1:]):
        if s1 - e0 < min_gap_frames:
            out[e0:s1] = True
    return out


def apply_hangover(mask: np.ndarray, hangover_frames: int) -> np.ndarray:
    """Extend each speech run end by ``hangover_frames`` (avoids clipping word tails)."""
    out = np.asarray(mask).astype(bool).copy()
    if hangover_frames <= 0:
        return out
    n = out.size
    for _, e in _runs(out):
        out[e:min(n, e + hangover_frames)] = True
    return out


def mask_to_segments(mask: np.ndarray, hop_s: float) -> list[Segment]:
    """``{0,1}`` frame mask -> list of ``(start_s, end_s)`` speech segments."""
    return [(s * hop_s, e * hop_s) for s, e in _runs(mask)]


# -------------------------------------------------------------------- pipeline
def postprocess_pipeline(probs: np.ndarray, threshold: float, hop_s: float, *,
                         smooth_median_ms: float = 0.0, min_speech_ms: float = 0.0,
                         min_silence_ms: float = 0.0, hangover_ms: float = 0.0,
                         causal_median: bool = False, return_mask: bool = False):
    """Compose the stages in the conventional order and return segments.

    Any stage with a 0 duration is skipped — that is exactly how the ablation
    turns each brick on/off. Order: smooth -> threshold -> close silences ->
    open speech -> hangover.
    """
    hop_ms = hop_s * 1000.0
    p = np.asarray(probs, dtype=float).ravel()

    win = int(round(smooth_median_ms / hop_ms)) if smooth_median_ms else 0
    if win > 1:
        p = median_smooth(p, win, causal=causal_median)

    mask = p >= threshold
    if min_silence_ms:
        mask = merge_short_silence(mask, int(round(min_silence_ms / hop_ms)))
    if min_speech_ms:
        mask = apply_min_speech(mask, int(round(min_speech_ms / hop_ms)))
    if hangover_ms:
        mask = apply_hangover(mask, int(round(hangover_ms / hop_ms)))

    segs = mask_to_segments(mask, hop_s)
    return (segs, mask) if return_mask else segs


def lookahead_ms(smooth_median_ms: float = 0.0, causal_median: bool = False,
                 **_ignored) -> float:
    """Output-time look-ahead (ms) implied by the post-processing config.

    Only the centred median needs *future* frames to produce its current output.
    Min-speech / min-silence add *confirmation* latency online (documented in the
    memo) but not output look-ahead when applied as offline array ops; hangover
    adds none.
    """
    return (smooth_median_ms / 2.0) if (smooth_median_ms and not causal_median) else 0.0