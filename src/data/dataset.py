"""Provided-corpus loading and frame-level speech-mask labelling.

Provided format: <stem>.wav (16 kHz mono) + <stem>.json with
    {"speech_segments": [{"start_time": s, "end_time": e}, ...]}  (seconds)

Labelling convention:
  - speech mask = union of speech_segments, with word-level micro-gaps below
    `merge_gap_s` merged into coherent speech regions;
  - rasterised at a fixed frame hop (default 10 ms); frame i is labelled speech
    if its centre time (i+0.5)*hop falls inside a (merged) speech segment.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
import numpy as np
import soundfile as sf

Segment = Tuple[float, float]


# ---------------------------------------------------------------- helpers
def list_utterances(root: str) -> List[str]:
    """Sorted stems that have BOTH a .wav and a .json."""
    root = Path(root)
    return sorted(p.stem for p in root.glob("*.wav")
                  if (root / f"{p.stem}.json").exists())


def speaker_of(stem: str) -> str:
    """LibriSpeech naming: speaker-chapter-utterance -> speaker id."""
    return stem.split("-")[0]


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a mono wav as float32 in [-1, 1]."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:                       # safety: downmix if ever stereo
        wav = wav.mean(axis=1)
    return wav, sr


# ------------------------------------------------------------ segments/mask
def load_segments(json_path: str, merge_gap_s: float = 0.0) -> List[Segment]:
    """Load speech_segments, sort, and merge overlaps + gaps <= merge_gap_s."""
    raw = json.load(open(json_path))["speech_segments"]
    segs = sorted((float(s["start_time"]), float(s["end_time"])) for s in raw)
    merged: List[Segment] = []
    for a, b in segs:
        if merged and a <= merged[-1][1] + merge_gap_s:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def n_frames_for(duration_s: float, hop_s: float) -> int:
    """Number of frames covering a signal of the given duration."""
    return int(round(duration_s / hop_s))


def segments_to_frame_mask(segments: List[Segment], n_frames: int,
                           hop_s: float) -> np.ndarray:
    """Rasterise merged segments to a {0,1} float32 mask of length n_frames.

    Frame i is speech iff its centre (i+0.5)*hop lies within a segment.
    """
    mask = np.zeros(n_frames, dtype=np.float32)
    if not segments or n_frames == 0:
        return mask
    centres = (np.arange(n_frames) + 0.5) * hop_s
    for a, b in segments:
        lo = np.searchsorted(centres, a, side="left")
        hi = np.searchsorted(centres, b, side="right")
        mask[lo:hi] = 1.0
    return mask


def build_frame_mask(json_path: str, duration_s: float, hop_s: float,
                     merge_gap_s: float) -> np.ndarray:
    """Convenience: json + duration -> frame mask."""
    segs = load_segments(json_path, merge_gap_s)
    return segments_to_frame_mask(segs, n_frames_for(duration_s, hop_s), hop_s)


# ------------------------------------------------------------------ dataset
class VADDataset:
    """Yields one labelled utterance.

    Returns (x, mask) where:
      - if `feature_fn` is given: x = features [T, n_mels], mask [T]
      - else:                     x = waveform [n_samples], mask [n_frames]
    `augment_fn(wav, sr) -> wav` (optional) degrades the *audio only*; the mask
    is unchanged because it is derived from the clean signal's segments.
    Feature extraction and augmentation are wired in later steps (features.py,
    mixing.py); this class already exposes the hooks.
    """
    def __init__(self, stems: List[str], root: str, hop_s: float = 0.010,
                 merge_gap_s: float = 0.050, feature_fn=None, augment_fn=None):
        self.stems = list(stems)
        self.root = Path(root)
        self.hop_s = hop_s
        self.merge_gap_s = merge_gap_s
        self.feature_fn = feature_fn
        self.augment_fn = augment_fn

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, i: int):
        stem = self.stems[i]
        wav, sr = read_wav(self.root / f"{stem}.wav")
        mask = build_frame_mask(self.root / f"{stem}.json",
                                len(wav) / sr, self.hop_s, self.merge_gap_s)
        if self.augment_fn is not None:                 # audio only
            wav = self.augment_fn(wav, sr)
        if self.feature_fn is not None:
            feats = self.feature_fn(wav, sr)            # [T, n_mels]
            T = min(len(feats), len(mask))              # align off-by-one
            return feats[:T], mask[:T]
        return wav, mask