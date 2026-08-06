"""Consistency tests for the data layer.

Key check: a reconstructed frame mask must reproduce the segment-based speech
ratio (label correctness), and speaker splits must be disjoint and exhaustive.
"""
import numpy as np
from src.data.dataset import (
    list_utterances, speaker_of, read_wav, load_segments,
    n_frames_for, segments_to_frame_mask, build_frame_mask,
)
from src.data.splits import speaker_disjoint_split

HOP = 0.010
MERGE = 0.050


def _seg_ratio(segments, dur):
    return sum(b - a for a, b in segments) / dur


def test_speaker_of():
    assert speaker_of("103-1240-0001") == "103"


def test_mask_matches_speech_ratio(root):
    """mask.mean() ~= segment speech ratio for every utterance."""
    stems = list_utterances(root)
    assert stems, "no data found"
    max_err = 0.0
    for stem in stems:
        wav, sr = read_wav(f"{root}/{stem}.wav")
        dur = len(wav) / sr
        segs = load_segments(f"{root}/{stem}.json", MERGE)
        mask = segments_to_frame_mask(segs, n_frames_for(dur, HOP), HOP)
        err = abs(mask.mean() - _seg_ratio(segs, dur))
        max_err = max(max_err, err)
    assert max_err < 0.02, f"mask/ratio mismatch, max_err={max_err:.4f}"


def test_splits_disjoint_and_exhaustive(root):
    stems = list_utterances(root)
    sp = speaker_disjoint_split(stems, (0.8, 0.1, 0.1), seed=42)
    tr = {speaker_of(s) for s in sp["train"]}
    va = {speaker_of(s) for s in sp["val"]}
    te = {speaker_of(s) for s in sp["test"]}
    assert tr & va == set() and tr & te == set() and va & te == set()
    all_stems = sorted(sp["train"] + sp["val"] + sp["test"])
    assert all_stems == sorted(stems)          # exhaustive, no dup
