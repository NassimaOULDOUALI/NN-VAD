"""Feature front-end tests (torch): frame alignment + CMVN sanity."""
import numpy as np
import torch
from src.data.dataset import (list_utterances, read_wav, n_frames_for,
                              VADDataset)
from src.data.features import logmel, causal_cmvn, make_feature_fn

HOP_S, MERGE = 0.010, 0.050
FEATCFG = {"n_mels": 64, "win_ms": 25, "hop_ms": 10, "cmvn": "causal"}


def test_frame_alignment(root):
    """|T_features - T_mask| <= 1 for every utterance (no systematic drift)."""
    stems = list_utterances(root)
    max_diff = 0
    for stem in stems:
        wav, sr = read_wav(f"{root}/{stem}.wav")
        T_feat = logmel(wav, sr, n_mels=64, win_ms=25, hop_ms=10).shape[0]
        T_mask = n_frames_for(len(wav) / sr, HOP_S)
        max_diff = max(max_diff, abs(T_feat - T_mask))
    assert max_diff <= 1, f"frame drift too large: {max_diff}"


def test_cmvn_shape_and_finite(root):
    stem = list_utterances(root)[0]
    wav, sr = read_wav(f"{root}/{stem}.wav")
    f = logmel(wav, sr, 64, 25, 10)
    z = causal_cmvn(f)
    assert z.shape == f.shape
    assert torch.isfinite(z).all()


def test_dataset_feature_mask_equal_length(root):
    """VADDataset + feature_fn returns feats and mask of equal length."""
    stems = list_utterances(root)[:20]
    ds = VADDataset(stems, root, hop_s=HOP_S, merge_gap_s=MERGE,
                    feature_fn=make_feature_fn(FEATCFG))
    for i in range(len(ds)):
        feats, mask = ds[i]
        assert feats.shape[0] == mask.shape[0]
        assert feats.shape[1] == FEATCFG["n_mels"]
