"""Augmentation tests — self-contained (synthetic signals, no external data).

Checks: measured SNR matches target; reverb preserves length and time
alignment; SpecAugment preserves shape. The label is never an input to these
functions, so mask correctness reduces to 'audio in, audio out, same timing'.
"""
import math
import torch
from src.data.mixing import add_noise, add_reverb, spec_augment


def _measured_snr(clean, mixed):
    noise = mixed - clean
    return 20 * math.log10(clean.pow(2).mean().sqrt() /
                           noise.pow(2).mean().sqrt())


def test_add_noise_hits_target_snr():
    torch.manual_seed(0)
    speech = torch.randn(16000)                 # 1 s
    noise = torch.randn(16000)
    for target in (-5.0, 0.0, 10.0, 20.0):
        mixed = add_noise(speech, noise, target)
        assert abs(_measured_snr(speech, mixed) - target) < 0.5
        assert mixed.shape == speech.shape


def test_reverb_preserves_length_and_alignment():
    speech = torch.randn(16000)
    # delta RIR -> identity (after peak-align + energy preservation)
    rir = torch.zeros(200); rir[0] = 1.0
    out = add_reverb(speech, rir)
    assert out.shape == speech.shape
    assert torch.allclose(out, speech, atol=1e-4)
    # RIR with a leading delay: peak-align must remove the shift
    rir2 = torch.zeros(200); rir2[50] = 1.0
    out2 = add_reverb(speech, rir2)
    assert out2.shape == speech.shape
    # cross-correlation peak at lag 0 (no net delay introduced)
    lags = torch.arange(-5, 6)
    corr = [torch.dot(speech[5:-5], out2[5 + int(l):-5 + int(l) or None])
            for l in lags]
    assert int(lags[int(torch.tensor(corr).argmax())]) == 0


def test_specaugment_shape_and_finite():
    feats = torch.randn(300, 64)
    g = torch.Generator().manual_seed(0)
    out = spec_augment(feats, generator=g)
    assert out.shape == feats.shape
    assert torch.isfinite(out).all()
