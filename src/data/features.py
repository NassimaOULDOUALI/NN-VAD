"""Log-mel front-end (PyTorch / torchaudio) with causal normalisation.

Framing (center=True) matches the standard convention:
    n_frames = 1 + n_samples // hop
so it aligns with the label grid `n_frames_for(dur, hop) = round(dur/hop)`
to within one frame (handled by VADDataset via T = min(len(feats), len(mask))).

Causality note: center=True pads win/2 (=12.5 ms) symmetrically, i.e. a small
look-ahead. For a strictly causal front-end pass center=False (left context
only), at a <=1-frame alignment cost. Report the chosen look-ahead accordingly.

Returns torch.FloatTensor [T, n_mels]. The mask stays a numpy array in
VADDataset; the training collate converts masks to tensors when batching.
"""
from __future__ import annotations
from functools import lru_cache
import torch
import torchaudio.transforms as T


@lru_cache(maxsize=8)
def _mel_transform(sr: int, n_fft: int, hop: int, n_mels: int,
                   f_min: float, center: bool):
    """Cached MelSpectrogram transform (built once per parameter set)."""
    return T.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, win_length=n_fft, hop_length=hop,
        n_mels=n_mels, f_min=f_min, power=2.0, center=center, pad_mode="reflect",
    )


def _to_tensor(wav) -> torch.Tensor:
    x = wav if isinstance(wav, torch.Tensor) else torch.as_tensor(wav)
    x = x.float()
    return x.mean(0) if x.dim() > 1 else x     # downmix if ever multichannel


def logmel(wav, sr: int, n_mels: int = 64, win_ms: float = 25.0,
           hop_ms: float = 10.0, center: bool = True,
           f_min: float = 20.0) -> torch.Tensor:
    """Return a log-mel spectrogram [T, n_mels] (float32)."""
    n_fft = int(round(sr * win_ms / 1000.0))
    hop = int(round(sr * hop_ms / 1000.0))
    x = _to_tensor(wav)
    tf = _mel_transform(sr, n_fft, hop, n_mels, float(f_min), bool(center))
    mel = tf(x)                                # [n_mels, T]
    logmel = torch.log(mel + 1e-6)
    return logmel.transpose(0, 1).contiguous() # [T, n_mels]


def causal_cmvn(feats: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Running (causal) per-bin mean-variance normalisation over time.

    Frame t uses statistics from frames 0..t only (streaming-safe).
    """
    n = feats.shape[0]
    t = torch.arange(1, n + 1, device=feats.device,
                     dtype=feats.dtype).unsqueeze(1)
    mean = torch.cumsum(feats, dim=0) / t
    var = torch.cumsum(feats ** 2, dim=0) / t - mean ** 2
    return (feats - mean) / torch.sqrt(var.clamp_min(eps))


def make_feature_fn(cfg_features: dict):
    """Build feature_fn(wav, sr) -> [T, n_mels] tensor from the config block."""
    n_mels = cfg_features.get("n_mels", 64)
    win_ms = cfg_features.get("win_ms", 25)
    hop_ms = cfg_features.get("hop_ms", 10)
    center = cfg_features.get("center", True)
    use_cmvn = cfg_features.get("cmvn", "causal") == "causal"

    def feature_fn(wav, sr):
        f = logmel(wav, sr, n_mels=n_mels, win_ms=win_ms, hop_ms=hop_ms,
                   center=center)
        return causal_cmvn(f) if use_cmvn else f
    return feature_fn