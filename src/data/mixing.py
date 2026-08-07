"""Noise + reverberation augmentation (PyTorch). The label is NEVER touched.

Pipeline (wav domain), applied inside VADDataset via the `augment_fn` hook:
    clean speech --(reverb, p_reverb)--> --(+ MUSAN noise @ SNR, p_noise)--> mixed
The frame mask is derived from the CLEAN signal's segments, so it stays exact:
we only degrade the audio. `add_reverb` peak-aligns the RIR so onsets do not
shift (which would otherwise invalidate the mask).

Efficiency: noise clips are read by SEGMENT (soundfile header + windowed read)
instead of loading whole multi-minute MUSAN files; RIRs are small and LRU-cached.
This keeps disk I/O per sample tiny, which is the CPU-training bottleneck.

External sources (public only):
  - NoiseBank: MUSAN music/ and noise/  (NEVER speech/, which is speech).
  - RIRBank:   RIRS_NOISES impulse responses.
Both reserve a disjoint pool for test (held-out files) to avoid noise leakage.
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from typing import List, Optional, Dict
import random
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf


# ----------------------------------------------------------------- core ops
def _rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(x ** 2) + eps)


def add_noise(speech: torch.Tensor, noise: torch.Tensor,
              snr_db: float) -> torch.Tensor:
    """Mix noise into speech at a target SNR (dB). Same length as `speech`."""
    speech = speech.float()
    noise = noise.float()[:speech.numel()]
    scale = _rms(speech) / (_rms(noise) * (10.0 ** (snr_db / 20.0)))
    return speech + scale * noise


def add_reverb(speech: torch.Tensor, rir: torch.Tensor,
               preserve_energy: bool = True) -> torch.Tensor:
    """Convolve speech with a room impulse response, keeping time alignment.

    The RIR is peak-aligned (direct path) so the output onset matches the input
    onset -> the frame mask remains valid. Output length == input length.
    """
    speech = speech.float()
    rir = rir.float()
    rir = rir / (rir.abs().max() + 1e-8)
    peak = int(torch.argmax(rir.abs()))
    L, K = speech.numel(), rir.numel()
    full = F.conv1d(speech.view(1, 1, -1),
                    rir.flip(0).view(1, 1, -1),
                    padding=K - 1).view(-1)          # length L + K - 1
    out = full[peak:peak + L]                        # align on direct path
    if preserve_energy:
        out = out * (_rms(speech) / _rms(out))
    return out


def spec_augment(feats: torch.Tensor, n_time: int = 2, n_freq: int = 2,
                 max_time_frac: float = 0.10, max_freq_frac: float = 0.20,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Time/frequency masking on a feature matrix [T, n_mels]. Labels untouched."""
    T, M = feats.shape
    out = feats.clone()
    fill = feats.mean()

    def _randint(hi):
        return 0 if hi <= 0 else int(torch.randint(0, hi, (1,), generator=generator))

    for _ in range(n_time):
        w = _randint(int(T * max_time_frac))
        if w > 0:
            t0 = _randint(T - w); out[t0:t0 + w, :] = fill
    for _ in range(n_freq):
        w = _randint(int(M * max_freq_frac))
        if w > 0:
            f0 = _randint(M - w); out[:, f0:f0 + w] = fill
    return out


# ------------------------------------------------------- efficient IO helpers
def _downmix(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim > 1 else x


def _read_noise_segment(path: str, n: int, n_frames_total: int,
                        rng: random.Random) -> torch.Tensor:
    """Read exactly `n` samples from a noise file WITHOUT loading it whole.

    Long file -> read a random window [start, start+n]. Short file -> read all
    and tile. Only a small windowed decode happens per sample.
    """
    if n_frames_total <= n:
        x = _downmix(sf.read(path, dtype="float32", always_2d=False)[0])
        reps = n // max(len(x), 1) + 1
        x = np.tile(x, reps)[:n]
    else:
        start = rng.randint(0, n_frames_total - n)
        x = _downmix(sf.read(path, start=start, frames=n,
                             dtype="float32", always_2d=False)[0])
        if len(x) < n:                                # guard rounding
            x = np.pad(x, (0, n - len(x)))
    return torch.from_numpy(np.ascontiguousarray(x))


@lru_cache(maxsize=1024)
def _load_rir_cached(path: str) -> torch.Tensor:
    """Load a (small) RIR once; cached across calls."""
    x, _ = sf.read(path, dtype="float32", always_2d=False)
    return torch.from_numpy(np.ascontiguousarray(_downmix(x)))


# ------------------------------------------------------------ external banks
class NoiseBank:
    """Indexes MUSAN music/ and noise/ (NEVER speech/); reserves a test pool.

    Caches per-file frame counts (header-only, cheap) so sampling reads only the
    needed window.
    """
    def __init__(self, musan_root: str, split: str = "train",
                 test_frac: float = 0.15, seed: int = 42):
        root = Path(musan_root)
        files: List[Path] = []
        for cat in ("music", "noise"):            # deliberately NOT "speech"
            files += sorted((root / cat).rglob("*.wav"))
        rng = random.Random(seed); rng.shuffle(files)
        cut = int(len(files) * (1 - test_frac))
        self.files = [str(p) for p in (files[:cut] if split == "train"
                                       else files[cut:])]
        if not self.files:
            raise RuntimeError(f"no MUSAN wavs under {musan_root} "
                               f"(expected music/ and noise/)")
        self._nframes: Dict[str, int] = {}

    def _frames(self, path: str) -> int:
        n = self._nframes.get(path)
        if n is None:
            n = sf.info(path).frames                 # header only, no decode
            self._nframes[path] = n
        return n

    def sample(self, n: int, rng: random.Random) -> torch.Tensor:
        path = self.files[rng.randrange(len(self.files))]
        return _read_noise_segment(path, n, self._frames(path), rng)


class RIRBank:
    """Indexes RIR wavs; reserves a test pool (held-out rooms). LRU-cached."""
    def __init__(self, rirs_root: str, split: str = "train",
                 test_frac: float = 0.15, seed: int = 43):
        files = sorted(str(p) for p in Path(rirs_root).rglob("*.wav"))
        rng = random.Random(seed); rng.shuffle(files)
        cut = int(len(files) * (1 - test_frac))
        self.files = files[:cut] if split == "train" else files[cut:]
        if not self.files:
            raise RuntimeError(f"no RIR wavs under {rirs_root}")

    def sample(self, rng: random.Random) -> torch.Tensor:
        return _load_rir_cached(self.files[rng.randrange(len(self.files))])


# -------------------------------------------------------------- augment hook
def make_augment_fn(cfg_aug: dict, noise_bank: Optional[NoiseBank],
                    rir_bank: Optional[RIRBank], seed: int = 42):
    """Build augment_fn(wav, sr) -> degraded wav (reverb then noise)."""
    snr_lo, snr_hi = cfg_aug.get("snr_db_range", [-5, 20])
    p_noise = cfg_aug.get("p_noise", 0.8)
    p_reverb = cfg_aug.get("p_reverb", 0.5)          # set 1.0 for reverb-on-all
    rng = random.Random(seed)

    def augment_fn(wav, sr):
        x = wav if isinstance(wav, torch.Tensor) else torch.as_tensor(wav)
        x = x.float()
        if rir_bank is not None and rng.random() < p_reverb:
            x = add_reverb(x, rir_bank.sample(rng))
        if noise_bank is not None and rng.random() < p_noise:
            snr = rng.uniform(snr_lo, snr_hi)
            x = add_noise(x, noise_bank.sample(x.numel(), rng), snr)
        return x
    return augment_fn