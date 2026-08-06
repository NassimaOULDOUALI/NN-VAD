"""Noise + reverberation augmentation (PyTorch). The label is NEVER touched.

Pipeline (wav domain), applied inside VADDataset via the `augment_fn` hook:
    clean speech --(reverb, p_reverb)--> --(+ MUSAN noise @ SNR, p_noise)--> mixed
The frame mask is derived from the CLEAN signal's segments, so it stays exact:
we only degrade the audio. `add_reverb` peak-aligns the RIR so onsets do not
shift (which would otherwise invalidate the mask).

SpecAugment operates in the FEATURE domain (not wav), so it is provided here but
applied after feature extraction (in the training step), not in augment_fn.

External sources (public only):
  - NoiseBank: MUSAN music/ and noise/  (NEVER speech/, which is speech).
  - RIRBank:   RIRS_NOISES impulse responses.
Both reserve a disjoint pool for test (held-out files) to avoid noise leakage.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import random
import torch
import torch.nn.functional as F
import soundfile as sf


# ----------------------------------------------------------------- core ops
def _rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(x ** 2) + eps)


def _fit_length(noise: torch.Tensor, n: int, rng: random.Random) -> torch.Tensor:
    """Tile/crop a noise clip to exactly n samples (random offset)."""
    if noise.numel() < n:
        reps = n // noise.numel() + 1
        noise = noise.repeat(reps)
    start = rng.randint(0, noise.numel() - n)
    return noise[start:start + n]


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
    rir = rir / (rir.abs().max() + 1e-8)          # normalise
    peak = int(torch.argmax(rir.abs()))           # direct-path index
    L, K = speech.numel(), rir.numel()
    # full convolution via conv1d (flip kernel -> true convolution)
    full = F.conv1d(speech.view(1, 1, -1),
                    rir.flip(0).view(1, 1, -1),
                    padding=K - 1).view(-1)        # length L + K - 1
    out = full[peak:peak + L]                      # align on direct path
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
        if hi <= 0:
            return 0
        return int(torch.randint(0, hi, (1,), generator=generator))

    for _ in range(n_time):
        w = _randint(int(T * max_time_frac))
        if w > 0:
            t0 = _randint(T - w); out[t0:t0 + w, :] = fill
    for _ in range(n_freq):
        w = _randint(int(M * max_freq_frac))
        if w > 0:
            f0 = _randint(M - w); out[:, f0:f0 + w] = fill
    return out


# ------------------------------------------------------------ external banks
def _load_mono(path: str) -> torch.Tensor:
    wav, _ = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return torch.from_numpy(wav)


class NoiseBank:
    """Indexes MUSAN music/ and noise/ (NEVER speech/); reserves a test pool."""
    def __init__(self, musan_root: str, split: str = "train",
                 test_frac: float = 0.15, seed: int = 42):
        root = Path(musan_root)
        files: List[Path] = []
        for cat in ("music", "noise"):            # deliberately NOT "speech"
            files += sorted((root / cat).rglob("*.wav"))
        rng = random.Random(seed); rng.shuffle(files)
        cut = int(len(files) * (1 - test_frac))
        self.files = files[:cut] if split == "train" else files[cut:]
        if not self.files:
            raise RuntimeError(f"no MUSAN wavs under {musan_root} "
                               f"(expected music/ and noise/)")

    def sample(self, n: int, rng: random.Random) -> torch.Tensor:
        f = self.files[rng.randrange(len(self.files))]
        return _fit_length(_load_mono(f), n, rng)


class RIRBank:
    """Indexes RIR wavs; reserves a test pool (held-out rooms)."""
    def __init__(self, rirs_root: str, split: str = "train",
                 test_frac: float = 0.15, seed: int = 43):
        files = sorted(Path(rirs_root).rglob("*.wav"))
        rng = random.Random(seed); rng.shuffle(files)
        cut = int(len(files) * (1 - test_frac))
        self.files = files[:cut] if split == "train" else files[cut:]
        if not self.files:
            raise RuntimeError(f"no RIR wavs under {rirs_root}")

    def sample(self, rng: random.Random) -> torch.Tensor:
        return _load_mono(self.files[rng.randrange(len(self.files))])


# -------------------------------------------------------------- augment hook
def make_augment_fn(cfg_aug: dict, noise_bank: Optional[NoiseBank],
                    rir_bank: Optional[RIRBank], seed: int = 42):
    """Build augment_fn(wav, sr) -> degraded wav (reverb then noise)."""
    snr_lo, snr_hi = cfg_aug.get("snr_db_range", [-5, 20])
    p_noise = cfg_aug.get("p_noise", 0.8)
    p_reverb = cfg_aug.get("p_reverb", 0.5)       # set 1.0 for reverb-on-all
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