"""Noise + reverberation augmentation for VAD training.

Pipeline:
    clean waveform
        -> optional reverb
        -> optional MUSAN noise at random SNR

The VAD reference labels are not modified.

MUSAN music/ and noise/ are used as interference.
MUSAN speech/ is deliberately excluded.

Train/test noise and RIR pools are file-disjoint and deterministic for
a fixed seed.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional
import random

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import get_worker_info


# ---------------------------------------------------------------------
# Basic signal operations
# ---------------------------------------------------------------------

def _rms(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return torch.sqrt(
        torch.mean(x ** 2) + eps
    )


def add_noise(
    speech: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """Mix noise into speech at the requested global RMS SNR."""
    speech = speech.float()
    noise = noise.float()

    if noise.numel() < speech.numel():
        repeats = (
            speech.numel() // max(noise.numel(), 1)
            + 1
        )
        noise = noise.repeat(repeats)

    noise = noise[:speech.numel()]

    speech_rms = _rms(speech)
    noise_rms = _rms(noise)

    scale = (
        speech_rms
        / (
            noise_rms
            * (10.0 ** (float(snr_db) / 20.0))
        )
    )

    return speech + scale * noise


def add_reverb(
    speech: torch.Tensor,
    rir: torch.Tensor,
    preserve_energy: bool = True,
) -> torch.Tensor:
    """Convolve with a RIR while aligning the direct-path peak."""
    speech = speech.float()
    rir = rir.float()

    if speech.numel() == 0:
        return speech

    if rir.numel() == 0:
        return speech

    rir = rir / (
        rir.abs().max() + 1e-8
    )

    peak = int(
        torch.argmax(rir.abs()).item()
    )

    signal_length = speech.numel()
    rir_length = rir.numel()

    full = F.conv1d(
        speech.view(1, 1, -1),
        rir.flip(0).view(1, 1, -1),
        padding=rir_length - 1,
    ).view(-1)

    out = full[
        peak:peak + signal_length
    ]

    if out.numel() < signal_length:
        out = F.pad(
            out,
            (0, signal_length - out.numel()),
        )

    if preserve_energy:
        out = out * (
            _rms(speech)
            / _rms(out)
        )

    return out


# ---------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------

def _torch_randint_inclusive(
    low: int,
    high: int,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Sample integer uniformly from [low, high]."""
    if high <= low:
        return int(low)

    return int(
        torch.randint(
            low,
            high + 1,
            (1,),
            generator=generator,
        ).item()
    )


def spec_augment(
    feats: torch.Tensor,
    n_time: int = 2,
    n_freq: int = 2,
    max_time_frac: float = 0.10,
    max_freq_frac: float = 0.20,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Time/frequency masking for [T, n_mels] features."""
    if feats.ndim != 2:
        raise ValueError(
            f"Expected [T, M], got {tuple(feats.shape)}"
        )

    T, M = feats.shape

    if T == 0 or M == 0:
        return feats.clone()

    out = feats.clone()
    fill = feats.mean()

    max_time_width = max(
        int(T * max_time_frac),
        0,
    )

    max_freq_width = max(
        int(M * max_freq_frac),
        0,
    )

    for _ in range(n_time):
        width = _torch_randint_inclusive(
            0,
            max_time_width,
            generator,
        )

        if width > 0:
            start = _torch_randint_inclusive(
                0,
                T - width,
                generator,
            )

            out[
                start:start + width,
                :
            ] = fill

    for _ in range(n_freq):
        width = _torch_randint_inclusive(
            0,
            max_freq_width,
            generator,
        )

        if width > 0:
            start = _torch_randint_inclusive(
                0,
                M - width,
                generator,
            )

            out[
                :,
                start:start + width
            ] = fill

    return out


# ---------------------------------------------------------------------
# Efficient audio I/O
# ---------------------------------------------------------------------

def _downmix(
    x: np.ndarray,
) -> np.ndarray:
    if x.ndim > 1:
        return x.mean(axis=1)

    return x


def _read_noise_segment(
    path: str,
    n: int,
    n_frames_total: int,
    rng: random.Random,
) -> torch.Tensor:
    """Read a random segment without decoding a whole long recording."""
    if n <= 0:
        return torch.empty(
            0,
            dtype=torch.float32,
        )

    if n_frames_total <= n:
        x = sf.read(
            path,
            dtype="float32",
            always_2d=False,
        )[0]

        x = _downmix(x)

        if len(x) == 0:
            raise RuntimeError(
                f"Empty noise file: {path}"
            )

        repeats = (
            n // len(x)
            + 1
        )

        x = np.tile(
            x,
            repeats,
        )[:n]

    else:
        start = rng.randint(
            0,
            n_frames_total - n,
        )

        x = sf.read(
            path,
            start=start,
            frames=n,
            dtype="float32",
            always_2d=False,
        )[0]

        x = _downmix(x)

        if len(x) < n:
            x = np.pad(
                x,
                (0, n - len(x)),
            )

    return torch.from_numpy(
        np.ascontiguousarray(x)
    )


@lru_cache(maxsize=1024)
def _load_rir_cached(
    path: str,
) -> torch.Tensor:
    x, _ = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )

    x = _downmix(x)

    return torch.from_numpy(
        np.ascontiguousarray(x)
    )


# ---------------------------------------------------------------------
# External augmentation banks
# ---------------------------------------------------------------------

class NoiseBank:
    """MUSAN music/noise bank with deterministic file-disjoint test split."""

    def __init__(
        self,
        musan_root: str,
        split: str = "train",
        test_frac: float = 0.15,
        seed: int = 42,
    ):
        if split not in {"train", "test"}:
            raise ValueError(
                f"split must be train or test, got {split}"
            )

        root = Path(musan_root)

        files: List[Path] = []

        for category in (
            "music",
            "noise",
        ):
            files += sorted(
                (root / category).rglob("*.wav")
            )

        rng = random.Random(seed)
        rng.shuffle(files)

        cut = int(
            len(files) * (1.0 - test_frac)
        )

        selected = (
            files[:cut]
            if split == "train"
            else files[cut:]
        )

        self.files = [
            str(path)
            for path in selected
        ]

        if not self.files:
            raise RuntimeError(
                f"No MUSAN wav files found for split={split} "
                f"under {musan_root}. Expected music/ and noise/."
            )

        self._nframes: Dict[str, int] = {}

    def _frames(
        self,
        path: str,
    ) -> int:
        n_frames = self._nframes.get(path)

        if n_frames is None:
            n_frames = sf.info(path).frames
            self._nframes[path] = n_frames

        return n_frames

    def sample(
        self,
        n: int,
        rng: random.Random,
    ) -> torch.Tensor:
        path = self.files[
            rng.randrange(len(self.files))
        ]

        return _read_noise_segment(
            path,
            n,
            self._frames(path),
            rng,
        )


class RIRBank:
    """RIR bank with deterministic file-disjoint test split."""

    def __init__(
        self,
        rirs_root: str,
        split: str = "train",
        test_frac: float = 0.15,
        seed: int = 43,
    ):
        if split not in {"train", "test"}:
            raise ValueError(
                f"split must be train or test, got {split}"
            )

        files = sorted(
            str(path)
            for path in Path(rirs_root).rglob("*.wav")
        )

        rng = random.Random(seed)
        rng.shuffle(files)

        cut = int(
            len(files) * (1.0 - test_frac)
        )

        self.files = (
            files[:cut]
            if split == "train"
            else files[cut:]
        )

        if not self.files:
            raise RuntimeError(
                f"No RIR wav files found for split={split} "
                f"under {rirs_root}"
            )

    def sample(
        self,
        rng: random.Random,
    ) -> torch.Tensor:
        path = self.files[
            rng.randrange(len(self.files))
        ]

        return _load_rir_cached(path)


# ---------------------------------------------------------------------
# Augmentation hook
# ---------------------------------------------------------------------

def make_augment_fn(
    cfg_aug: dict,
    noise_bank: Optional[NoiseBank],
    rir_bank: Optional[RIRBank],
    seed: int = 42,
):
    """Build augment_fn(wav, sr) with one independent RNG per worker."""
    snr_lo, snr_hi = cfg_aug.get(
        "snr_db_range",
        [-5, 20],
    )

    p_noise = float(
        cfg_aug.get(
            "p_noise",
            0.8,
        )
    )

    p_reverb = float(
        cfg_aug.get(
            "p_reverb",
            0.5,
        )
    )

    rngs: Dict[int, random.Random] = {}

    def get_rng() -> random.Random:
        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = -1
            worker_seed = int(seed)
        else:
            worker_id = int(
                worker_info.id
            )

            worker_seed = int(
                worker_info.seed
                % (2 ** 32)
            )

        if worker_id not in rngs:
            rngs[worker_id] = random.Random(
                worker_seed
            )

        return rngs[worker_id]

    def augment_fn(
        wav,
        sr,
    ):
        del sr

        rng = get_rng()

        x = (
            wav
            if isinstance(wav, torch.Tensor)
            else torch.as_tensor(wav)
        )

        x = x.float()

        if (
            rir_bank is not None
            and rng.random() < p_reverb
        ):
            rir = rir_bank.sample(rng)

            x = add_reverb(
                x,
                rir,
            )

        if (
            noise_bank is not None
            and rng.random() < p_noise
        ):
            snr_db = rng.uniform(
                float(snr_lo),
                float(snr_hi),
            )

            noise = noise_bank.sample(
                x.numel(),
                rng,
            )

            x = add_noise(
                x,
                noise,
                snr_db,
            )

        return x

    return augment_fn
