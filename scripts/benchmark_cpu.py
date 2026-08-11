from __future__ import annotations

import statistics
import time

import numpy as np
import torch

from src.utils import load_config
from src.train import build_model
from src.data.features import make_feature_fn
from src.postprocess import full_postprocess


CONFIG = "configs/config_full.yaml"
CHECKPOINT = "checkpoints/crnn_full_job834609_best.pt"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

cfg = load_config(CONFIG)

ckpt = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

model = build_model(cfg).cpu().eval()
model.load_state_dict(ckpt["model"])

threshold = float(ckpt["threshold"])
feature_fn = make_feature_fn(cfg["features"])

sr = int(cfg["audio"]["sample_rate"])
fps = int(round(1000 / cfg["features"]["hop_ms"]))

pp = cfg["postprocess"]

smooth_frames = max(
    1,
    round(pp["smooth_median_ms"] / cfg["features"]["hop_ms"]),
)
min_speech_frames = max(
    0,
    round(pp["min_speech_ms"] / cfg["features"]["hop_ms"]),
)
min_silence_frames = max(
    0,
    round(pp["min_silence_ms"] / cfg["features"]["hop_ms"]),
)
hangover_frames = max(
    0,
    round(pp["hangover_ms"] / cfg["features"]["hop_ms"]),
)

n_params = sum(p.numel() for p in model.parameters())
fp32_mb = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
) / 1e6

print(f"threads        : {torch.get_num_threads()}")
print(f"parameters     : {n_params:,}")
print(f"FP32 weights   : {fp32_mb:.3f} MB")
print(f"threshold      : {threshold:.6f}")
print()


def bench(fn, warmup=3, repeats=20):
    for _ in range(warmup):
        fn()

    samples = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)

    return {
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "p95": sorted(samples)[
            min(
                len(samples) - 1,
                int(np.ceil(0.95 * len(samples))) - 1,
            )
        ],
    }


for seconds in (1, 10, 30, 60):
    n_samples = seconds * sr
    n_frames = seconds * fps

    rng = np.random.default_rng(42 + seconds)

    wav = (
        0.05
        * rng.standard_normal(n_samples)
    ).astype(np.float32)

    feats = torch.randn(
        1,
        n_frames,
        cfg["features"]["n_mels"],
    )

    @torch.inference_mode()
    def model_only():
        model(feats)

    @torch.inference_mode()
    def end_to_end():
        f = feature_fn(wav, sr)

        logits = model(
            f.unsqueeze(0)
        )

        probs = (
            torch.sigmoid(logits)
            .squeeze(0)
            .numpy()
        )

        full_postprocess(
            probs,
            threshold,
            smooth_frames,
            min_speech_frames,
            min_silence_frames,
            hangover_frames,
        )

    b_model = bench(model_only)
    b_e2e = bench(end_to_end)

    print(
        f"{seconds:>2d}s | "
        f"model median={b_model['median']*1000:8.2f} ms "
        f"RTF={b_model['median']/seconds:.6f} | "
        f"E2E median={b_e2e['median']*1000:8.2f} ms "
        f"RTF={b_e2e['median']/seconds:.6f} | "
        f"E2E p95={b_e2e['p95']*1000:8.2f} ms"
    )

print()
print("Algorithmic look-ahead:")
print("  neural model : 0 ms")
print("  frontend     : 12.5 ms (25-ms centered window)")
