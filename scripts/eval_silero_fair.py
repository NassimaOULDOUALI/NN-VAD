#!/usr/bin/env python3

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
)

from silero_vad import load_silero_vad

from src.utils import load_config, set_seed
from src.data.dataset import list_utterances
from src.data.splits import speaker_disjoint_split
from src.data.features import make_feature_fn
from src.train import build_model
from src.evaluate import load_reference, crnn_probs
from src.baselines import energy_scores
from src.postprocess import full_postprocess


CONFIG = "configs/config_full.yaml"
CHECKPOINT = "checkpoints/crnn_full_job834609_best.pt"

OUT = Path("report/results")
FIG = Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def binary_metrics(y, pred):
    y = np.asarray(y).astype(np.uint8)
    pred = np.asarray(pred).astype(np.uint8)

    tp = np.sum((y == 1) & (pred == 1))
    fp = np.sum((y == 0) & (pred == 1))
    fn = np.sum((y == 1) & (pred == 0))
    tn = np.sum((y == 0) & (pred == 0))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    fa = fp / max(fp + tn, 1)
    miss = fn / max(fn + tp, 1)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarm_rate": float(fa),
        "miss_rate": float(miss),
    }


def threshold_max_f1(y, scores):
    p, r, thresholds = precision_recall_curve(y, scores)

    if len(thresholds) == 0:
        return 0.5

    f1 = (
        2 * p[:-1] * r[:-1]
        / np.maximum(p[:-1] + r[:-1], 1e-12)
    )

    return float(thresholds[int(np.argmax(f1))])


def threshold_for_fa(y, scores, target_fa):
    fpr, tpr, thresholds = roc_curve(y, scores)

    finite = np.isfinite(thresholds)
    fpr = fpr[finite]
    thresholds = thresholds[finite]

    idx = int(np.argmin(np.abs(fpr - target_fa)))

    return float(thresholds[idx]), float(fpr[idx])


@torch.inference_mode()
def silero_frame_probs(model, wav, sr, n_frames, hop_s):
    if sr != 16000:
        raise ValueError(
            f"Expected 16 kHz audio, got {sr}"
        )

    x = torch.as_tensor(
        wav,
        dtype=torch.float32,
    ).flatten()

    window = 512

    model.reset_states()

    chunk_probs = []

    for start in range(0, len(x), window):
        chunk = x[start:start + window]

        if len(chunk) < window:
            chunk = torch.nn.functional.pad(
                chunk,
                (0, window - len(chunk)),
            )

        p = model(
            chunk,
            sr,
        ).item()

        chunk_probs.append(p)

    model.reset_states()

    if not chunk_probs:
        return np.zeros(n_frames, dtype=np.float32)

    chunk_probs = np.asarray(
        chunk_probs,
        dtype=np.float32,
    )

    # Reference frames are defined by their centre.
    frame_sample = (
        (np.arange(n_frames) + 0.5)
        * hop_s
        * sr
    )

    chunk_idx = np.floor(
        frame_sample / window
    ).astype(int)

    chunk_idx = np.clip(
        chunk_idx,
        0,
        len(chunk_probs) - 1,
    )

    return chunk_probs[chunk_idx]


def postprocess_cfg(cfg):
    hop = cfg["features"]["hop_ms"]
    pp = cfg["postprocess"]

    return {
        "smooth_frames": max(
            1,
            round(pp["smooth_median_ms"] / hop),
        ),
        "min_speech_frames": max(
            0,
            round(pp["min_speech_ms"] / hop),
        ),
        "min_silence_frames": max(
            0,
            round(pp["min_silence_ms"] / hop),
        ),
        "hangover_frames": max(
            0,
            round(pp["hangover_ms"] / hop),
        ),
    }


def collect_split(
    stems,
    cfg,
    crnn,
    silero,
    feature_fn,
    device,
):
    refs = []
    crnn_scores = []
    silero_scores = []
    energy = []

    per_utt = []

    hop_s = cfg["features"]["hop_ms"] / 1000.0
    merge_gap_s = (
        cfg["labels"]["merge_gaps_below_ms"]
        / 1000.0
    )

    for i, stem in enumerate(stems, 1):
        wav, sr, ref = load_reference(
            stem,
            cfg["paths"]["provided_data"],
            hop_s,
            merge_gap_s,
        )

        c = crnn_probs(
            crnn,
            wav,
            sr,
            feature_fn,
            device,
            len(ref),
        )

        s = silero_frame_probs(
            silero,
            wav,
            sr,
            len(ref),
            hop_s,
        )

        e = energy_scores(
            wav,
            sr,
            n_frames=len(ref),
            hop_ms=cfg["features"]["hop_ms"],
            win_ms=cfg["features"]["win_ms"],
        )

        n = min(
            len(ref),
            len(c),
            len(s),
            len(e),
        )

        ref = np.asarray(ref[:n], dtype=np.uint8)
        c = np.asarray(c[:n], dtype=np.float32)
        s = np.asarray(s[:n], dtype=np.float32)
        e = np.asarray(e[:n], dtype=np.float32)

        refs.append(ref)
        crnn_scores.append(c)
        silero_scores.append(s)
        energy.append(e)

        per_utt.append({
            "stem": stem,
            "ref": ref,
            "crnn": c,
            "silero": s,
            "energy": e,
        })

        if i % 20 == 0 or i == len(stems):
            print(f"[scores] {i}/{len(stems)}")

    return {
        "ref": np.concatenate(refs),
        "crnn": np.concatenate(crnn_scores),
        "silero": np.concatenate(silero_scores),
        "energy": np.concatenate(energy),
        "per_utt": per_utt,
    }


def bootstrap_f1(
    utterances,
    crnn_threshold,
    silero_threshold,
    pp,
    n_boot=10000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    crnn_vals = np.empty(n_boot)
    silero_vals = np.empty(n_boot)
    delta_vals = np.empty(n_boot)

    # Freeze final predictions per utterance first.
    cached = []

    for item in utterances:
        ref = item["ref"]

        crnn_pred = full_postprocess(
            item["crnn"],
            crnn_threshold,
            pp["smooth_frames"],
            pp["min_speech_frames"],
            pp["min_silence_frames"],
            pp["hangover_frames"],
        ).astype(np.uint8)

        silero_pred = (
            item["silero"] >= silero_threshold
        ).astype(np.uint8)

        cached.append(
            (ref, crnn_pred, silero_pred)
        )

    n = len(cached)

    for b in range(n_boot):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        refs = np.concatenate(
            [cached[i][0] for i in indices]
        )
        cp = np.concatenate(
            [cached[i][1] for i in indices]
        )
        sp = np.concatenate(
            [cached[i][2] for i in indices]
        )

        cf1 = binary_metrics(
            refs,
            cp,
        )["f1"]

        sf1 = binary_metrics(
            refs,
            sp,
        )["f1"]

        crnn_vals[b] = cf1
        silero_vals[b] = sf1
        delta_vals[b] = cf1 - sf1

    return {
        "crnn": crnn_vals,
        "silero": silero_vals,
        "delta": delta_vals,
    }


def ci(x):
    return np.percentile(
        x,
        [2.5, 50, 97.5],
    )


def save_curves(test):
    systems = {
        "CRNN": test["crnn"],
        "Silero": test["silero"],
        "Energy": test["energy"],
    }

    y = test["ref"]

    # ROC
    plt.figure(figsize=(6.5, 5.0))

    for name, scores in systems.items():
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUROC={auc:.4f})",
        )

    plt.xlabel("False alarm rate")
    plt.ylabel("True positive rate")
    plt.title("ROC — held-out clean test")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIG / "roc_all_systems.png",
        dpi=220,
    )
    plt.close()

    # Precision-recall
    plt.figure(figsize=(6.5, 5.0))

    for name, scores in systems.items():
        p, r, _ = precision_recall_curve(
            y,
            scores,
        )

        ap = average_precision_score(
            y,
            scores,
        )

        plt.plot(
            r,
            p,
            label=f"{name} (AUPRC={ap:.4f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall — held-out clean test")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIG / "precision_recall_all_systems.png",
        dpi=220,
    )
    plt.close()

    # DET
    plt.figure(figsize=(6.5, 5.0))

    eps = 1e-5

    for name, scores in systems.items():
        fpr, tpr, _ = roc_curve(y, scores)

        fnr = 1.0 - tpr

        x = norm.ppf(
            np.clip(fpr, eps, 1 - eps)
        )
        yy = norm.ppf(
            np.clip(fnr, eps, 1 - eps)
        )

        plt.plot(x, yy, label=name)

    probs = np.asarray(
        [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    )

    ticks = norm.ppf(probs)
    labels = [
        f"{100*p:g}%"
        for p in probs
    ]

    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)

    plt.xlabel("False alarm probability")
    plt.ylabel("Miss probability")
    plt.title("DET — held-out clean test")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIG / "det_all_systems.png",
        dpi=220,
    )
    plt.close()


def main():
    cfg = load_config(CONFIG)
    set_seed(cfg["seed"])

    device = torch.device("cpu")

    stems = list_utterances(
        cfg["paths"]["provided_data"]
    )

    split = speaker_disjoint_split(
        stems,
        tuple(
            cfg["split"][name]
            for name in ("train", "val", "test")
        ),
        cfg["seed"],
    )

    print("===== SPLIT =====")
    print("val :", len(split["val"]))
    print("test:", len(split["test"]))

    ckpt = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    crnn = build_model(cfg)
    crnn.load_state_dict(ckpt["model"])
    crnn.eval()

    crnn_threshold = float(
        ckpt["threshold"]
    )

    feature_fn = make_feature_fn(
        cfg["features"]
    )

    silero = load_silero_vad()
    silero.eval()

    print()
    print("===== VALIDATION SCORES =====")

    val = collect_split(
        split["val"],
        cfg,
        crnn,
        silero,
        feature_fn,
        device,
    )

    print()
    print("===== TEST SCORES =====")

    test = collect_split(
        split["test"],
        cfg,
        crnn,
        silero,
        feature_fn,
        device,
    )

    # ------------------------------------------------------
    # Validation-calibrated Silero max-F1 threshold
    # ------------------------------------------------------

    silero_threshold = threshold_max_f1(
        val["ref"],
        val["silero"],
    )

    energy_threshold = threshold_max_f1(
        val["ref"],
        val["energy"],
    )

    print()
    print("===== VALIDATION THRESHOLDS =====")
    print(
        f"CRNN   : {crnn_threshold:.6f}"
    )
    print(
        f"Silero : {silero_threshold:.6f}"
    )
    print(
        f"Energy : {energy_threshold:.6f}"
    )

    # ------------------------------------------------------
    # Raw CRNN validation operating point
    # Used as target for matched-FA comparison.
    # ------------------------------------------------------

    val_crnn_raw = binary_metrics(
        val["ref"],
        val["crnn"] >= crnn_threshold,
    )

    target_fa = 0.07574621268936553

    silero_fa_threshold, achieved_val_fa = (
        threshold_for_fa(
            val["ref"],
            val["silero"],
            target_fa,
        )
    )

    print()
    print("===== MATCHED FA ON VALIDATION =====")
    print(
        f"CRNN raw validation FA : {target_fa:.6f}"
    )
    print(
        "Silero threshold       : "
        f"{silero_fa_threshold:.6f}"
    )
    print(
        "Silero achieved val FA : "
        f"{achieved_val_fa:.6f}"
    )

    # ------------------------------------------------------
    # Test metrics
    # ------------------------------------------------------

    pp = postprocess_cfg(cfg)

    crnn_final_preds = []

    for item in test["per_utt"]:
        crnn_final_preds.append(
            full_postprocess(
                item["crnn"],
                crnn_threshold,
                pp["smooth_frames"],
                pp["min_speech_frames"],
                pp["min_silence_frames"],
                pp["hangover_frames"],
            )
        )

    crnn_final_preds = np.concatenate(
        crnn_final_preds
    ).astype(np.uint8)

    crnn_raw = binary_metrics(
        test["ref"],
        test["crnn"] >= crnn_threshold,
    )

    crnn_final = binary_metrics(
        test["ref"],
        crnn_final_preds,
    )

    silero_cal = binary_metrics(
        test["ref"],
        test["silero"] >= silero_threshold,
    )

    silero_matched = binary_metrics(
        test["ref"],
        test["silero"] >= silero_fa_threshold,
    )

    energy = binary_metrics(
        test["ref"],
        test["energy"] >= energy_threshold,
    )

    print()
    print("===== CLEAN TEST =====")

    rows = [
        ("CRNN raw", crnn_raw),
        ("CRNN + post", crnn_final),
        ("Silero calibrated", silero_cal),
        ("Silero matched-FA", silero_matched),
        ("Energy", energy),
    ]

    for name, m in rows:
        print(
            f"{name:20s} | "
            f"F1={m['f1']:.6f} | "
            f"P={m['precision']:.6f} | "
            f"R={m['recall']:.6f} | "
            f"FA={m['false_alarm_rate']:.6f} | "
            f"Miss={m['miss_rate']:.6f}"
        )

    print()
    print("===== CONTINUOUS TEST METRICS =====")

    for name, scores in (
        ("CRNN", test["crnn"]),
        ("Silero", test["silero"]),
        ("Energy", test["energy"]),
    ):
        print(
            f"{name:8s} | "
            f"AUROC={roc_auc_score(test['ref'], scores):.6f} | "
            f"AUPRC={average_precision_score(test['ref'], scores):.6f}"
        )

    # ------------------------------------------------------
    # Bootstrap final CRNN vs calibrated Silero
    # ------------------------------------------------------

    print()
    print("===== BOOTSTRAP CLEAN TEST =====")

    boot = bootstrap_f1(
        test["per_utt"],
        crnn_threshold,
        silero_threshold,
        pp,
        n_boot=10000,
        seed=42,
    )

    c = ci(boot["crnn"])
    s = ci(boot["silero"])
    d = ci(boot["delta"])

    print(
        "CRNN F1 95% CI   : "
        f"[{c[0]:.6f}, {c[2]:.6f}] "
        f"(median={c[1]:.6f})"
    )

    print(
        "Silero F1 95% CI : "
        f"[{s[0]:.6f}, {s[2]:.6f}] "
        f"(median={s[1]:.6f})"
    )

    print(
        "Delta F1 95% CI  : "
        f"[{d[0]:.6f}, {d[2]:.6f}] "
        f"(median={d[1]:.6f})"
    )

    # Bootstrap figure
    plt.figure(figsize=(6.5, 4.5))
    plt.hist(
        boot["delta"],
        bins=60,
    )
    plt.axvline(
        0.0,
        linestyle="--",
        linewidth=1.2,
    )
    plt.xlabel(
        "F1(CRNN + post) - F1(Silero calibrated)"
    )
    plt.ylabel("Bootstrap samples")
    plt.title(
        "Paired utterance bootstrap — clean test"
    )
    plt.tight_layout()
    plt.savefig(
        FIG / "bootstrap_delta_f1.png",
        dpi=220,
    )
    plt.close()

    # Curves with all three continuous-score systems.
    save_curves(test)

    # ------------------------------------------------------
    # CSV
    # ------------------------------------------------------

    with (
        OUT / "silero_fair_comparison.csv"
    ).open("w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "system",
            "precision",
            "recall",
            "f1",
            "false_alarm_rate",
            "miss_rate",
        ])

        for name, m in rows:
            writer.writerow([
                name,
                m["precision"],
                m["recall"],
                m["f1"],
                m["false_alarm_rate"],
                m["miss_rate"],
            ])

    np.savez_compressed(
        OUT / "silero_fair_bootstrap.npz",
        crnn=boot["crnn"],
        silero=boot["silero"],
        delta=boot["delta"],
    )

    print()
    print("===== SAVED =====")
    print(OUT / "silero_fair_comparison.csv")
    print(OUT / "silero_fair_bootstrap.npz")
    print(FIG / "roc_all_systems.png")
    print(FIG / "precision_recall_all_systems.png")
    print(FIG / "det_all_systems.png")
    print(FIG / "bootstrap_delta_f1.png")


if __name__ == "__main__":
    main()
