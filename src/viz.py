"""Figures used in the VAD technical report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def _prepare_path(out_path):
    path = Path(out_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    return path


def plot_mask_overlay(
    wav,
    sr,
    ref_mask,
    pred_mask,
    probs,
    out_path,
    hop_s=0.010,
):
    """Waveform, probability trace, and reference/predicted VAD masks."""
    path = _prepare_path(
        out_path
    )

    wav = np.asarray(wav)
    ref_mask = np.asarray(ref_mask)
    pred_mask = np.asarray(pred_mask)
    probs = np.asarray(probs)

    t_wav = (
        np.arange(len(wav))
        / sr
    )

    t_frames = (
        np.arange(len(probs))
        + 0.5
    ) * hop_s

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 7),
        sharex=True,
    )

    axes[0].plot(
        t_wav,
        wav,
        linewidth=0.7,
    )
    axes[0].set_ylabel(
        "Amplitude"
    )
    axes[0].set_title(
        "VAD qualitative example"
    )

    axes[1].plot(
        t_frames,
        probs,
        label="CRNN speech probability",
    )
    axes[1].set_ylim(
        -0.03,
        1.03,
    )
    axes[1].set_ylabel(
        "Probability"
    )
    axes[1].legend(
        loc="upper right"
    )

    axes[2].step(
        t_frames[:len(ref_mask)],
        ref_mask,
        where="mid",
        label="Reference",
    )

    axes[2].step(
        t_frames[:len(pred_mask)],
        pred_mask,
        where="mid",
        label="Prediction",
    )

    axes[2].set_ylim(
        -0.1,
        1.1,
    )
    axes[2].set_ylabel(
        "Speech"
    )
    axes[2].set_xlabel(
        "Time (s)"
    )
    axes[2].legend(
        loc="upper right"
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_f1_vs_snr(
    results,
    out_path,
):
    """Plot F1 against SNR for several VAD systems.

    results:
        {
            "CRNN": {snr: f1, ...},
            "Silero": {snr: f1, ...},
            ...
        }
    """
    path = _prepare_path(
        out_path
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for system, values in results.items():
        snrs = sorted(
            values.keys()
        )

        f1s = [
            values[snr]
            for snr in snrs
        ]

        ax.plot(
            snrs,
            f1s,
            marker="o",
            label=system,
        )

    ax.set_xlabel(
        "SNR (dB)"
    )
    ax.set_ylabel(
        "Frame F1"
    )
    ax.set_title(
        "Noise robustness"
    )
    ax.set_ylim(
        0.0,
        1.02,
    )
    ax.grid(
        True,
        alpha=0.25,
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_det(
    det_by_system,
    out_path,
):
    """Plot DET curves using normal-deviate axes."""
    path = _prepare_path(
        out_path
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    eps = 1e-5

    for system, (
        false_alarm,
        miss,
    ) in det_by_system.items():

        fa = np.clip(
            np.asarray(false_alarm),
            eps,
            1.0 - eps,
        )

        mr = np.clip(
            np.asarray(miss),
            eps,
            1.0 - eps,
        )

        if len(fa) == 0:
            continue

        ax.plot(
            norm.ppf(fa),
            norm.ppf(mr),
            label=system,
        )

    ticks = np.asarray(
        [
            0.001,
            0.005,
            0.01,
            0.02,
            0.05,
            0.10,
            0.20,
            0.40,
            0.60,
            0.80,
        ]
    )

    tick_pos = norm.ppf(
        ticks
    )

    tick_labels = [
        f"{100 * x:g}"
        for x in ticks
    ]

    ax.set_xticks(
        tick_pos
    )
    ax.set_xticklabels(
        tick_labels,
    )

    ax.set_yticks(
        tick_pos
    )
    ax.set_yticklabels(
        tick_labels,
    )

    ax.set_xlabel(
        "False alarm probability (%)"
    )
    ax.set_ylabel(
        "Miss probability (%)"
    )
    ax.set_title(
        "Detection Error Tradeoff (DET)"
    )
    ax.grid(
        True,
        alpha=0.25,
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_roc(
    roc_by_system,
    out_path,
):
    path = _prepare_path(
        out_path
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    for system, (
        fpr,
        tpr,
        auc_value,
    ) in roc_by_system.items():
        ax.plot(
            fpr,
            tpr,
            label=(
                f"{system} "
                f"(AUROC={auc_value:.4f})"
            ),
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "False positive rate"
    )
    ax.set_ylabel(
        "True positive rate"
    )
    ax.set_title(
        "ROC curve"
    )
    ax.legend()
    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_pr(
    pr_by_system,
    out_path,
):
    path = _prepare_path(
        out_path
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    for system, (
        precision,
        recall,
        auc_value,
    ) in pr_by_system.items():
        ax.plot(
            recall,
            precision,
            label=(
                f"{system} "
                f"(AUPRC={auc_value:.4f})"
            ),
        )

    ax.set_xlabel(
        "Recall"
    )
    ax.set_ylabel(
        "Precision"
    )
    ax.set_title(
        "Precision–Recall curve"
    )
    ax.set_xlim(
        0,
        1,
    )
    ax.set_ylim(
        0,
        1.02,
    )
    ax.legend()
    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_curves(
    history,
    out_path,
):
    """Optional generic training-history plot."""
    path = _prepare_path(
        out_path
    )

    epochs = history.get(
        "epoch",
        list(
            range(
                1,
                len(
                    history.get(
                        "val_f1",
                        [],
                    )
                )
                + 1,
            )
        ),
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    if "train_loss" in history:
        ax.plot(
            epochs,
            history["train_loss"],
            label="Train loss",
        )

    if "val_f1" in history:
        ax.plot(
            epochs,
            history["val_f1"],
            label="Validation F1",
        )

    ax.set_xlabel(
        "Epoch"
    )
    ax.set_title(
        "Training history"
    )
    ax.grid(
        True,
        alpha=0.25,
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
