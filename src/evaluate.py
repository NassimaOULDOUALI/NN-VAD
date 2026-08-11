"""Final evaluation of the trained CRNN VAD.

Protocol
--------
1. Reconstruct the deterministic speaker-disjoint train/val/test split.
2. Load the best training checkpoint.
3. Freeze the CRNN threshold stored in the checkpoint (selected on val).
4. Select the Energy baseline threshold on validation only.
5. Evaluate all systems on the untouched test split.
6. Evaluate post-processing ablations.
7. Evaluate robustness on test utterances mixed with held-out MUSAN files.
8. Save tables and figures under report/figures and report/results.

Usage
-----
python -m src.evaluate \
    --config configs/config_full.yaml \
    --checkpoint checkpoints/crnn_full_job834609_best.pt
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch

from .baselines import (
    energy_scores,
    load_silero,
    silero_vad,
    webrtc_vad,
)
from .data.dataset import (
    build_frame_mask,
    list_utterances,
    read_wav,
)
from .data.features import (
    make_feature_fn,
)
from .data.mixing import (
    NoiseBank,
    add_noise,
)
from .data.splits import (
    speaker_disjoint_split,
    split_summary,
)
from .metrics import (
    det_points,
    frame_metrics,
    roc_auc,
    segment_detection_error_rate,
    select_threshold,
)
from .postprocess import (
    apply_hangover,
    apply_min_speech,
    full_postprocess,
    mask_to_segments,
    median_smooth,
    merge_short_silence,
)
from .train import build_model
from .utils import (
    load_config,
    set_seed,
)
from .viz import (
    plot_det,
    plot_f1_vs_snr,
    plot_mask_overlay,
    plot_pr,
    plot_roc,
)


# =====================================================================
# Utilities
# =====================================================================

def save_csv(
    rows,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def safe_float(
    value,
):
    value = float(value)

    if math.isnan(value):
        return float("nan")

    return value


def concatenate(
    arrays,
    dtype=None,
):
    arrays = [
        np.asarray(x)
        for x in arrays
        if len(x) > 0
    ]

    if not arrays:
        return np.empty(
            0,
            dtype=(
                dtype
                if dtype is not None
                else np.float32
            ),
        )

    out = np.concatenate(
        arrays
    )

    if dtype is not None:
        out = out.astype(
            dtype
        )

    return out


# =====================================================================
# Reference loading
# =====================================================================

def load_reference(
    stem,
    root,
    hop_s,
    merge_gap_s,
):
    wav, sr = read_wav(
        Path(root)
        / f"{stem}.wav"
    )

    mask = build_frame_mask(
        Path(root)
        / f"{stem}.json",
        duration_s=(
            len(wav) / sr
        ),
        hop_s=hop_s,
        merge_gap_s=merge_gap_s,
    )

    return wav, sr, mask


# =====================================================================
# CRNN inference
# =====================================================================

@torch.no_grad()
def crnn_probs(
    model,
    wav,
    sr,
    feature_fn,
    device,
    n_frames,
):
    feats = feature_fn(
        wav,
        sr,
    )

    feats = torch.as_tensor(
        feats,
        dtype=torch.float32,
    )

    length = min(
        len(feats),
        n_frames,
    )

    if length <= 0:
        return np.empty(
            0,
            dtype=np.float32,
        )

    feats = feats[
        :length
    ]

    logits = model(
        feats
        .unsqueeze(0)
        .to(
            device,
            non_blocking=True,
        )
    )

    probs = (
        torch.sigmoid(
            logits
        )
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    return probs[:length]


# =====================================================================
# Corpus DER
# =====================================================================

def corpus_der(
    pred_masks,
    ref_masks,
    hop_s,
    collar_s,
):
    """Compute one time-weighted DER across all utterances.

    Utterances are shifted apart by one second so boundaries cannot merge.
    """
    pred_segments = []
    ref_segments = []

    offset = 0.0

    for pred, ref in zip(
        pred_masks,
        ref_masks,
    ):
        n = min(
            len(pred),
            len(ref),
        )

        pred = np.asarray(
            pred[:n],
            dtype=np.uint8,
        )

        ref = np.asarray(
            ref[:n],
            dtype=np.uint8,
        )

        for start, end in mask_to_segments(
            pred,
            hop_s,
        ):
            pred_segments.append(
                (
                    start + offset,
                    end + offset,
                )
            )

        for start, end in mask_to_segments(
            ref,
            hop_s,
        ):
            ref_segments.append(
                (
                    start + offset,
                    end + offset,
                )
            )

        offset += (
            n * hop_s
            + 1.0
        )

    return segment_detection_error_rate(
        pred_segments,
        ref_segments,
        collar_s,
    )


# =====================================================================
# Metric rows
# =====================================================================

def continuous_system_metrics(
    name,
    scores_by_utt,
    refs_by_utt,
    threshold,
    hop_s,
    collar_s,
):
    scores = concatenate(
        scores_by_utt,
        np.float64,
    )

    refs = concatenate(
        refs_by_utt,
        np.int64,
    )

    metrics = frame_metrics(
        scores,
        refs,
        threshold,
    )

    (
        fpr,
        tpr,
        auroc,
    ), (
        precision_curve,
        recall_curve,
        auprc,
    ) = roc_auc(
        scores,
        refs,
    )

    pred_masks = [
        (
            np.asarray(scores_i)
            >= threshold
        ).astype(
            np.uint8
        )
        for scores_i
        in scores_by_utt
    ]

    der = corpus_der(
        pred_masks,
        refs_by_utt,
        hop_s,
        collar_s,
    )

    row = {
        "system": name,
        "threshold": float(
            threshold
        ),
        "precision": metrics[
            "precision"
        ],
        "recall": metrics[
            "recall"
        ],
        "f1": metrics[
            "f1"
        ],
        "accuracy": metrics[
            "accuracy"
        ],
        "auroc": auroc,
        "auprc": auprc,
        "false_alarm_rate": metrics[
            "false_alarm_rate"
        ],
        "miss_rate": metrics[
            "miss_rate"
        ],
        "der": der,
    }

    curves = {
        "roc": (
            fpr,
            tpr,
            auroc,
        ),
        "pr": (
            precision_curve,
            recall_curve,
            auprc,
        ),
        "det": det_points(
            scores,
            refs,
        ),
    }

    return row, pred_masks, curves


def binary_system_metrics(
    name,
    preds_by_utt,
    refs_by_utt,
    hop_s,
    collar_s,
):
    pred = concatenate(
        preds_by_utt,
        np.uint8,
    )

    ref = concatenate(
        refs_by_utt,
        np.uint8,
    )

    metrics = frame_metrics(
        pred.astype(
            np.float64
        ),
        ref,
        threshold=0.5,
    )

    der = corpus_der(
        preds_by_utt,
        refs_by_utt,
        hop_s,
        collar_s,
    )

    # AUROC/AUPRC are deliberately not reported for a binary-only
    # operating-point baseline.
    row = {
        "system": name,
        "threshold": float("nan"),
        "precision": metrics[
            "precision"
        ],
        "recall": metrics[
            "recall"
        ],
        "f1": metrics[
            "f1"
        ],
        "accuracy": metrics[
            "accuracy"
        ],
        "auroc": float("nan"),
        "auprc": float("nan"),
        "false_alarm_rate": metrics[
            "false_alarm_rate"
        ],
        "miss_rate": metrics[
            "miss_rate"
        ],
        "der": der,
    }

    return row


# =====================================================================
# Post-processing
# =====================================================================

def postprocess_parameters(
    cfg,
):
    hop_ms = float(
        cfg["features"]["hop_ms"]
    )

    pp = cfg[
        "postprocess"
    ]

    return {
        "smooth": max(
            1,
            int(
                round(
                    pp[
                        "smooth_median_ms"
                    ]
                    / hop_ms
                )
            ),
        ),
        "min_speech": max(
            0,
            int(
                round(
                    pp[
                        "min_speech_ms"
                    ]
                    / hop_ms
                )
            ),
        ),
        "min_silence": max(
            0,
            int(
                round(
                    pp[
                        "min_silence_ms"
                    ]
                    / hop_ms
                )
            ),
        ),
        "hangover": max(
            0,
            int(
                round(
                    pp[
                        "hangover_ms"
                    ]
                    / hop_ms
                )
            ),
        ),
    }


def make_postprocess_masks(
    probs_by_utt,
    threshold,
    cfg,
):
    pars = postprocess_parameters(
        cfg
    )

    stages = {
        "raw": [],
        "+median": [],
        "+min_speech": [],
        "+merge_silence": [],
        "+hangover": [],
    }

    for probs in probs_by_utt:
        probs = np.asarray(
            probs,
            dtype=np.float32,
        )

        raw = (
            probs >= threshold
        ).astype(
            np.uint8
        )

        smooth_probs = median_smooth(
            probs,
            pars["smooth"],
        )

        median = (
            smooth_probs
            >= threshold
        ).astype(
            np.uint8
        )

        min_speech = apply_min_speech(
            median,
            pars["min_speech"],
        )

        merged = merge_short_silence(
            min_speech,
            pars["min_silence"],
        )

        hangover = apply_hangover(
            merged,
            pars["hangover"],
        )

        stages["raw"].append(
            raw
        )

        stages["+median"].append(
            median
        )

        stages["+min_speech"].append(
            min_speech
        )

        stages["+merge_silence"].append(
            merged
        )

        stages["+hangover"].append(
            hangover
        )

    return stages


def postprocess_ablation(
    probs_by_utt,
    refs_by_utt,
    threshold,
    cfg,
):
    hop_s = (
        cfg["features"][
            "hop_ms"
        ]
        / 1000.0
    )

    collar_s = (
        cfg["eval"][
            "collar_ms"
        ]
        / 1000.0
    )

    stages = make_postprocess_masks(
        probs_by_utt,
        threshold,
        cfg,
    )

    rows = []

    refs = concatenate(
        refs_by_utt,
        np.uint8,
    )

    for stage_name, masks in stages.items():
        pred = concatenate(
            masks,
            np.uint8,
        )

        metrics = frame_metrics(
            pred.astype(
                np.float64
            ),
            refs,
            0.5,
        )

        rows.append(
            {
                "stage": stage_name,
                "precision": metrics[
                    "precision"
                ],
                "recall": metrics[
                    "recall"
                ],
                "f1": metrics[
                    "f1"
                ],
                "accuracy": metrics[
                    "accuracy"
                ],
                "false_alarm_rate": metrics[
                    "false_alarm_rate"
                ],
                "miss_rate": metrics[
                    "miss_rate"
                ],
                "der": corpus_der(
                    masks,
                    refs_by_utt,
                    hop_s,
                    collar_s,
                ),
            }
        )

    return rows, stages


# =====================================================================
# Energy validation threshold
# =====================================================================

def select_energy_threshold(
    stems,
    cfg,
):
    root = cfg[
        "paths"
    ][
        "provided_data"
    ]

    hop_s = (
        cfg["features"][
            "hop_ms"
        ]
        / 1000.0
    )

    merge_gap_s = (
        cfg["labels"][
            "merge_gaps_below_ms"
        ]
        / 1000.0
    )

    scores_all = []
    refs_all = []

    for stem in stems:
        wav, sr, ref = load_reference(
            stem,
            root,
            hop_s,
            merge_gap_s,
        )

        scores = energy_scores(
            wav,
            sr,
            n_frames=len(ref),
            hop_ms=cfg[
                "features"
            ][
                "hop_ms"
            ],
            win_ms=cfg[
                "features"
            ][
                "win_ms"
            ],
        )

        n = min(
            len(scores),
            len(ref),
        )

        scores_all.append(
            scores[:n]
        )

        refs_all.append(
            ref[:n]
        )

    scores = concatenate(
        scores_all,
        np.float64,
    )

    refs = concatenate(
        refs_all,
        np.int64,
    )

    threshold = select_threshold(
        scores,
        refs,
        criterion="max_f1",
    )

    return threshold


# =====================================================================
# Robustness
# =====================================================================

@torch.no_grad()
def robustness_sweep(
    test_stems,
    cfg,
    model,
    feature_fn,
    device,
    crnn_threshold,
    energy_threshold,
    use_silero,
):
    root = cfg[
        "paths"
    ][
        "provided_data"
    ]

    hop_s = (
        cfg["features"][
            "hop_ms"
        ]
        / 1000.0
    )

    merge_gap_s = (
        cfg["labels"][
            "merge_gaps_below_ms"
        ]
        / 1000.0
    )

    snrs = cfg[
        "eval"
    ].get(
        "snr_sweep_db",
        [],
    )

    noise_bank = NoiseBank(
        cfg["paths"]["musan"],
        split="test",
        seed=cfg["seed"],
    )

    print(
        f"[robustness] held-out MUSAN files: "
        f"{len(noise_bank.files)}"
    )

    results = {
        "CRNN raw": {},
        "CRNN + post": {},
        "Energy": {},
    }

    if use_silero:
        results[
            "Silero"
        ] = {}

        silero_model = load_silero()
    else:
        silero_model = None

    pars = postprocess_parameters(
        cfg
    )

    for snr_index, snr_db in enumerate(
        snrs
    ):
        rng = random.Random(
            cfg["seed"]
            + 10000
            + snr_index
        )

        refs_all = []
        crnn_raw_all = []
        crnn_post_all = []
        energy_all = []
        silero_all = []

        for stem in test_stems:
            wav, sr, ref = load_reference(
                stem,
                root,
                hop_s,
                merge_gap_s,
            )

            wav_tensor = torch.as_tensor(
                wav,
                dtype=torch.float32,
            )

            noise = noise_bank.sample(
                len(wav),
                rng,
            )

            noisy = add_noise(
                wav_tensor,
                noise,
                float(snr_db),
            ).numpy()

            probs = crnn_probs(
                model,
                noisy,
                sr,
                feature_fn,
                device,
                len(ref),
            )

            n = min(
                len(probs),
                len(ref),
            )

            probs = probs[:n]
            ref_n = ref[:n]

            raw = (
                probs
                >= crnn_threshold
            ).astype(
                np.uint8
            )

            post = full_postprocess(
                probs,
                crnn_threshold,
                pars["smooth"],
                pars["min_speech"],
                pars["min_silence"],
                pars["hangover"],
            )

            energy = (
                energy_scores(
                    noisy,
                    sr,
                    n_frames=n,
                    hop_ms=cfg[
                        "features"
                    ][
                        "hop_ms"
                    ],
                    win_ms=cfg[
                        "features"
                    ][
                        "win_ms"
                    ],
                )
                >= energy_threshold
            ).astype(
                np.uint8
            )

            refs_all.append(
                ref_n
            )

            crnn_raw_all.append(
                raw
            )

            crnn_post_all.append(
                post
            )

            energy_all.append(
                energy
            )

            if use_silero:
                sil = silero_vad(
                    noisy,
                    sr,
                    n_frames=n,
                    hop_s=hop_s,
                    model=silero_model,
                )

                silero_all.append(
                    sil[:n]
                )

        refs_flat = concatenate(
            refs_all,
            np.uint8,
        )

        def f1_from_masks(masks):
            pred = concatenate(
                masks,
                np.uint8,
            )

            return frame_metrics(
                pred.astype(
                    np.float64
                ),
                refs_flat,
                0.5,
            )["f1"]

        results[
            "CRNN raw"
        ][
            float(snr_db)
        ] = f1_from_masks(
            crnn_raw_all
        )

        results[
            "CRNN + post"
        ][
            float(snr_db)
        ] = f1_from_masks(
            crnn_post_all
        )

        results[
            "Energy"
        ][
            float(snr_db)
        ] = f1_from_masks(
            energy_all
        )

        if use_silero:
            results[
                "Silero"
            ][
                float(snr_db)
            ] = f1_from_masks(
                silero_all
            )

        print(
            f"[robustness] SNR={snr_db:>4} dB | "
            f"CRNN raw="
            f"{results['CRNN raw'][float(snr_db)]:.4f} | "
            f"CRNN+post="
            f"{results['CRNN + post'][float(snr_db)]:.4f} | "
            f"Energy="
            f"{results['Energy'][float(snr_db)]:.4f}"
            + (
                f" | Silero="
                f"{results['Silero'][float(snr_db)]:.4f}"
                if use_silero
                else ""
            )
        )

    rows = []

    for system, values in results.items():
        for snr_db, f1 in sorted(
            values.items()
        ):
            rows.append(
                {
                    "system": system,
                    "snr_db": snr_db,
                    "f1": f1,
                }
            )

    return results, rows


# =====================================================================
# Main
# =====================================================================

def main(
    config_path,
    checkpoint_path,
    skip_robustness=False,
    skip_silero=False,
):
    cfg = load_config(
        config_path
    )

    set_seed(
        cfg["seed"]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[device] {device}"
    )

    if device.type == "cuda":
        print(
            f"[gpu] "
            f"{torch.cuda.get_device_name(0)}"
        )

    root = cfg[
        "paths"
    ][
        "provided_data"
    ]

    hop_s = (
        cfg["features"][
            "hop_ms"
        ]
        / 1000.0
    )

    merge_gap_s = (
        cfg["labels"][
            "merge_gaps_below_ms"
        ]
        / 1000.0
    )

    collar_s = (
        cfg["eval"][
            "collar_ms"
        ]
        / 1000.0
    )

    figures_dir = Path(
        cfg["paths"][
            "report_figs"
        ]
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dir = Path(
        "report/results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Reconstruct the exact split used during training.
    # -----------------------------------------------------------------

    stems = list_utterances(
        root
    )

    split = speaker_disjoint_split(
        stems,
        tuple(
            cfg["split"][name]
            for name in (
                "train",
                "val",
                "test",
            )
        ),
        cfg["seed"],
    )

    print(
        "===== SPLIT ====="
    )

    print(
        split_summary(
            split
        )
    )

    # -----------------------------------------------------------------
    # Load checkpoint and freeze validation-selected threshold.
    # -----------------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = build_model(
        cfg
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model = model.to(
        device
    )

    model.eval()

    crnn_threshold = float(
        checkpoint[
            "threshold"
        ]
    )

    print(
        "===== CHECKPOINT ====="
    )

    print(
        f"checkpoint: {checkpoint_path}"
    )

    print(
        f"best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"validation F1: "
        f"{checkpoint['val_f1']:.6f}"
    )

    print(
        f"frozen CRNN threshold: "
        f"{crnn_threshold:.6f}"
    )

    feature_fn = make_feature_fn(
        cfg[
            "features"
        ]
    )

    # -----------------------------------------------------------------
    # Energy threshold: VALIDATION ONLY.
    # -----------------------------------------------------------------

    print(
        "===== ENERGY VALIDATION THRESHOLD ====="
    )

    energy_threshold = (
        select_energy_threshold(
            split["val"],
            cfg,
        )
    )

    print(
        f"energy threshold: "
        f"{energy_threshold:.6f}"
    )

    # -----------------------------------------------------------------
    # Load Silero once.
    # -----------------------------------------------------------------

    silero_model = None
    use_silero = not skip_silero

    if use_silero:
        try:
            print(
                "===== SILERO ====="
            )

            silero_model = (
                load_silero()
            )

            print(
                "Silero model loaded."
            )

        except Exception as exc:
            print(
                f"[warning] Silero disabled: "
                f"{exc}"
            )

            use_silero = False

    # -----------------------------------------------------------------
    # Clean test inference.
    # -----------------------------------------------------------------

    print(
        "===== CLEAN TEST ====="
    )

    crnn_probs_by_utt = []
    energy_scores_by_utt = []
    refs_by_utt = []
    silero_masks = []

    qualitative = []

    for index, stem in enumerate(
        split["test"],
        start=1,
    ):
        wav, sr, ref = load_reference(
            stem,
            root,
            hop_s,
            merge_gap_s,
        )

        probs = crnn_probs(
            model,
            wav,
            sr,
            feature_fn,
            device,
            len(ref),
        )

        n = min(
            len(probs),
            len(ref),
        )

        probs = probs[:n]
        ref = ref[:n]

        e_scores = energy_scores(
            wav,
            sr,
            n_frames=n,
            hop_ms=cfg[
                "features"
            ][
                "hop_ms"
            ],
            win_ms=cfg[
                "features"
            ][
                "win_ms"
            ],
        )[:n]

        crnn_probs_by_utt.append(
            probs
        )

        energy_scores_by_utt.append(
            e_scores
        )

        refs_by_utt.append(
            ref
        )

        if use_silero:
            sil_mask = silero_vad(
                wav,
                sr,
                n_frames=n,
                hop_s=hop_s,
                model=silero_model,
            )

            silero_masks.append(
                sil_mask[:n]
            )

        raw_pred = (
            probs
            >= crnn_threshold
        ).astype(
            np.uint8
        )

        err = float(
            np.mean(
                raw_pred
                != ref.astype(
                    np.uint8
                )
            )
        )

        qualitative.append(
            {
                "stem": stem,
                "wav": wav,
                "sr": sr,
                "ref": ref,
                "probs": probs,
                "error": err,
            }
        )

        if (
            index % 20 == 0
            or index == len(
                split["test"]
            )
        ):
            print(
                f"[test] "
                f"{index}/"
                f"{len(split['test'])}"
            )

    # -----------------------------------------------------------------
    # Main metrics.
    # -----------------------------------------------------------------

    main_rows = []

    crnn_row, crnn_raw_masks, crnn_curves = (
        continuous_system_metrics(
            "CRNN raw",
            crnn_probs_by_utt,
            refs_by_utt,
            crnn_threshold,
            hop_s,
            collar_s,
        )
    )

    main_rows.append(
        crnn_row
    )

    energy_row, energy_masks, energy_curves = (
        continuous_system_metrics(
            "Energy",
            energy_scores_by_utt,
            refs_by_utt,
            energy_threshold,
            hop_s,
            collar_s,
        )
    )

    main_rows.append(
        energy_row
    )

    if use_silero:
        silero_row = (
            binary_system_metrics(
                "Silero",
                silero_masks,
                refs_by_utt,
                hop_s,
                collar_s,
            )
        )

        main_rows.append(
            silero_row
        )

    # -----------------------------------------------------------------
    # Post-processing ablation.
    # -----------------------------------------------------------------

    ablation_rows, pp_stages = (
        postprocess_ablation(
            crnn_probs_by_utt,
            refs_by_utt,
            crnn_threshold,
            cfg,
        )
    )

    final_masks = (
        pp_stages[
            "+hangover"
        ]
    )

    refs_flat = concatenate(
        refs_by_utt,
        np.uint8,
    )

    final_flat = concatenate(
        final_masks,
        np.uint8,
    )

    final_metrics = frame_metrics(
        final_flat.astype(
            np.float64
        ),
        refs_flat,
        0.5,
    )

    crnn_post_row = {
        "system": "CRNN + postprocess",
        "threshold": crnn_threshold,
        "precision": final_metrics[
            "precision"
        ],
        "recall": final_metrics[
            "recall"
        ],
        "f1": final_metrics[
            "f1"
        ],
        "accuracy": final_metrics[
            "accuracy"
        ],
        # Continuous AUROC/AUPRC belong to raw probabilities and are
        # unchanged conceptually by binary duration post-processing.
        "auroc": crnn_row[
            "auroc"
        ],
        "auprc": crnn_row[
            "auprc"
        ],
        "false_alarm_rate": final_metrics[
            "false_alarm_rate"
        ],
        "miss_rate": final_metrics[
            "miss_rate"
        ],
        "der": corpus_der(
            final_masks,
            refs_by_utt,
            hop_s,
            collar_s,
        ),
    }

    main_rows.insert(
        1,
        crnn_post_row,
    )

    save_csv(
        main_rows,
        results_dir
        / "main_results.csv",
    )

    save_csv(
        ablation_rows,
        results_dir
        / "postprocess_ablation.csv",
    )

    # -----------------------------------------------------------------
    # Curves.
    # -----------------------------------------------------------------

    plot_roc(
        {
            "CRNN": crnn_curves[
                "roc"
            ],
            "Energy": energy_curves[
                "roc"
            ],
        },
        figures_dir
        / "roc.png",
    )

    plot_pr(
        {
            "CRNN": crnn_curves[
                "pr"
            ],
            "Energy": energy_curves[
                "pr"
            ],
        },
        figures_dir
        / "precision_recall.png",
    )

    plot_det(
        {
            "CRNN": crnn_curves[
                "det"
            ],
            "Energy": energy_curves[
                "det"
            ],
        },
        figures_dir
        / "det.png",
    )

    # -----------------------------------------------------------------
    # Qualitative example: choose the test utterance with largest raw
    # frame disagreement, not a cherry-picked best case.
    # -----------------------------------------------------------------

    if qualitative:
        example = max(
            qualitative,
            key=lambda x: x[
                "error"
            ],
        )

        example_final = full_postprocess(
            example[
                "probs"
            ],
            crnn_threshold,
            **{
                "smooth_frames":
                    postprocess_parameters(
                        cfg
                    )[
                        "smooth"
                    ],
                "min_speech_frames":
                    postprocess_parameters(
                        cfg
                    )[
                        "min_speech"
                    ],
                "min_silence_frames":
                    postprocess_parameters(
                        cfg
                    )[
                        "min_silence"
                    ],
                "hangover_frames":
                    postprocess_parameters(
                        cfg
                    )[
                        "hangover"
                    ],
            },
        )

        plot_mask_overlay(
            example[
                "wav"
            ],
            example[
                "sr"
            ],
            example[
                "ref"
            ],
            example_final,
            example[
                "probs"
            ],
            figures_dir
            / "qualitative_error_case.png",
            hop_s=hop_s,
        )

        print(
            f"[qualitative] "
            f"error-case stem: "
            f"{example['stem']}"
        )

    # -----------------------------------------------------------------
    # Robustness.
    # -----------------------------------------------------------------

    if not skip_robustness:
        print(
            "===== ROBUSTNESS ====="
        )

        robust_results, robust_rows = (
            robustness_sweep(
                split["test"],
                cfg,
                model,
                feature_fn,
                device,
                crnn_threshold,
                energy_threshold,
                use_silero,
            )
        )

        save_csv(
            robust_rows,
            results_dir
            / "snr_robustness.csv",
        )

        plot_f1_vs_snr(
            robust_results,
            figures_dir
            / "f1_vs_snr.png",
        )

    # -----------------------------------------------------------------
    # Print headline table.
    # -----------------------------------------------------------------

    print()
    print(
        "===== FINAL TEST RESULTS ====="
    )

    for row in main_rows:
        auroc = (
            f"{row['auroc']:.4f}"
            if not math.isnan(
                row["auroc"]
            )
            else "N/A"
        )

        auprc = (
            f"{row['auprc']:.4f}"
            if not math.isnan(
                row["auprc"]
            )
            else "N/A"
        )

        print(
            f"{row['system']:20s} | "
            f"F1={row['f1']:.4f} | "
            f"P={row['precision']:.4f} | "
            f"R={row['recall']:.4f} | "
            f"AUROC={auroc} | "
            f"AUPRC={auprc} | "
            f"FA={row['false_alarm_rate']:.4f} | "
            f"Miss={row['miss_rate']:.4f} | "
            f"DER={row['der']:.4f}"
        )

    print()
    print(
        "Results:"
    )
    print(
        results_dir
        / "main_results.csv"
    )
    print(
        results_dir
        / "postprocess_ablation.csv"
    )

    if not skip_robustness:
        print(
            results_dir
            / "snr_robustness.csv"
        )

    print()
    print(
        "Figures:"
    )

    for path in sorted(
        figures_dir.glob(
            "*.png"
        )
    ):
        print(
            path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/config_full.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "crnn_full_job834609_best.pt"
        ),
    )

    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help=(
            "Evaluate clean test only; "
            "skip the SNR sweep."
        ),
    )

    parser.add_argument(
        "--skip-silero",
        action="store_true",
        help=(
            "Skip Silero if the package "
            "is unavailable."
        ),
    )

    args = parser.parse_args()

    main(
        args.config,
        args.checkpoint,
        skip_robustness=(
            args.skip_robustness
        ),
        skip_silero=(
            args.skip_silero
        ),
    )
