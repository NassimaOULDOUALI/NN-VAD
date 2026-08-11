"""Training entry point for the lightweight VAD.

Usage:
    python -m src.train --config configs/config.yaml

Smoke test:
    python -m src.train \
        --config configs/config.yaml \
        --epochs 2 \
        --limit 100 \
        --no-wandb
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from .data.dataset import (
    VADDataset,
    list_utterances,
)
from .data.features import make_feature_fn
from .data.mixing import (
    NoiseBank,
    RIRBank,
    make_augment_fn,
    spec_augment,
)
from .data.splits import (
    speaker_disjoint_split,
    split_summary,
)
from .metrics import (
    frame_metrics,
    select_threshold,
)
from .models.crnn import SepConvGRUVAD
from .utils import (
    load_config,
    set_seed,
)


# =====================================================================
# Dataset wrapper
# =====================================================================

class FramedVAD(Dataset):
    """Return feature and VAD label sequences.

    Training utterances longer than ``max_frames`` are randomly cropped.

    SpecAugment is used only for the training dataset.
    """

    def __init__(
        self,
        base: VADDataset,
        max_frames: int | None = None,
        spec_aug: bool = False,
    ):
        self.base = base
        self.max_frames = max_frames
        self.spec_aug = spec_aug

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        feats, mask = self.base[index]

        feats = torch.as_tensor(
            feats,
            dtype=torch.float32,
        )

        mask = torch.as_tensor(
            mask,
            dtype=torch.float32,
        )

        length = min(
            feats.shape[0],
            mask.shape[0],
        )

        feats = feats[:length]
        mask = mask[:length]

        if (
            self.max_frames is not None
            and length > self.max_frames
        ):
            max_start = (
                length
                - self.max_frames
            )

            start = int(
                torch.randint(
                    low=0,
                    high=max_start + 1,
                    size=(1,),
                ).item()
            )

            end = (
                start
                + self.max_frames
            )

            feats = feats[
                start:end
            ]

            mask = mask[
                start:end
            ]

        if self.spec_aug:
            feats = spec_augment(feats)

        return feats, mask


# =====================================================================
# Collate
# =====================================================================

def collate(batch):
    """Pad variable length batches.

    Returns
    -------
    X : [B, T, M]
    Y : [B, T]
    P : [B, T], where 1 marks valid frames.
    """
    if len(batch) == 0:
        raise RuntimeError(
            "Received an empty batch."
        )

    lengths = [
        feats.shape[0]
        for feats, _ in batch
    ]

    if min(lengths) <= 0:
        raise RuntimeError(
            "Found an utterance with zero frames."
        )

    batch_size = len(batch)
    max_length = max(lengths)
    n_mels = batch[0][0].shape[1]

    X = torch.zeros(
        batch_size,
        max_length,
        n_mels,
        dtype=torch.float32,
    )

    Y = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.float32,
    )

    P = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.float32,
    )

    for index, (feats, mask) in enumerate(batch):
        length = feats.shape[0]

        X[
            index,
            :length,
        ] = feats

        Y[
            index,
            :length,
        ] = mask

        P[
            index,
            :length,
        ] = 1.0

    return X, Y, P


# =====================================================================
# Model
# =====================================================================

def build_model(
    cfg,
) -> torch.nn.Module:
    model_name = cfg["model"].get(
        "name",
        "crnn",
    )

    if model_name != "crnn":
        raise ValueError(
            f"This training script currently supports model=crnn, "
            f"got model={model_name!r}."
        )

    model_cfg = cfg["model"].get(
        "crnn",
        {},
    )

    return SepConvGRUVAD(
        n_mels=cfg["features"]["n_mels"],
        **model_cfg,
    )


# =====================================================================
# Loss
# =====================================================================

def masked_bce(
    logits,
    targets,
    pad,
    pos_weight,
):
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )

    denominator = (
        pad.sum()
        .clamp_min(1.0)
    )

    return (
        (loss * pad).sum()
        / denominator
    )


def estimate_pos_weight(
    dataset: FramedVAD,
    k: int = 200,
) -> float:
    """Estimate #negative / #positive from training labels."""
    positive = 0.0
    negative = 0.0

    n_items = min(
        k,
        len(dataset),
    )

    if n_items == 0:
        raise RuntimeError(
            "Training dataset is empty."
        )

    for index in range(n_items):
        _, mask = dataset[index]

        positive += float(
            mask.sum().item()
        )

        negative += float(
            (1.0 - mask).sum().item()
        )

    if positive <= 0:
        raise RuntimeError(
            "No positive speech frames found in training labels."
        )

    return max(
        negative / positive,
        1e-3,
    )


# =====================================================================
# Reproducible DataLoader workers
# =====================================================================

def seed_worker(
    worker_id: int,
):
    del worker_id

    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =====================================================================
# Validation
# =====================================================================

@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device,
):
    model.eval()

    probabilities = []
    targets = []

    for X, Y, P in loader:
        X = X.to(
            device,
            non_blocking=True,
        )

        logits = model(X)

        probs = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )

        target_np = Y.numpy()

        valid = (
            P.numpy()
            .astype(bool)
        )

        probabilities.append(
            probs[valid]
        )

        targets.append(
            target_np[valid]
        )

    if not probabilities:
        raise RuntimeError(
            "Validation loader produced no samples."
        )

    probs = np.concatenate(
        probabilities
    ).astype(
        np.float64
    )

    target = np.concatenate(
        targets
    ).astype(
        np.int64
    )

    return probs, target


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    threshold_criterion="max_f1",
):
    """Validation evaluation.

    The threshold is selected on validation only.
    """
    probs, target = collect_predictions(
        model,
        loader,
        device,
    )

    threshold = select_threshold(
        probs,
        target,
        criterion=threshold_criterion,
    )

    metrics = frame_metrics(
        probs,
        target,
        threshold,
    )

    if len(np.unique(target)) > 1:
        metrics["auroc"] = float(
            roc_auc_score(
                target,
                probs,
            )
        )

        metrics["auprc"] = float(
            average_precision_score(
                target,
                probs,
            )
        )
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    return metrics


# =====================================================================
# Training
# =====================================================================

def main(
    cfg_path,
    epochs=None,
    limit=None,
    no_wandb=False,
):
    cfg = load_config(cfg_path)

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
            f"[gpu] {torch.cuda.get_device_name(0)}"
        )

    paths = cfg["paths"]
    features_cfg = cfg["features"]
    train_cfg = cfg["train"]

    hop_seconds = (
        features_cfg["hop_ms"]
        / 1000.0
    )

    max_frames = int(
        train_cfg["seq_seconds"]
        / hop_seconds
    )

    n_epochs = (
        epochs
        if epochs is not None
        else train_cfg["max_epochs"]
    )

    # -----------------------------------------------------------------
    # Speaker-disjoint split
    # -----------------------------------------------------------------

    stems = list_utterances(
        paths["provided_data"]
    )

    if limit is not None:
        stems = stems[:limit]

    if not stems:
        raise RuntimeError(
            f"No utterances found under "
            f"{paths['provided_data']}"
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
        split_summary(split)
    )

    if len(split["train"]) == 0:
        raise RuntimeError(
            "Training split is empty."
        )

    if len(split["val"]) == 0:
        raise RuntimeError(
            "Validation split is empty."
        )

    # -----------------------------------------------------------------
    # Features
    # -----------------------------------------------------------------

    feature_fn = make_feature_fn(
        features_cfg
    )

    # -----------------------------------------------------------------
    # Training augmentation
    # -----------------------------------------------------------------

    augment_fn = None

    try:
        noise_bank = NoiseBank(
            paths["musan"],
            split="train",
            seed=cfg["seed"],
        )

        rir_bank = RIRBank(
            paths["rirs"],
            split="train",
            seed=cfg["seed"] + 1,
        )

        augment_fn = make_augment_fn(
            cfg["augment"],
            noise_bank,
            rir_bank,
            cfg["seed"],
        )

        print(
            "augmentation: MUSAN + RIR enabled"
        )

        print(
            f"[augmentation] train MUSAN files: "
            f"{len(noise_bank.files)}"
        )

        print(
            f"[augmentation] train RIR files: "
            f"{len(rir_bank.files)}"
        )

    except Exception as exc:
        print(
            "augmentation: DISABLED; "
            f"training on clean audio -> {exc}"
        )

    # -----------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------

    train_base = VADDataset(
        split["train"],
        paths["provided_data"],
        hop_seconds,
        cfg["labels"]["merge_gaps_below_ms"]
        / 1000.0,
        feature_fn=feature_fn,
        augment_fn=augment_fn,
    )

    val_base = VADDataset(
        split["val"],
        paths["provided_data"],
        hop_seconds,
        cfg["labels"]["merge_gaps_below_ms"]
        / 1000.0,
        feature_fn=feature_fn,
        augment_fn=None,
    )

    train_dataset = FramedVAD(
        train_base,
        max_frames=max_frames,
        spec_aug=cfg["augment"].get(
            "specaugment",
            False,
        ),
    )

    val_dataset = FramedVAD(
        val_base,
        max_frames=None,
        spec_aug=False,
    )

    # -----------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------

    num_workers = int(
        cfg.get(
            "num_workers",
            2,
        )
    )

    pin_memory = (
        device.type == "cuda"
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(
        cfg["seed"]
    )

    val_generator = torch.Generator()
    val_generator.manual_seed(
        cfg["seed"] + 1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=(
            num_workers > 0
        ),
        worker_init_fn=seed_worker,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=(
            num_workers > 0
        ),
        worker_init_fn=seed_worker,
        generator=val_generator,
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    model = build_model(
        cfg
    ).to(device)

    print(
        f"[params] "
        f"{model.num_parameters():,}"
    )

    pos_weight_value = estimate_pos_weight(
        train_dataset
    )

    pos_weight = torch.tensor(
        pos_weight_value,
        dtype=torch.float32,
        device=device,
    )

    print(
        f"[pos_weight] "
        f"{pos_weight_value:.4f}"
    )

    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            train_cfg["lr"]
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )

    amp_enabled = (
        device.type == "cuda"
        and bool(
            train_cfg.get(
                "fp16",
                False,
            )
        )
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled
    )

    print(
        f"[mixed precision] {amp_enabled}"
    )

    # -----------------------------------------------------------------
    # WandB
    # -----------------------------------------------------------------

    use_wandb = (
        cfg.get(
            "wandb",
            {},
        ).get(
            "enabled",
            False,
        )
        and not no_wandb
    )

    wandb = None

    if use_wandb:
        try:
            import wandb as wandb_module

            wandb = wandb_module

            wandb.init(
                entity=cfg["wandb"].get(
                    "entity"
                ),
                project=cfg["wandb"].get(
                    "project",
                    "NN-VAD",
                ),
                name=cfg["wandb"].get(
                    "name"
                ),
                mode=cfg["wandb"].get(
                    "mode",
                    "online",
                ),
                config=cfg,
            )

        except Exception as exc:
            print(
                f"[wandb] disabled -> {exc}"
            )
            use_wandb = False
            wandb = None

    # -----------------------------------------------------------------
    # Checkpoint directory
    # -----------------------------------------------------------------

    checkpoint_dir = Path(
        paths["checkpoints"]
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_f1 = -1.0
    best_threshold = 0.5

    patience = int(
        train_cfg[
            "early_stopping"
        ][
            "patience"
        ]
    )

    bad_epochs = 0
    global_step = 0

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------

    for epoch in range(
        1,
        n_epochs + 1,
    ):
        model.train()

        epoch_loss_sum = 0.0
        epoch_batches = 0

        for X, Y, P in train_loader:
            X = X.to(
                device,
                non_blocking=True,
            )

            Y = Y.to(
                device,
                non_blocking=True,
            )

            P = P.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=amp_enabled
            ):
                logits = model(X)

                loss = masked_bce(
                    logits,
                    Y,
                    P,
                    pos_weight,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            loss_value = float(
                loss.detach().cpu().item()
            )

            epoch_loss_sum += loss_value
            epoch_batches += 1

            global_step += 1

            if use_wandb:
                wandb.log(
                    {
                        "train/loss_step":
                            loss_value,
                        "step":
                            global_step,
                    }
                )

        if epoch_batches == 0:
            raise RuntimeError(
                "No training batch was produced. "
                "Try reducing batch_size or check the dataset."
            )

        train_loss = (
            epoch_loss_sum
            / epoch_batches
        )

        # -------------------------------------------------------------
        # Validation and threshold selection
        # -------------------------------------------------------------

        val = evaluate(
            model,
            val_loader,
            device,
            threshold_criterion=cfg[
                "eval"
            ].get(
                "threshold_selection",
                "max_f1",
            ),
        )

        scheduler.step(
            val["f1"]
        )

        learning_rate = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            f"epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} | "
            f"val F1 {val['f1']:.4f} | "
            f"P {val['precision']:.4f} | "
            f"R {val['recall']:.4f} | "
            f"AUROC {val['auroc']:.4f} | "
            f"AUPRC {val['auprc']:.4f} | "
            f"thr {val['threshold']:.4f} | "
            f"lr {learning_rate:.2e}"
        )

        if use_wandb:
            wandb.log(
                {
                    "train/loss_epoch":
                        train_loss,
                    "val/f1":
                        val["f1"],
                    "val/precision":
                        val["precision"],
                    "val/recall":
                        val["recall"],
                    "val/accuracy":
                        val["accuracy"],
                    "val/auroc":
                        val["auroc"],
                    "val/auprc":
                        val["auprc"],
                    "val/false_alarm_rate":
                        val["false_alarm_rate"],
                    "val/miss_rate":
                        val["miss_rate"],
                    "val/threshold":
                        val["threshold"],
                    "lr":
                        learning_rate,
                    "epoch":
                        epoch,
                }
            )

        # -------------------------------------------------------------
        # Best checkpoint
        # -------------------------------------------------------------

        if val["f1"] > best_f1:
            best_f1 = float(
                val["f1"]
            )

            best_threshold = float(
                val["threshold"]
            )

            bad_epochs = 0

            checkpoint = {
                "model":
                    model.state_dict(),
                "cfg":
                    cfg,
                "epoch":
                    epoch,
                "val_f1":
                    best_f1,
                "threshold":
                    best_threshold,
                "val_metrics":
                    val,
            }

            torch.save(
                checkpoint,
                checkpoint_dir
                / "best.pt",
            )

            print(
                f"[checkpoint] new best -> "
                f"{checkpoint_dir / 'best.pt'}"
            )

        else:
            bad_epochs += 1

            if bad_epochs >= patience:
                print(
                    f"early stopping at epoch "
                    f"{epoch}; "
                    f"best val F1="
                    f"{best_f1:.4f}"
                )
                break

    print(
        f"done. best val F1="
        f"{best_f1:.4f} | "
        f"threshold="
        f"{best_threshold:.4f} | "
        f"checkpoint="
        f"{checkpoint_dir / 'best.pt'}"
    )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of utterances for smoke testing.",
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
    )

    args = parser.parse_args()

    main(
        args.config,
        args.epochs,
        args.limit,
        args.no_wandb,
    )
