"""Training entry point for the causal VAD.

Usage:
    python -m src.train --config configs/config.yaml
    # quick smoke run (few epochs, subset, no logging):
    python -m src.train --config configs/config.yaml --epochs 2 --limit 200 --no-wandb

Assembles VADDataset + feature_fn (+ augment_fn if MUSAN/RIRS are present),
pads variable-length sequences with a validity mask, trains with weighted BCE,
early-stops on validation frame-F1, and checkpoints the best model.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .utils import load_config, set_seed
from .data.dataset import list_utterances, VADDataset
from .data.splits import speaker_disjoint_split, split_summary
from .data.features import make_feature_fn
from .data.mixing import NoiseBank, RIRBank, make_augment_fn, spec_augment
from .models.crnn import SepConvGRUVAD


# ----------------------------------------------------------------- dataset
class FramedVAD(Dataset):
    """Wraps VADDataset: returns (feats[T,M] float tensor, mask[T] float tensor).

    Long utterances are randomly cropped to `max_frames` during training;
    optional feature-domain SpecAugment is applied on training samples.
    """
    def __init__(self, base: VADDataset, max_frames: int | None = None,
                 spec_aug: bool = False, seed: int = 42):
        self.base = base
        self.max_frames = max_frames
        self.spec_aug = spec_aug
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        feats, mask = self.base[i]                       # tensor[T,M], np[T]
        mask = torch.as_tensor(mask, dtype=torch.float32)
        T = min(feats.shape[0], mask.shape[0])
        feats, mask = feats[:T], mask[:T]
        if self.max_frames and T > self.max_frames:
            s = int(torch.randint(0, T - self.max_frames, (1,), generator=self.g))
            feats, mask = feats[s:s + self.max_frames], mask[s:s + self.max_frames]
        if self.spec_aug:
            feats = spec_augment(feats, generator=self.g)
        return feats, mask


def collate(batch):
    """Pad to the batch's max length; return X[B,T,M], Y[B,T], P[B,T] (valid=1)."""
    lengths = [f.shape[0] for f, _ in batch]
    B, Tmax, M = len(batch), max(lengths), batch[0][0].shape[1]
    X = torch.zeros(B, Tmax, M)
    Y = torch.zeros(B, Tmax)
    P = torch.zeros(B, Tmax)
    for i, (f, m) in enumerate(batch):
        t = f.shape[0]
        X[i, :t], Y[i, :t], P[i, :t] = f, m, 1.0
    return X, Y, P


# ------------------------------------------------------------------- model
def build_model(cfg) -> torch.nn.Module:
    mc = cfg["model"].get("crnn", {})
    return SepConvGRUVAD(n_mels=cfg["features"]["n_mels"], **mc)


# --------------------------------------------------------------- utilities
def masked_bce(logits, targets, pad, pos_weight):
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none")
    return (loss * pad).sum() / pad.sum().clamp_min(1.0)


def estimate_pos_weight(ds: FramedVAD, k: int = 200) -> float:
    """pos_weight = #neg / #pos over a sample of the training masks (balances)."""
    pos = neg = 0.0
    for i in range(min(k, len(ds))):
        _, m = ds[i]
        pos += float(m.sum()); neg += float((1 - m).sum())
    return max(neg / max(pos, 1.0), 1e-3)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    from sklearn.metrics import f1_score, roc_auc_score
    model.eval()
    P_all, Y_all = [], []
    for X, Y, P in loader:
        logits = model(X.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        m = P.numpy().astype(bool)
        P_all.append(probs[m]); Y_all.append(Y.numpy()[m])
    probs = np.concatenate(P_all); y = np.concatenate(Y_all).astype(int)
    preds = (probs >= threshold).astype(int)
    auroc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    return {"f1": f1_score(y, preds, zero_division=0), "auroc": auroc}


# ------------------------------------------------------------------- train
def main(cfg_path, epochs=None, limit=None, no_wandb=False):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths, feat, tr = cfg["paths"], cfg["features"], cfg["train"]
    hop_s = feat["hop_ms"] / 1000.0
    max_frames = int(tr["seq_seconds"] / hop_s)
    n_epochs = epochs or tr["max_epochs"]

    # ---- splits
    stems = list_utterances(paths["provided_data"])
    if limit:
        stems = stems[:limit]
    sp = speaker_disjoint_split(stems, tuple(cfg["split"][k]
                                for k in ("train", "val", "test")), cfg["seed"])
    print(split_summary(sp))

    # ---- augmentation banks (optional; train on clean if absent)
    feature_fn = make_feature_fn(feat)
    augment_fn = None
    try:
        nb = NoiseBank(paths["musan"], "train")
        rb = RIRBank(paths["rirs"], "train")
        augment_fn = make_augment_fn(cfg["augment"], nb, rb, cfg["seed"])
        print("augmentation: MUSAN + RIR enabled")
    except Exception as e:
        print(f"augmentation: DISABLED (train on clean) -> {e}")

    train_ds = FramedVAD(
        VADDataset(sp["train"], paths["provided_data"], hop_s,
                   cfg["labels"]["merge_gaps_below_ms"] / 1000.0,
                   feature_fn=feature_fn, augment_fn=augment_fn),
        max_frames=max_frames, spec_aug=cfg["augment"].get("specaugment", False),
        seed=cfg["seed"])
    val_ds = FramedVAD(
        VADDataset(sp["val"], paths["provided_data"], hop_s,
                   cfg["labels"]["merge_gaps_below_ms"] / 1000.0,
                   feature_fn=feature_fn, augment_fn=None),
        max_frames=None, spec_aug=False, seed=cfg["seed"])

    train_dl = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True,
                          collate_fn=collate, num_workers=cfg.get("num_workers", 2),
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False,
                        collate_fn=collate, num_workers=cfg.get("num_workers", 2))

    # ---- model / optim
    model = build_model(cfg).to(device)
    print(f"[params] {model.num_parameters():,}")
    pos_weight = torch.tensor(estimate_pos_weight(train_ds), device=device)
    print(f"[pos_weight] {float(pos_weight):.3f}")
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and tr.get("fp16")))

    # ---- wandb
    use_wandb = cfg.get("wandb", {}).get("enabled", False) and not no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            entity=cfg["wandb"].get("entity"),
            project=cfg["wandb"].get("project", "sonos-vad"),
            name=cfg["wandb"].get("name"),
            mode=cfg["wandb"].get("mode", "online"),
            config=cfg,
        )
        wandb.watch(model, log_freq=100)

    # ---- loop
    ckpt_dir = Path(paths["checkpoints"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1, patience, bad = -1.0, tr["early_stopping"]["patience"], 0
    step = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        for X, Y, P in train_dl:
            X, Y, P = X.to(device), Y.to(device), P.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss = masked_bce(model(X), Y, P, pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            step += 1
            if use_wandb:
                import wandb; wandb.log({"train/loss": float(loss), "step": step})

        val = evaluate(model, val_dl, device)
        sched.step(val["f1"])
        lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch:3d} | val F1 {val['f1']:.4f} | "
              f"val AUROC {val['auroc']:.4f} | lr {lr:.2e}")
        if use_wandb:
            import wandb
            wandb.log({"val/f1": val["f1"], "val/auroc": val["auroc"],
                       "lr": lr, "epoch": epoch})

        if val["f1"] > best_f1:
            best_f1, bad = val["f1"], 0
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "epoch": epoch, "val_f1": best_f1},
                       ckpt_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"early stopping at epoch {epoch} (best F1 {best_f1:.4f})")
                break

    print(f"done. best val F1 = {best_f1:.4f} -> {ckpt_dir/'best.pt'}")
    if use_wandb:
        import wandb; wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap #utterances (smoke)")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    main(a.config, a.epochs, a.limit, a.no_wandb)
