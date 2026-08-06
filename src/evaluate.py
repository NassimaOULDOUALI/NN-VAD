"""Evaluation entry point.

Usage:
    python -m src.evaluate --config configs/config.yaml

Produces: main results table, ROC/DET, F1-vs-SNR, per-noise breakdown,
post-processing ablation, error cases — all figures into report/figures/.
"""
from __future__ import annotations
import argparse
from .utils import load_config, set_seed


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    # TODO:
    #  1. load best checkpoint; run on in-domain test set
    #  2. select_threshold on val; frame metrics + ROC/DET
    #  3. post-process -> segment DER; ablation table
    #  4. robustness: F1 vs SNR sweep; per-noise +/- reverb
    #  5. baselines (energy, silero) on same axes
    #  6. dump figures to cfg["paths"]["report_figs"]
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    main(ap.parse_args().config)
