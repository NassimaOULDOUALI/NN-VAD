# Neural Voice Activity Detection 

> Challenge-provided data are not versioned or redistributed. Only public external resources used by the pipeline (e.g. MUSAN / RIR resources) may be downloaded by the setup scripts.

## Overview

This repository implements a lightweight **causal neural Voice Activity Detector (VAD)** in PyTorch for low-complexity, low-look-ahead deployment.

The final system is a **47,873-parameter causal convolutional-recurrent network (CRNN)** operating on 64-bin log-mel features at a 10 ms frame rate. It is evaluated as a detection system rather than only as a frame classifier: thresholds are calibrated on validation data, temporal post-processing is selected on validation only, and the test analysis includes false-alarm / miss trade-offs, ROC/PR/DET curves, bootstrap uncertainty, additive-noise robustness, empirical causality checks, and CPU runtime.

### Key properties

- **47,873 trainable parameters** (~0.191 MB in FP32)
- 16 kHz mono input
- 64-bin log-mel front-end
- 25 ms analysis window, 10 ms hop
- causal cumulative mean/variance normalization
- 4 causal depthwise-separable Conv1D residual blocks
- dilations `{1, 2, 4, 8}`, kernel size 5
- unidirectional GRU, hidden size 64
- ~600 ms explicit convolutional left context, plus recurrent past context
- no future context in the neural network
- **12.5 ms front-end look-ahead** from centered 25 ms framing
- empirical prefix-invariance discrepancy `< 3e-8`
- speaker-disjoint train / validation / test split
- validation-only threshold and post-processing selection
- single-thread CPU end-to-end RTF **0.00348** on 60 s audio

---

## Repository layout

```text
configs/
    config.yaml
    config_full.yaml
    config_smoke.yaml

scripts/
    download_external_data.sh
    benchmark_cpu.py
    eval_silero_fair.py
    plot_training_curves.py
    smoke_a100.slurm
    train_full_a100.slurm
    make_report_figures.py

src/
    train.py
    evaluate.py
    metrics.py
    postprocess.py
    baselines.py
    utils.py
    viz.py

    data/
        dataset.py
        eda.py
        features.py
        mixing.py
        splits.py

    models/
        crnn.py
        tcn.py

tests/
    test_dataset.py
    test_features.py
    test_mixing.py
    test_model.py
    test_shape.py
    test_causality_empirical.py

report/
    figures/
    results/

requirements.txt
README.md
LICENSE
```

The challenge corpus, external datasets, logs, and checkpoints are intentionally excluded from version control.

---

## Data

### Provided corpus

The provided corpus contains paired `.wav` / `.json` files with labelled speech intervals.

| Statistic | Value |
|---|---:|
| Utterances | 957 |
| Speakers | 34 |
| Total duration | 3.28 h |
| Sampling rate | 16 kHz |
| Channels | mono |
| Mean utterance duration | 12.3 s |
| Mean speech ratio | 0.81 |

### Speaker-disjoint split

Whole speakers are assigned to exactly one partition.

| Split | Utterances | Speakers |
|---|---:|---:|
| Train | 770 | 26 |
| Validation | 94 | 4 |
| Test | 93 | 4 |

The split is deterministic with seed `42`.

Test speakers are never used for:

- model training;
- checkpoint selection;
- threshold calibration;
- post-processing selection.

### Labels

Reference speech intervals are sorted and overlapping intervals are merged. Reference gaps of at most **50 ms** are merged before rasterization onto the 10 ms frame grid.

A frame is labelled as speech when its center lies inside a merged reference interval.

---

## External data and augmentation

The final reported model uses **MUSAN music/noise** for additive augmentation.

MUSAN speech is deliberately excluded because adding competing speech while preserving only the source utterance mask would create ambiguous binary targets.

Final augmentation configuration:

```yaml
augment:
  p_noise: 0.8
  snr_db_range: [-5, 20]
  p_reverb: 0.0
  specaugment: true
```

A deterministic split separates MUSAN recordings used for training augmentation from the **239 files reserved for robustness evaluation**.

Noise scaling uses whole-utterance / global RMS.

### Reverberation

RIR support exists in the codebase, but **RIR augmentation is disabled in the final reported model**. The initial direct time-domain convolution created a CPU input-pipeline bottleneck and was not retained without adequate validation.

Therefore, the final results demonstrate robustness to held-out **additive noise**, not validated far-field reverberation robustness.

---

## Model

### Acoustic front-end

```text
waveform
  ↓
64-bin log-mel
25 ms window / 10 ms hop
  ↓
causal cumulative mean/variance normalization
```

The neural network itself is causal. The centered 25 ms analysis window contributes **12.5 ms of algorithmic look-ahead**.

### CRNN architecture

```text
64-d log-mel features
        ↓
1 × 1 temporal projection
64 channels
        ↓
4 causal depthwise-separable Conv1D residual blocks
kernel size = 5
dilations = {1, 2, 4, 8}
        ↓
unidirectional GRU
hidden size = 64
        ↓
linear frame classifier
        ↓
speech posterior
```

Each convolutional residual block contains:

- causal left-padded depthwise convolution;
- pointwise `1 × 1` convolution;
- batch normalization;
- ReLU;
- dropout;
- residual connection.

The dilated stack provides

```text
(5 - 1) × (1 + 2 + 4 + 8) = 60 frames
```

of explicit left context, i.e. approximately **600 ms** at a 10 ms hop. The unidirectional GRU can integrate longer past context.

### Model size

```text
Trainable parameters : 47,873
FP32 weights         : ~0.191 MB
```

---

## Causality verification

Neural-network causality is tested empirically using prefix invariance.

For several cut points, inference on the prefix `x[:t]` is compared with the corresponding prefix of full-sequence inference.

Tested cut points:

```text
1, 5, 10, 25, 50, 100, 150 frames
```

Maximum observed discrepancy:

```text
< 3 × 10^-8
```

This is consistent with floating-point noise and confirms that the network output at frame `t` does not depend on future feature frames.

The only future context in the reported pipeline is the **12.5 ms** look-ahead introduced by centered acoustic framing.

---

## Training

Final training configuration:

```yaml
seed: 42

features:
  n_mels: 64
  win_ms: 25
  hop_ms: 10
  center: true
  cmvn: causal

labels:
  merge_gaps_below_ms: 50

model:
  channels: 64
  kernel_size: 5
  dilations: [1, 2, 4, 8]
  gru_hidden: 64
  dropout: 0.1

train:
  loss: weighted_bce
  optimizer: adamw
  lr: 1e-3
  batch_size: 32
  seq_seconds: 8
  max_epochs: 50
  early_stopping_patience: 5
  fp16: true

augment:
  p_noise: 0.8
  snr_db_range: [-5, 20]
  p_reverb: 0.0
  specaugment: true
```

Training stopped at epoch **22** through early stopping. The best checkpoint was obtained at epoch **17**, with raw validation F1:

```text
0.9720
```

The final training job took **1 min 17 s** on one NVIDIA A100 80 GB GPU. The A100 reflects available institutional compute rather than a deployment requirement.

---

## Operating-point calibration

All thresholds are selected **exclusively on validation data** and frozen before test evaluation.

| System | Validation-selected threshold |
|---|---:|
| CRNN | `0.246333` |
| Silero, max-F1 | `0.456269` |
| Silero, validation FA-matched | `0.822688` |
| Energy, log-RMS | `-6.086459` |

The validation FA target used for the matched Silero comparison is:

```text
0.0757
```

---

## Temporal post-processing

The final post-processing chain is selected on validation data only:

```yaml
postprocess:
  smooth_median_ms: 90
  min_speech_ms: 120
  min_silence_ms: 100
  hangover_ms: 0
```

### Validation ablation

| Stage | Precision | Recall | F1 | FA | Miss |
|---|---:|---:|---:|---:|---:|
| Raw threshold | 0.9789 | 0.9652 | 0.9720 | 0.0825 | 0.0348 |
| + median | 0.9806 | 0.9667 | 0.9736 | 0.0756 | 0.0333 |
| + min speech | 0.9814 | 0.9667 | 0.9740 | 0.0724 | 0.0333 |
| + merge silence | 0.9806 | 0.9691 | **0.9748** | 0.0757 | **0.0309** |

Hangover is disabled because it is a no-op on top of the selected short-silence merge.

---

# Results

## Clean held-out test set

All operating thresholds below are frozen from validation.

| System | Precision | Recall | F1 | False alarm | Miss |
|---|---:|---:|---:|---:|---:|
| Energy, calibrated | 0.9449 | 0.9410 | 0.9429 | 0.2945 | 0.0590 |
| Silero, calibrated | 0.9765 | **0.9760** | 0.9762 | 0.1262 | **0.0240** |
| CRNN raw | 0.9860 | 0.9655 | 0.9757 | 0.0734 | 0.0345 |
| **CRNN + post** | **0.9870** | 0.9699 | **0.9784** | **0.0686** | 0.0301 |

The final CRNN and calibrated Silero are close in aggregate clean F1:

```text
CRNN + post : 0.9784
Silero      : 0.9762
ΔF1         : +0.0022
```

The more deployment-relevant distinction is their operating point:

```text
False alarm
CRNN + post : 0.0686
Silero      : 0.1262
```

At their respective validation-calibrated max-F1 thresholds, the CRNN therefore operates at roughly half Silero's clean-test false-alarm rate, at the cost of a moderately higher miss rate.

### Segment-level detection error

Using a 100 ms collar around reference boundaries:

```text
CRNN raw      : 1.15 % DER
CRNN + post   : 0.64 % DER
Energy        : 7.07 % DER
```

Here `DER` denotes a **speech-detection error rate**, not diarization DER.

---

## Continuous-score metrics

| System | AUROC | AUPRC |
|---|---:|---:|
| Energy | 0.8286 | 0.9391 |
| Silero | 0.9818 | 0.9961 |
| **CRNN** | **0.9839** | **0.9965** |

ROC, precision-recall and DET curves are generated from continuous frame scores and stored under `report/figures/`.

---

## Validation-matched false-alarm comparison

To reduce the operating-point confound, a second Silero threshold is selected on validation to match the final CRNN validation false-alarm rate.

The frozen thresholds are then transferred to test without using test labels for matching.

| System | Precision | Recall | F1 | FA | Miss |
|---|---:|---:|---:|---:|---:|
| **CRNN + post** | **0.9870** | **0.9699** | **0.9784** | **0.0686** | **0.0301** |
| Silero, val. FA-matched | 0.9832 | 0.9644 | 0.9737 | 0.0884 | 0.0356 |

Under this protocol, the final CRNN obtains both a lower test false-alarm rate and a lower miss rate.

---

## Bootstrap uncertainty

A paired utterance-level bootstrap with **10,000 resamples** is used on the clean test partition.

95% percentile intervals:

```text
F1 CRNN   : [0.9764, 0.9803]
F1 Silero : [0.9740, 0.9784]

ΔF1 = F1(CRNN) - F1(Silero)
95% CI : [0.0009, 0.0035]
median : 0.0022
```

This interval quantifies uncertainty under **utterance-level resampling within this test partition**.

It should not be interpreted as a general claim of population-level superiority over Silero because the test set contains only four speakers; speaker-level uncertainty is not adequately estimated by this bootstrap.

---

## Additive-noise robustness

The 93 held-out test utterances are mixed with recordings from a **239-file MUSAN evaluation pool disjoint from the training augmentation files**.

Global-RMS SNRs:

```text
-5, 0, 5, 10, 15, 20 dB
```

### Frame F1 versus SNR

| SNR (dB) | Energy | Silero* | CRNN raw | **CRNN + post** |
|---:|---:|---:|---:|---:|
| -5 | 0.9164 | 0.9128 | 0.9260 | **0.9316** |
| 0 | 0.9173 | 0.9502 | 0.9464 | **0.9521** |
| 5 | 0.9173 | **0.9664** | 0.9571 | 0.9614 |
| 10 | 0.9180 | 0.9690 | 0.9647 | **0.9690** |
| 15 | 0.9191 | **0.9720** | 0.9681 | 0.9718 |
| 20 | 0.9239 | 0.9738 | 0.9704 | **0.9739** |

\* The Silero values in this noise sweep come from its standard inference pipeline, **not** from the validation-calibrated threshold used in the clean comparison. The noisy Silero comparison should therefore be treated as descriptive until rerun under a protocol-consistent calibration.

At the most severe tested condition:

```text
-5 dB:
CRNN + post : 0.9316 F1
Silero*     : 0.9128 F1
```

Post-processing improves the CRNN by approximately `+0.004` to `+0.006` absolute F1 across all evaluated SNRs.

The noise sweep evaluates generalization to **held-out noise recordings inside the training SNR range**; it is not an extrapolation experiment to unseen SNR values.

---

## CPU runtime

CPU throughput is measured with PyTorch restricted to one intra-op and one inter-op thread on an **Intel Xeon Gold 6248 @ 2.50 GHz**.

The end-to-end path is:

```text
waveform
  → log-mel
  → causal CMVN
  → CRNN
  → sigmoid
  → post-processing
```

| Audio | Model time | Model RTF | E2E time | E2E RTF | E2E p95 |
|---:|---:|---:|---:|---:|---:|
| 1 s | 4.32 ms | 0.00432 | 5.44 ms | 0.00544 | 5.84 ms |
| 10 s | 28.80 ms | 0.00288 | 33.45 ms | 0.00335 | 35.06 ms |
| 30 s | 83.69 ms | 0.00279 | 103.67 ms | 0.00346 | 105.98 ms |
| 60 s | 166.26 ms | 0.00277 | 208.87 ms | **0.00348** | 214.20 ms |

For 60 s of audio:

```text
E2E RTF ≈ 0.0035
≈ 287× real time
```

These are complete-sequence throughput measurements on a server-class CPU. They are **not** target-device measurements and should not be interpreted as first-frame or persistent-state streaming latency.

---

## Reproducibility

### Environment

Final reported environment:

```text
PyTorch 2.8.0+cu128
seed = 42
```

Seeds are fixed for:

- PyTorch;
- NumPy;
- Python `random`;
- cuDNN-related configuration used by the training pipeline.

All data paths and hyperparameters are configuration-driven; there are no hardcoded corpus paths in the training/evaluation code.

### Install

```bash
pip install -r requirements.txt
```

### Data

Place the challenge corpus at:

```text
data/vad_data/
```

This directory is excluded by `.gitignore`.

Download public external resources with:

```bash
bash scripts/download_external_data.sh
```

### Train the full model

```bash
python -u -m src.train --config configs/config_full.yaml
```

The final report used:

```text
checkpoints/crnn_full_job834609_best.pt
```

Checkpoints are not versioned in this repository. The exported final checkpoint and Hugging Face inference package are available here:

https://huggingface.co/nassimaODL/streaming-vad-crnn

A fresh training run may produce a different checkpoint filename.

### Evaluate

```bash
python -m src.evaluate \
  --config configs/config_full.yaml \
  --checkpoint checkpoints/crnn_full_job834609_best.pt
```

### Fair Silero comparison

```bash
PYTHONPATH=. python scripts/eval_silero_fair.py
```

### Smoke configuration

For a quick pipeline check:

```bash
python -u -m src.train --config configs/config_smoke.yaml
```

---

## Result artifacts

Main machine-readable outputs:

```text
report/results/main_results.csv
report/results/postprocess_ablation.csv
report/results/validation_postprocess_ablation.csv
report/results/snr_robustness.csv
report/results/silero_fair_comparison.csv
report/results/silero_fair_bootstrap.npz
```

Main figures include:

```text
report/figures/training_curves.png
report/figures/roc.png
report/figures/roc_all_systems.png
report/figures/precision_recall.png
report/figures/precision_recall_all_systems.png
report/figures/det.png
report/figures/det_all_systems.png
report/figures/f1_vs_snr.png
report/figures/bootstrap_delta_f1.png
```

Report-only figures are regenerated deterministically from result files where applicable.

---

## Tests

The repository includes tests for:

- dataset construction;
- feature extraction;
- augmentation / mixing;
- model shape and forward behavior;
- end-to-end shape consistency;
- empirical causality / prefix invariance.

Run with:

```bash
pytest -q
```

---

## Limitations

The final reported system has several deliberately documented limitations.

1. **Only four held-out test speakers.**  
   The utterance-level bootstrap does not estimate speaker-population uncertainty.

2. **Single training seed.**  
   Run-to-run optimization variability has not been measured.

3. **Reverberation remains untested in the final model.**  
   This is the largest remaining gap relative to the far-field deployment target.

4. **Noise-sweep Silero protocol is not fully matched.**  
   The noisy Silero values use its standard inference pipeline rather than the clean-test validation-calibrated threshold.

5. **Noise robustness currently reports F1 only.**  
   Per-SNR false-alarm and miss rates are needed to expose operating-point drift.

6. **No confidence interval is reported at -5 dB.**

7. **BatchNorm / padded-frame interaction remains to be audited.**

8. **Centered acoustic framing introduces 12.5 ms look-ahead.**

9. **Silero's 32 ms chunk grid is mapped onto the 10 ms reference grid**, which may introduce small boundary quantization effects.

10. **Synthetic additive noise is not a complete far-field simulation.**  
    Room acoustics, array effects, echo-cancellation residuals, competing speakers, Lombard speech, and related conditions are not represented.

11. **No persistent-state chunked streaming benchmark yet.**  
    Neural causality is verified, but chunk-by-chunk stateful GRU inference has not yet been benchmarked on target-class hardware.

---

## Prioritized next experiments

In order of scientific leverage:

1. Re-run Silero over the noise sweep using the validation-calibrated protocol, report FA/miss at each SNR, and add a paired bootstrap at `-5 dB`.
2. Add efficient RIR augmentation/evaluation and report clean / noise / reverb / noise+reverb conditions.
3. Report per-speaker F1 / FA / miss for the four held-out test speakers.
4. Audit BatchNorm statistics with padded training batches.
5. Implement persistent-state chunked streaming, verify equivalence with full-sequence inference, then benchmark first-decision latency and evaluate ONNX / INT8 deployment.

---

## AI-assisted development

ChatGPT was used as a coding assistant for debugging, experimental scripting and refactoring.

AI-generated suggestions were reviewed, tested, and adapted before inclusion. Experimental execution, configuration choices, output inspection, result interpretation, and all final reported values were performed and checked by the author.

---

## Summary

The final system is a **47,873-parameter causal CRNN VAD** with validation-only operating-point selection and a small deployment footprint.

On the held-out clean test partition it reaches:

```text
F1        : 0.9784
Precision : 0.9870
Recall    : 0.9699
FA        : 0.0686
Miss      : 0.0301
AUROC     : 0.9839
AUPRC     : 0.9965
```

Compared with validation-calibrated Silero, aggregate clean F1 is close (`0.9784` vs. `0.9762`), while the final CRNN operates at substantially lower false-alarm rate (`0.0686` vs. `0.1262`).

Under validation-FA matching, the CRNN attains both lower test FA and lower miss than Silero. It also remains competitive across held-out MUSAN noise conditions down to `-5 dB`, where the final CRNN reaches `0.9316` F1 under the current noise-sweep protocol.

The complete waveform-to-decision pipeline reaches **RTF 0.00348** on 60 s inputs using one CPU thread. The principal remaining evidence required for the stated far-field target is a validated reverberation experiment and protocol-consistent noisy-baseline comparison.
