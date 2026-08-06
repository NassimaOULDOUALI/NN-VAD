# Neural Voice Activity Detection — Sonos SVC AudioML Challenge

Private repository. **Do not make public.** Per the challenge terms, neither the
provided data nor the challenge itself is redistributed; only public datasets
(MUSAN, LibriSpeech, RIRS) are downloaded by the setup script.

## Goal
A lightweight, **causal / streaming** neural VAD in PyTorch, robust to noise and
reverberation (far-field target), evaluated as a detection system (frame + segment
metrics, ROC/DET, robustness vs SNR), with standard post-processing.

## Layout
```
configs/        config.yaml — all paths & hyperparameters (no hardcoded paths)
scripts/        download_external_data.sh — public noise/reverb datasets only
src/data/       dataset, frame-mask labelling, mixing (noise+reverb), features, splits
src/models/     crnn.py (causal CRNN), tcn.py (causal TCN)
src/train.py    training loop (seeds, early stopping on val F1/AUROC)
src/evaluate.py frame + segment metrics, ROC/DET, robustness, baselines
src/postprocess.py  smoothing, min-duration, merge, hangover
src/baselines.py    energy (webrtc/auditok) + Silero (torch.hub) references
src/metrics.py  frame metrics, DET, segment detection error rate
src/viz.py      waveform/spectrogram/mask overlays, curves
report/         report.tex (the deliverable) + figures/
notebooks/      01_eda.ipynb
```

## Data (not versioned)
Place the provided corpus at `data/vad_data/` (excluded by `.gitignore`).
Public datasets: `bash scripts/download_external_data.sh`.

## Run
```bash
pip install -r requirements.txt
python -m src.train    --config configs/config.yaml
python -m src.evaluate --config configs/config.yaml
```

## Reproducibility
All seeds set to 42. No hardcoded paths (see `configs/config.yaml`).

## Note on AI code assistant
An AI code assistant was used for [scope]; reviewed and adapted by the author.
Disclosed in the report per the challenge instructions.
