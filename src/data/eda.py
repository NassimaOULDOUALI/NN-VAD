"""Exploratory data analysis of the provided VAD corpus.

Generates the report's data figures + a LaTeX stats table.
Run:
    python -m src.data.eda --root data/vad_data --out report/figures
Outputs:
    report/figures/data_stats.png     (4-panel corpus characteristics)
    report/figures/mask_example.png   (waveform + spectrogram + speech mask)
    report/figures/stats_table.tex    (booktabs table for the report)
"""
from __future__ import annotations
import argparse, json, wave, contextlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft

MERGE_GAP_S = 0.050
ACCENT = "#00508C"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
})


def _read_wav(path):
    with contextlib.closing(wave.open(str(path), "r")) as f:
        sr, n = f.getframerate(), f.getnframes()
        x = np.frombuffer(f.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def _merge(segs, gap_s=0.0):
    segs = sorted((float(s["start_time"]), float(s["end_time"])) for s in segs)
    out = []
    for a, b in segs:
        if out and a <= out[-1][1] + gap_s:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def collect(root):
    root = Path(root)
    durs, ratios, nsegs, gaps, spk = [], [], [], {}, {}
    for j in sorted(root.glob("*.json")):
        stem = j.stem
        with contextlib.closing(wave.open(str(root / f"{stem}.wav"), "r")) as f:
            dur = f.getnframes() / f.getframerate()
        merged = _merge(json.load(open(j))["speech_segments"], MERGE_GAP_S)
        sp = sum(b - a for a, b in merged)
        durs.append(dur); ratios.append(min(sp / dur, 1.0)); nsegs.append(len(merged))
        for (a1, b1), (a2, b2) in zip(merged, merged[1:]):
            gaps.setdefault("g", []).append(a2 - b1)
        s = stem.split("-")[0]
        spk[s] = spk.get(s, 0) + 1
    return (np.array(durs), np.array(ratios), np.array(nsegs),
            np.array(gaps.get("g", [])), spk)


def fig_stats(durs, ratios, nsegs, spk, out):
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    ax[0, 0].hist(durs, bins=30, color=ACCENT, alpha=0.85)
    ax[0, 0].axvline(durs.mean(), color="k", ls="--", lw=1,
                     label=f"mean {durs.mean():.1f}s")
    ax[0, 0].set(title="Utterance duration", xlabel="seconds", ylabel="count")
    ax[0, 0].legend()

    ax[0, 1].hist(ratios, bins=30, color=ACCENT, alpha=0.85)
    ax[0, 1].axvline(ratios.mean(), color="k", ls="--", lw=1,
                     label=f"mean {ratios.mean():.2f}")
    ax[0, 1].set(title="Speech ratio (speech/total)", xlabel="ratio", ylabel="count")
    ax[0, 1].legend()

    ax[1, 0].hist(nsegs, bins=range(1, nsegs.max() + 2), color=ACCENT, alpha=0.85)
    ax[1, 0].set(title=f"Speech segments per utterance (merged <{int(MERGE_GAP_S*1000)}ms)",
                 xlabel="segments", ylabel="count")

    counts = sorted(spk.values(), reverse=True)
    ax[1, 1].bar(range(len(counts)), counts, color=ACCENT, alpha=0.85)
    ax[1, 1].set(title=f"Utterances per speaker (n={len(spk)})",
                 xlabel="speaker (sorted)", ylabel="utterances")

    fig.suptitle("Provided corpus — characteristics", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_mask_example(root, out):
    root = Path(root)
    # pick a representative utterance: moderate ratio, a few segments
    best = None
    for j in sorted(root.glob("*.json")):
        merged = _merge(json.load(open(j))["speech_segments"], MERGE_GAP_S)
        if 3 <= len(merged) <= 6:
            best = (j.stem, merged); break
    stem, merged = best
    x, sr = _read_wav(root / f"{stem}.wav")
    t = np.arange(len(x)) / sr
    f, tt, Z = stft(x, fs=sr, nperseg=400, noverlap=240)
    S = 20 * np.log10(np.abs(Z) + 1e-6)

    fig, ax = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.3]})
    ax[0].plot(t, x, color="#444", lw=0.5)
    for a, b in merged:
        ax[0].axvspan(a, b, color=ACCENT, alpha=0.20)
    ax[0].set(title=f"Waveform + ground-truth speech mask — {stem}",
              ylabel="amplitude")
    ax[0].margins(x=0)

    ax[1].pcolormesh(tt, f, S, shading="auto", cmap="magma")
    for a, b in merged:
        ax[1].axvspan(a, b, color="cyan", alpha=0.12)
    ax[1].set(title="Log-magnitude spectrogram (shaded = speech)",
              xlabel="time (s)", ylabel="Hz")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def write_table(durs, ratios, nsegs, gaps, spk, out):
    tex = r"""\begin{table}[h]\centering\small
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Statistic} & \textbf{Value} \\ \midrule
Utterances & %d \\
Sample rate / channels & 16\,kHz / mono \\
Total duration & %.2f\,h \\
Utterance duration (mean/med/min/max) & %.1f\,/\,%.1f\,/\,%.1f\,/\,%.1f\,s \\
Speakers & %d \\
Speech ratio (mean/median) & %.2f / %.2f \\
Speech segments per utt.\ (merged, mean/max) & %.1f / %d \\
Silence gap (median/p95/max) & %.2f / %.2f / %.2f\,s \\
\bottomrule
\end{tabular}
\caption{Provided corpus statistics (computed from the archive).}
\label{tab:provided}
\end{table}""" % (
        len(durs), durs.sum() / 3600, durs.mean(), np.median(durs), durs.min(), durs.max(),
        len(spk), ratios.mean(), np.median(ratios), nsegs.mean(), int(nsegs.max()),
        np.median(gaps), np.percentile(gaps, 95), gaps.max())
    Path(out).write_text(tex)


def main(root, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    durs, ratios, nsegs, gaps, spk = collect(root)
    fig_stats(durs, ratios, nsegs, spk, outdir / "data_stats.png")
    fig_mask_example(root, outdir / "mask_example.png")
    write_table(durs, ratios, nsegs, gaps, spk, outdir / "stats_table.tex")
    print(f"wrote data_stats.png, mask_example.png, stats_table.tex -> {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/vad_data")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    main(a.root, a.out)
