import re
from pathlib import Path

import matplotlib.pyplot as plt


log_path = Path("logs/full_834609.out")
out_path = Path("report/figures/training_curves.png")
out_path.parent.mkdir(parents=True, exist_ok=True)

pattern = re.compile(
    r"epoch\s+(\d+)\s+\|\s+"
    r"train loss\s+([0-9.eE+-]+)\s+\|\s+"
    r"val F1\s+([0-9.eE+-]+)\s+\|\s+"
    r"P\s+([0-9.eE+-]+)\s+\|\s+"
    r"R\s+([0-9.eE+-]+)\s+\|\s+"
    r"AUROC\s+([0-9.eE+-]+)\s+\|\s+"
    r"AUPRC\s+([0-9.eE+-]+)\s+\|\s+"
    r"thr\s+([0-9.eE+-]+)\s+\|\s+"
    r"lr\s+([0-9.eE+-]+)"
)

epochs = []
train_loss = []
val_f1 = []
auroc = []
auprc = []

for line in log_path.read_text().splitlines():
    m = pattern.search(line)
    if not m:
        continue

    epochs.append(int(m.group(1)))
    train_loss.append(float(m.group(2)))
    val_f1.append(float(m.group(3)))
    auroc.append(float(m.group(6)))
    auprc.append(float(m.group(7)))

fig, ax1 = plt.subplots(figsize=(8, 4.8))

ax1.plot(epochs, train_loss, label="Train loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training loss")

ax2 = ax1.twinx()
ax2.plot(epochs, val_f1, label="Validation F1")
ax2.plot(epochs, auroc, label="Validation AUROC")
ax2.plot(epochs, auprc, label="Validation AUPRC")
ax2.set_ylabel("Validation metric")

ax1.axvline(17, linestyle="--", linewidth=1, label="Best checkpoint")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight")

print(out_path)
