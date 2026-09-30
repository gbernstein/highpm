"""
Per-exposure free-fit m50 minus the S/N-threshold prediction, vs. exposure time,
from dev/snr_threshold_calibration.npz (scripts/calibrate_snr_threshold.py).
One panel per processing tag; color = band.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {"g": "#2a78d6", "r": "#eb6834", "i": "#1baf7a", "z": "#eda100"}
TICKS = [30, 45, 60, 90, 150, 200, 300]

d = np.load(os.path.join(HERE, "snr_threshold_calibration.npz"))["rows"]
tags = [t for t in dict.fromkeys(d["tag"]) if (d["tag"] == t).sum() >= 10]  # skip one-off tags
fig, axes = plt.subplots(1, len(tags), figsize=(5.5 * len(tags), 4.2), sharey=True, squeeze=False)
for ax, tag in zip(axes[0], tags):
    for b, col in COLORS.items():
        s = (d["tag"] == tag) & (d["band"] == b)
        ax.scatter(d["exptime"][s], d["emp_m50"][s] - d["pred_m50"][s], s=14, c=col, alpha=0.7, label=b)
    ax.axhline(0, color="#898781", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(TICKS)
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.minorticks_off()
    ax.set_xlabel("exposure time [s]")
    ax.set_title(f"{tag} processing")
axes[0, 0].set_ylabel("free-fit m50 - predicted m50 [mag]")
axes[0, 0].legend(loc="lower left", frameon=False)
fig.suptitle("S/N-threshold completeness model vs. DES Y6 Gold")
fig.tight_layout()
out = os.path.join(HERE, "snr_threshold_calibration.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
