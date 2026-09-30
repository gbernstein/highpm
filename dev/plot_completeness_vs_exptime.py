"""Plot empirical-minus-model m50 vs exposure time from check_completeness_vs_exptime.py."""

import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {"g": "#2a78d6", "r": "#eb6834", "i": "#1baf7a", "z": "#eda100"}

a = np.load(os.path.join(HERE, "completeness_vs_exptime.npz"))["rows"]
d = a["emp_m50"] - a["model_m50"]

fig, ax = plt.subplots(figsize=(7, 4.5))
for b, col in COLORS.items():
    s = a["band"] == b
    mk = np.where(a["estimated"][s], "o", "s")
    for m in ("o", "s"):
        ss = mk == m
        ax.scatter(a["exptime"][s][ss], d[s][ss], s=28, marker=m, color=col, alpha=0.6,
                   edgecolor="none", label=f"{b}" if m == "o" else None)
ax.axhline(0, color="#b0afa8", lw=1)
ax.set_xscale("log")
ticks = [30, 40, 50, 60, 90, 150, 200, 300]  # a readable subset; points sit at true exptimes
ax.set_xticks(ticks, minor=False)
ax.set_xticklabels([str(t) for t in ticks], fontsize=9)
ax.minorticks_off()
ax.set_xlabel("EXPTIME [s]")
ax.set_ylabel(r"empirical $-$ model $m_{50}$ [mag]")
ax.set_title("Per-exposure completeness vs. injection model (PMSculptor 8°)\n"
             "reference: DES Y6 Gold stars", fontsize=10)
ax.legend(frameon=False, fontsize=9, loc="lower left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
out = os.path.join(HERE, "completeness_vs_exptime.png")
fig.savefig(out, dpi=150)
print("wrote", out)
