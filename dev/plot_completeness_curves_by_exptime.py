"""Empirical detection-fraction curves for i-band exposures of different lengths
(similar T_EFF), with the injection model's curves overlaid.

Data points: fraction of DES Y6 Gold coadd stars detected in each
exposure (same estimator as check_completeness_vs_exptime.py). Dashed lines: the
model (m50, k, c) from data/delve.exposures.completeness.fits, unshifted.
"""

import os

import fitsio
import matplotlib.pyplot as plt
import numpy as np

from check_completeness_vs_exptime import COMPLETENESS, CORNERS, logistic, measure

HERE = os.path.dirname(os.path.abspath(__file__))
EXPOSURES = [  # (expnum, exptime s, T_EFF) -- i band, T_EFF ~0.4-0.6
    (356650, 30, 0.46),
    (896548, 60, 0.42),
    (492811, 90, 0.60),   # measured completeness
    (451571, 150, 0.55),
    (1006636, 180, 0.50),  # one of the 180 s exposures that come out shallow
    (957896, 300, 0.49),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BINS = np.arange(20.0, 25.51, 0.2)

comp = fitsio.read(COMPLETENESS)
corners = fitsio.read(CORNERS, ext=1, columns=["expnum", "ccdnum", "ra", "dec"])

fig, ax = plt.subplots(figsize=(8, 5))
centers = 0.5 * (BINS[1:] + BINS[:-1])
mm = np.linspace(BINS[0], BINS[-1], 300)
for (e, t, teff), col in zip(EXPOSURES, COLORS):
    mag, det = measure(e, "i", corners[corners["expnum"] == e])
    n, _ = np.histogram(mag, BINS)
    k, _ = np.histogram(mag[det], BINS)
    ok = n >= 20
    f = k[ok] / n[ok]
    err = np.sqrt(np.clip(f * (1 - f), 1e-4, None) / n[ok])
    row = comp[comp["expnum"] == e][0]
    tag = "measured" if not row["estimated"] else "estimated"
    ax.errorbar(centers[ok], f, yerr=err, fmt="o-", ms=4, color=col, lw=1.5, capsize=0,
                label=f"{t:>3d} s  (T_EFF {teff:.2f}, model {tag})")
    ax.plot(mm, logistic(mm, row["m50"], row["k"], row["c"]),
            "--", color=col, lw=1.5)

ax.axhline(0.5, color="#b0afa8", lw=1)
ax.set_xlim(20.5, 25.5)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("DES Y6 Gold PSF_MAG_APER_8_I [mag]")
ax.set_ylabel("fraction detected in exposure")
ax.set_title(f"i-band detection completeness by exposure length\n"
             "points + solid = data (0.2 mag bins, Y6 Gold stars); dashed = model",
             fontsize=10)
ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.9))
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
out = os.path.join(HERE, "completeness_curves_by_exptime.png")
fig.savefig(out, dpi=150)
print("wrote", out)
