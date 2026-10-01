"""
Histogram of detected stars vs. coadd (Y6 Gold) magnitude, normalized to peak 1, with the
predicted completeness curve overlaid, one panel per exposure (rows = band, columns =
exposure-time/processing class). Used to check by eye that the faint-end roll-off of the
detections matches the roll-off of the predicted completeness.

Detected stars: isolated Y6 Gold point sources on the exposure's CCDs matched to a
clean-flag skim detection (scripts/calibrate_snr_threshold.py definitions).
Prediction: data/delve.exposures.completeness.fits (noise-based model). Only exposures
not in the S/N-threshold calibration sample are drawn.

Note the histogram is completeness times the star counts, so where counts still rise
the histogram peaks near, not at, the completeness knee; past the peak both should fall
together.
"""

import os
import sys
from multiprocessing import Pool

import fitsio
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "scripts"))
import calibrate_snr_threshold as cal  # noqa: E402

COMPLETENESS = os.path.join(REPO, "data", "delve.exposures.completeness.fits")
BINS = np.arange(18.0, 26.01, 0.1)
HIST_COL, MODEL_COL = "#52514e", "#2a78d6"

# columns: (label, tags allowed, exptime lo, hi); first tag that has an exposure wins
COLUMNS = [("short", ["DES", "DELVE", "DECADE_F"], 25, 50),
           ("90 s DES", ["DES"], 89, 91),
           ("90 s DELVE", ["DELVE", "DECADE_F"], 89, 91),
           ("150-250 s", ["DELVE", "DES", "DECADE_F"], 140, 250),
           (">250 s", ["DES", "DELVE", "DECADE_F"], 251, 1000)]


def detected_mags(args):
    row, cc = args
    e, b = int(row["expnum"]), str(row["band"])
    mag, _, _, ra, dec = cal.reference_stars(e, b, cc)
    det = cal.detected(e, b, ra, dec)
    return e, np.histogram(mag[det], BINS)[0]


def main():
    rng = np.random.default_rng(5)
    depth = fitsio.read(cal.DEPTH)
    model = fitsio.read(COMPLETENESS)
    excluded = np.load(cal.DIAG)["rows"]["expnum"]
    ok = (np.isfinite(depth["m1"]) & (depth["y6_frac"] == 1) & (depth["n_ccd"] >= 50)
          & (depth["n_zp"] >= 50) & ~np.isin(depth["expnum"], excluded)
          & np.isin(depth["expnum"], model["expnum"]))

    grid = {}
    for b in "griz":
        for j, (_, tags, lo, hi) in enumerate(COLUMNS):
            for tag in tags:
                idx = np.where(ok & (depth["band"] == b) & (depth["tag"] == tag)
                               & (depth["exptime"] >= lo) & (depth["exptime"] <= hi))[0]
                if len(idx):
                    grid[b, j] = int(depth["expnum"][rng.choice(idx)])
                    break

    drow = {int(r["expnum"]): r for r in depth}
    mrow = {int(r["expnum"]): r for r in model}
    chosen = list(grid.values())
    corners = fitsio.read(cal.CORNERS, ext=1, columns=["expnum", "ccdnum", "ra", "dec"])
    corners = corners[np.isin(corners["expnum"], chosen)]
    jobs = [(drow[e], corners[corners["expnum"] == e]) for e in chosen]
    with Pool(min(len(jobs), 32)) as pool:
        hists = dict(pool.map(detected_mags, jobs))

    mm = np.linspace(BINS[0], BINS[-1], 400)
    fig, axes = plt.subplots(4, len(COLUMNS), figsize=(3.8 * len(COLUMNS), 3.0 * 4),
                             sharey=True, squeeze=False)
    for i, b in enumerate("griz"):
        for j, (label, *_) in enumerate(COLUMNS):
            ax = axes[i, j]
            if (b, j) not in grid:
                ax.text(0.5, 0.5, "no hold-out exposure", ha="center", transform=ax.transAxes,
                        color="#898781")
                continue
            e = grid[b, j]
            r, m = drow[e], mrow[e]
            h = hists[e]
            ax.stairs(h / h.max(), BINS, fill=True, color="#e1e0d9")
            ax.stairs(h / h.max(), BINS, color=HIST_COL, lw=1, label="detected stars (peak = 1)")
            ax.plot(mm, cal.logistic(mm, m["m50"], m["k"], m["c"]), color=MODEL_COL, lw=2,
                    label="predicted completeness")
            ax.axvline(m["m50"], color=MODEL_COL, lw=0.8, ls=":")
            ax.set_title(f"{e} {b} {r['exptime']:.0f}s {r['tag']}", fontsize=9)
            ax.text(0.03, 0.05, f"m50 {m['m50']:.2f}\nk {m['k']:.1f}  c {m['c']:.2f}\n"
                    f"N det {h.sum()}", transform=ax.transAxes, fontsize=7.5, color="#52514e")
            ax.set_xlim(m["m50"] - 4.5, m["m50"] + 1.5)
            ax.set_ylim(0, 1.08)
            ax.grid(color="#e1e0d9", lw=0.5)
            if i == 3:
                ax.set_xlabel("Y6 Gold PSF_MAG_APER_8 [mag]")
        axes[i, 0].set_ylabel(f"{b} band")
    for j, (label, *_) in enumerate(COLUMNS):
        axes[0, j].annotate(label, (0.5, 1.22), xycoords="axes fraction", ha="center",
                            fontsize=11, weight="bold")
    axes[0, 0].legend(loc="upper left", fontsize=7, frameon=False)
    fig.suptitle("Detected-star histogram vs. coadd magnitude (normalized to peak) "
                 "and predicted completeness, hold-out exposures", y=0.995)
    fig.tight_layout()
    out = os.path.join(HERE, "detection_hist_vs_completeness.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    for (b, j), e in sorted(grid.items()):
        print(b, COLUMNS[j][0], e, drow[e]["exptime"], drow[e]["tag"], f"m50={mrow[e]['m50']:.2f}")


if __name__ == "__main__":
    main()
