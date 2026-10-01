"""
Exposures that only got a zeropoint once the DELVE catalog became the zp reference (no
bright gold stars): histogram of detected stars vs. DELVE catalog magnitude (peak = 1)
with the predicted completeness, and the detected fraction (detected / catalog). The DELVE
catalog is only somewhat deeper than one exposure, so the histogram edge can be the
catalog's; the fraction is the direct test. Rows = band, columns = sky region.

The old depth file (gold-referenced zp) is passed with --old-depth; an exposure is "newly
calibrated" when it had no zp there and has one in data/delve.exposures.depth.fits.
Reference stars, isolation and the detection match are those of
dev/plot_detection_hist_outside_des.py.
"""

import argparse
import os
import sys
from multiprocessing import Pool

import fitsio
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, HERE)
import calibrate_snr_threshold as cal  # noqa: E402
from plot_detection_hist_outside_des import (BINS, COMPLETENESS, HIST_COL, NOISE_COL,  # noqa: E402
                                             REF_COL, histograms)

EXPOSURES = os.path.expanduser("~/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5")
MCS = [(80.89, -69.76, 12.0), (13.19, -72.83, 6.0)]
COLUMNS = ["north strip (Dec > +5)", "north strip (Dec > +5)", "near LMC/SMC", "elsewhere", "elsewhere"]


def region(ra, dec):
    near_mc = np.zeros(len(ra), bool)
    for r0, d0, rad in MCS:
        near_mc |= np.hypot((ra - r0) * np.cos(np.radians(dec)), dec - d0) < rad
    return np.where(dec > 5, 0, np.where(near_mc, 2, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-depth", required=True, help="depth file with the gold-referenced zp")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--control", action="store_true",
                    help="instead pick exposures inside Y6 that had a gold zp (tests the reference)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    depth = fitsio.read(cal.DEPTH)
    old = fitsio.read(args.old_depth, columns=["expnum", "zp"])
    model = fitsio.read(COMPLETENESS)
    had = set(old["expnum"][np.isfinite(old["zp"])].tolist())
    with h5py.File(EXPOSURES, "r") as f:
        t = f["__astropy_table__"]
        hpix = dict(zip(t["expnum"][:].astype(int).tolist(), t["hpix"][:].tolist()))
    new = np.array([int(e) not in had for e in depth["expnum"]])
    if args.control:
        new = ~new & (depth["y6_frac"] == 1)
    ok = new & np.isfinite(depth["m1"]) & (depth["n_zp"] >= 50) & np.isin(depth["expnum"], model["expnum"])
    ra, dec = hp.pix2ang(32, np.array([hpix[int(e)] for e in depth["expnum"]]), lonlat=True)
    reg = np.full(len(depth), 3) if args.control else region(ra, dec)

    grid = {}
    for b in "griz":
        used = set()
        for j, _ in enumerate(COLUMNS):
            want = 3 if args.control else {0: 0, 1: 0, 2: 2, 3: 3, 4: 3}[j]
            idx = rng.permutation(np.where(ok & (depth["band"] == b) & (reg == want))[0])
            for i in idx:
                if i not in used:
                    grid[b, j] = int(depth["expnum"][i])
                    used.add(i)
                    break

    drow = {int(r["expnum"]): r for r in depth}
    mrow = {int(r["expnum"]): r for r in model}
    chosen = list(grid.values())
    corners = fitsio.read(cal.CORNERS, ext=1, columns=["expnum", "ccdnum", "ra", "dec"])
    corners = corners[np.isin(corners["expnum"], chosen)]
    jobs = [(drow[e], corners[corners["expnum"] == e]) for e in chosen]
    with Pool(min(len(jobs), 32)) as pool:
        hists = {e: (n, k) for e, n, k in pool.map(histograms, jobs)}

    mm = np.linspace(BINS[0], BINS[-1], 500)
    fig, axes = plt.subplots(4, len(COLUMNS), figsize=(3.8 * len(COLUMNS), 3.0 * 4),
                             sharey=True, squeeze=False)
    for i, b in enumerate("griz"):
        for j, label in enumerate(COLUMNS):
            ax = axes[i, j]
            if (b, j) not in grid:
                ax.text(0.5, 0.5, "no exposure", ha="center", transform=ax.transAxes, color=REF_COL)
                continue
            e = grid[b, j]
            r, m = drow[e], mrow[e]
            n, k = hists[e]
            ax.stairs(k / k.max(), BINS, fill=True, color="#e1e0d9")
            ax.stairs(k / k.max(), BINS, color=HIST_COL, lw=1, label="detected (peak = 1)")
            ax.stairs(n / n.max(), BINS, color=REF_COL, lw=0.8, ls="--",
                      label="all DELVE point sources (peak = 1)")
            ax.plot(mm, cal.logistic(mm, m["m50"], m["k"], m["c"]), color=NOISE_COL, lw=2,
                    label="noise model")
            mid = 0.5 * (BINS[1:] + BINS[:-1])
            enough = n >= 20
            ax.plot(mid[enough], k[enough] / n[enough], "o-", color="black", ms=2.5, lw=0.8,
                    label="detected / catalog")
            ax.set_title(f"{e} {b} {r['exptime']:.0f}s {r['tag'].strip()}", fontsize=8.5)
            ax.text(0.03, 0.05, f"m50 {m['m50']:.2f}  fwhm {r['fwhm']:.2f}\"\nzp_mad {r['zp_mad']:.3f}"
                    f"  n_zp {r['n_zp']}\nN det {k.sum()}", transform=ax.transAxes, fontsize=7.5,
                    color=HIST_COL)
            ax.set_xlim(m["m50"] - 4.5, m["m50"] + 1.5)
            ax.set_ylim(0, 1.08)
            ax.grid(color="#e1e0d9", lw=0.5)
            if i == 3:
                ax.set_xlabel("DELVE WAVG_MAG_PSF [mag]")
        axes[i, 0].set_ylabel(f"{b} band")
    for j, label in enumerate(["inside Y6"] * len(COLUMNS) if args.control else COLUMNS):
        axes[0, j].annotate(label, (0.5, 1.22), xycoords="axes fraction", ha="center",
                            fontsize=11, weight="bold")
    axes[0, 0].legend(loc="upper left", fontsize=6.5, frameon=False)
    fig.suptitle(("Control: Y6 exposures with a gold zp" if args.control else
                  "Newly calibrated exposures (no bright-gold zp; zp from the DELVE catalog)")
                 + ": detected-star histogram and predicted completeness", y=0.995)
    fig.tight_layout()
    out = os.path.join(HERE, "detection_hist_newly_calibrated" + ("_control" * args.control) + ".png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    for (b, j), e in sorted(grid.items()):
        print(b, COLUMNS[j], e, drow[e]["exptime"], drow[e]["tag"].strip(), f"m50={mrow[e]['m50']:.2f}")


if __name__ == "__main__":
    main()
