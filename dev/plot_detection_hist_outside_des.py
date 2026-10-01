"""
Exposures entirely outside the DES Y6 footprint: histogram of detected stars vs. coadd
magnitude from the DELVE catalogs in /data8/shared/decampm/COADDS (normalized to peak 1),
with the predicted completeness overlaid. Rows = band, columns = exposure-time class.

Reference: DELVE point sources (EXTENDED_CLASS_<B> 0-1, WAVG_MAG_PSF_<B>) on the
exposure's CCDs, isolated by 2" from any other catalog object. Detected: matched within 1"
to a clean-flag skim detection (scripts/calibrate_snr_threshold.py definitions).
The DELVE catalog is built from single epochs, so its own depth is not far beyond the
exposure's; its point-source histogram (own peak = 1) is drawn too, so a roll-off of the
catalog can be told apart from a roll-off of the exposure.

Prediction: noise model from data/delve.exposures.completeness.fits (zp from the DELVE
catalog), and the T_EFF model of dev/plot_completeness_models_grid.py.
"""

import os
import sys
from multiprocessing import Pool

import fitsio
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, HERE)
import calibrate_snr_threshold as cal  # noqa: E402
from plot_completeness_models_grid import load_teff, teff_model  # noqa: E402

COADD = "/data8/shared/decampm/COADDS/cat_hpx_{:05d}.fits"
COMPLETENESS = os.path.join(REPO, "data", "delve.exposures.completeness.fits")
BINS = np.arange(16.0, 26.01, 0.1)
HIST_COL, REF_COL, NOISE_COL, TEFF_COL = "#52514e", "#898781", "#2a78d6", "#eb6834"

# columns: (label, tags in order of preference, exptime lo, hi)
COLUMNS = [("short (<= 50 s)", ["DELVE", "DES", "DECADE_F"], 0, 50),
           ("60-90 s DESDM", ["DES"], 60, 91),
           ("60-90 s DELVE", ["DELVE", "DECADE_F"], 60, 91),
           ("150-250 s", ["DELVE", "DES", "DECADE_F"], 140, 250),
           ("> 250 s", ["DES", "DELVE", "DECADE_F"], 251, 1000)]


def coadd_stars(band, cc):
    """DELVE catalog (mag, is point source, detected-eligible mask) of objects on the CCDs."""
    B = band.upper()
    pix = np.unique(hp.ang2pix(32, cc["ra"][:, :4].ravel(), cc["dec"][:, :4].ravel(), lonlat=True))
    cols = ["RA", "DEC", f"WAVG_MAG_PSF_{B}", f"EXTENDED_CLASS_{B}", f"WAVG_FLAGS_{B}"]
    c = np.concatenate([fitsio.read(COADD.format(p), columns=cols) for p in pix
                        if os.path.exists(COADD.format(p))])
    ra, dec = c["RA"], c["DEC"]
    on = np.zeros(len(c), bool)
    for row in cc:
        v = np.c_[row["ra"][:4], row["dec"][:4]]
        v = v.mean(0) + cal.SHRINK * (v - v.mean(0))
        on |= Path(v).contains_points(np.c_[ra, dec])
    cosd = np.cos(np.radians(np.median(dec[on])))
    nn = cKDTree(np.c_[ra[on] * cosd, dec[on]]).query(np.c_[ra[on] * cosd, dec[on]], k=2)[0][:, 1]
    isolated = np.zeros(len(c), bool)
    isolated[on] = nn > cal.ISOLATION_ARCSEC / 3600
    mag = c[f"WAVG_MAG_PSF_{B}"]
    ext = c[f"EXTENDED_CLASS_{B}"]
    keep = isolated & (ext >= 0) & (ext <= 1) & (c[f"WAVG_FLAGS_{B}"] < 4) & (mag > 10) & (mag < 30)
    return mag[keep], ra[keep], dec[keep]


def histograms(args):
    row, cc = args
    e, b = int(row["expnum"]), str(row["band"])
    mag, ra, dec = coadd_stars(b, cc)
    det = cal.detected(e, b, ra, dec)
    return e, np.histogram(mag, BINS)[0], np.histogram(mag[det], BINS)[0]


def main():
    rng = np.random.default_rng(7)
    depth = fitsio.read(cal.DEPTH)
    model = fitsio.read(COMPLETENESS)
    teff = load_teff()
    coeffs = teff_model()
    ok = ((depth["y6_frac"] == 0) & np.isfinite(depth["m1"]) & (depth["n_ccd"] >= 50)
          & (depth["n_zp"] >= 50) & np.isin(depth["expnum"], model["expnum"])
          & np.array([teff.get(int(e), 0) > 0 for e in depth["expnum"]]))

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
        hists = {e: (n, k) for e, n, k in pool.map(histograms, jobs)}

    def teff_params(e):
        r = drow[e]
        m0, k0, s, c = coeffs[str(r["band"])]
        lt = np.log10(teff[e] * r["exptime"] / 90.0)
        return m0 + 1.25 * lt, k0 + s * lt, c

    mm = np.linspace(BINS[0], BINS[-1], 500)
    fig, axes = plt.subplots(4, len(COLUMNS), figsize=(3.8 * len(COLUMNS), 3.0 * 4),
                             sharey=True, squeeze=False)
    for i, b in enumerate("griz"):
        for j, (label, *_) in enumerate(COLUMNS):
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
            tm50, tk, tc = teff_params(e)
            ax.plot(mm, cal.logistic(mm, tm50, tk, tc), color=TEFF_COL, lw=1.6, ls="--",
                    label="T_EFF model")
            ax.set_title(f"{e} {b} {r['exptime']:.0f}s {r['tag']}  T_EFF={teff[e]:.2f}", fontsize=8.5)
            ax.text(0.03, 0.05, f"m50 noise {m['m50']:.2f}\n       T_EFF {tm50:.2f}\nN det {k.sum()}",
                    transform=ax.transAxes, fontsize=7.5, color=HIST_COL)
            ax.set_xlim(m["m50"] - 4.5, m["m50"] + 1.5)
            ax.set_ylim(0, 1.08)
            ax.grid(color="#e1e0d9", lw=0.5)
            if i == 3:
                ax.set_xlabel("DELVE WAVG_MAG_PSF [mag]")
        axes[i, 0].set_ylabel(f"{b} band")
    for j, (label, *_) in enumerate(COLUMNS):
        axes[0, j].annotate(label, (0.5, 1.22), xycoords="axes fraction", ha="center",
                            fontsize=11, weight="bold")
    axes[0, 0].legend(loc="upper left", fontsize=6.5, frameon=False)
    fig.suptitle("Exposures outside DES Y6: detected-star histogram vs. DELVE catalog magnitude "
                 "and predicted completeness", y=0.995)
    fig.tight_layout()
    out = os.path.join(HERE, "detection_hist_outside_des.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    for (b, j), e in sorted(grid.items()):
        print(b, COLUMNS[j][0], e, drow[e]["exptime"], drow[e]["tag"],
              f"noise m50={mrow[e]['m50']:.2f} teff m50={teff_params(e)[0]:.2f}")


if __name__ == "__main__":
    main()
