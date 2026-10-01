"""
Measured completeness curves vs. the noise-based and T_EFF-based models, one panel per
exposure, for exposures NOT in the S/N-threshold calibration sample.

Data: isolated Y6 Gold point sources on the exposure's CCDs, detected = matched to a
clean-flag skim detection (the same definition used in scripts/calibrate_snr_threshold.py),
in 0.2 mag bins.
Noise model: data/delve.exposures.completeness.fits (scripts/build_completeness_table.py).
T_EFF model: the earlier estimator, refit here from the injection-measured 90 s exposures
(data/y6a1c.exposures.positions.fits) as in dev/TeffTesting.ipynb, evaluated at
T_EFF * exptime / 90 s with the median injection c.
"""

import os
import sys
from multiprocessing import Pool

import fitsio
import h5py
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "scripts"))
import calibrate_snr_threshold as cal  # noqa: E402

COMPLETENESS = os.path.join(REPO, "data", "delve.exposures.completeness.fits")
INJECTION = os.path.join(REPO, "data", "y6a1c.exposures.positions.fits")
EXPOSURES = os.path.expanduser("~/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5")
BINS = np.arange(19.0, 25.81, 0.2)
INK, NOISE_COL, TEFF_COL = "#0b0b0b", "#2a78d6", "#eb6834"

# (tag, band, exptime lo, hi): one random hold-out exposure each
PANELS = [("DES", "g", 25, 46), ("DES", "r", 55, 65), ("DES", "i", 90, 90), ("DES", "z", 90, 90),
          ("DES", "i", 140, 160), ("DES", "r", 250, 1000), ("DELVE", "g", 85, 95),
          ("DELVE", "i", 160, 250), ("DELVE", "z", 250, 1000), ("DECADE_F", "r", 0, 1000)]
N_CLOUDY = 2  # plus the DELVE exposures where the two models disagree most


def teff_model():
    """Per-band (m0, k0, s, c) from injection m50/k/c vs log10(T_EFF), all 90 s."""
    inj = fitsio.read(INJECTION, ext=1, columns=["expnum", "band", "m50", "k", "c"])
    teff = load_teff()
    t = np.array([teff.get(int(e), np.nan) for e in inj["expnum"]])
    coeffs = {}
    for b in "griz":
        s = (inj["band"] == b) & (t > 0) & (inj["c"] <= 1) & (inj["k"] < 7)  # notebook's cuts
        lt = np.log10(t[s])
        m0 = np.mean(inj["m50"][s] - 1.25 * lt)
        s_, k0 = np.polyfit(lt, inj["k"][s], 1)
        coeffs[b] = (m0, k0, s_, np.median(inj["c"][s]))
    return coeffs


def load_teff():
    with h5py.File(EXPOSURES, "r") as f:
        t = f["__astropy_table__"]
        return dict(zip(t["expnum"][:].tolist(), t["t_eff"][:].astype(float).tolist()))


def curve(args):
    row, cc = args
    e, b = int(row["expnum"]), str(row["band"])
    mag, _, _, ra, dec = cal.reference_stars(e, b, cc)
    det = cal.detected(e, b, ra, dec)
    n, _ = np.histogram(mag, BINS)
    k, _ = np.histogram(mag[det], BINS)
    return e, n, k


def pick(depth, model, teff, excluded, rng):
    ok = (np.isfinite(depth["m1"]) & (depth["y6_frac"] == 1) & (depth["n_ccd"] >= 50)
          & (depth["n_zp"] >= 50) & ~np.isin(depth["expnum"], excluded)
          & np.isin(depth["expnum"], model["expnum"])
          & np.array([teff.get(int(e), 0) > 0 for e in depth["expnum"]]))
    chosen = []
    for tag, b, lo, hi in PANELS:
        idx = np.where(ok & (depth["tag"] == tag) & (depth["band"] == b)
                       & (depth["exptime"] >= lo) & (depth["exptime"] <= hi))[0]
        if len(idx):
            chosen.append(int(depth["expnum"][rng.choice(idx)]))
    return chosen, ok


def main():
    rng = np.random.default_rng(3)
    depth = fitsio.read(cal.DEPTH)
    model = fitsio.read(COMPLETENESS)
    teff = load_teff()
    coeffs = teff_model()
    excluded = np.load(cal.DIAG)["rows"]["expnum"]
    chosen, ok = pick(depth, model, teff, excluded, rng)

    mrow = {int(r["expnum"]): r for r in model}
    drow = {int(r["expnum"]): r for r in depth}

    def teff_params(e):
        r = drow[e]
        m0, k0, s, c = coeffs[str(r["band"])]
        lt = np.log10(teff[e] * r["exptime"] / 90.0)
        return m0 + 1.25 * lt, k0 + s * lt, c

    # the DELVE exposures where the noise and T_EFF models disagree most
    cand = [int(e) for e in depth["expnum"][ok & (depth["tag"] == "DELVE")]]
    gap = np.array([mrow[e]["m50"] - teff_params(e)[0] for e in cand])
    for i in np.argsort(gap)[:N_CLOUDY]:
        chosen.append(cand[i])

    corners = fitsio.read(cal.CORNERS, ext=1, columns=["expnum", "ccdnum", "ra", "dec"])
    corners = corners[np.isin(corners["expnum"], chosen)]
    jobs = [(drow[e], corners[corners["expnum"] == e]) for e in chosen]
    with Pool(len(jobs)) as pool:
        curves = {e: (n, k) for e, n, k in pool.map(curve, jobs)}

    ncol = 4
    nrow = int(np.ceil(len(chosen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow), sharey=True, squeeze=False)
    mid = 0.5 * (BINS[1:] + BINS[:-1])
    mm = np.linspace(BINS[0], BINS[-1], 400)
    for ax, e in zip(axes.flat, chosen):
        r, m = drow[e], mrow[e]
        n, k = curves[e]
        use = n >= 20
        p = k[use] / n[use]
        ax.errorbar(mid[use], p, yerr=np.sqrt(p * (1 - p) / n[use]), fmt="o-", ms=3, lw=1.3,
                    color=INK, label="data (Y6 Gold)")
        ax.plot(mm, cal.logistic(mm, m["m50"], m["k"], m["c"]), color=NOISE_COL, lw=2,
                label="noise model")
        tm50, tk, tc = teff_params(e)
        ax.plot(mm, cal.logistic(mm, tm50, tk, tc), color=TEFF_COL, lw=2, ls="--",
                label="T_EFF model")
        ax.set_title(f"{e} {r['band']} {r['exptime']:.0f}s {r['tag']}  T_EFF={teff[e]:.2f}", fontsize=9)
        ax.text(0.03, 0.05, f"m50  noise {m['m50']:.2f}\n        T_EFF {tm50:.2f}",
                transform=ax.transAxes, fontsize=8, color="#52514e")
        ax.set_xlim(20, 25.5)
        ax.set_ylim(0, 1.05)
        ax.grid(color="#e1e0d9", lw=0.6)
    for ax in axes.flat[len(chosen):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Y6 Gold PSF_MAG_APER_8 [mag]")
    for ax in axes[:, 0]:
        ax.set_ylabel("detection fraction")
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 0.2), fontsize=8, frameon=False)
    fig.suptitle("Completeness: data vs. noise-based and T_EFF-based models "
                 f"(hold-out exposures; last {N_CLOUDY}: DELVE, largest model disagreement)")
    fig.tight_layout()
    out = os.path.join(HERE, "completeness_models_grid.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    for e in chosen:
        print(e, drow[e]["band"], drow[e]["exptime"], drow[e]["tag"],
              f"noise m50={mrow[e]['m50']:.2f}  teff m50={teff_params(e)[0]:.2f}")


if __name__ == "__main__":
    main()
