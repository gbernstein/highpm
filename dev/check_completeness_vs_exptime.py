"""
Empirical per-exposure detection completeness vs. the model in
data/delve.exposures.completeness.fits, split by exposure time.

The injection model's (m50, k, c) are measured only on 90 s DES wide exposures;
everything else is estimated from T_EFF alone (scripts/build_completeness_table.py).
Here, for a sample of exposures in a pipeline run, we measure completeness
directly: reference = DES Y6 Gold coadd stars (EXT_MASH <= 1, FLAGS_GOLD == 0)
on the exposure's CCDs, with PSF_MAG_APER_8 mags; detected = matched within 1"
in the exposure's own skim. The coadd is ~1 mag deeper than a single 90 s
exposure, so it's a fair truth catalog through the 300 s exposures' m50.
(The DELVE multi-epoch COADDS/cat_hpx_* are built from single-epoch detections,
only as deep as one 90 s exposure, and aren't usable for this.)
"""

import argparse
import os
from collections import defaultdict

import fitsio
import h5py
import healpy as hp
import numpy as np
from matplotlib.path import Path
from scipy.optimize import minimize
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DECAMPM = "/data8/shared/decampm"
GOLD = "/home2/dchgomes/Turbulence_GPR/DES_DELVE/coadd_skims/dr3_gold_{:05d}.fits"
SKIM = os.path.join(DECAMPM, "{B}", "D{e:08d}_{b}_cat.fits")
EXPOSURES = os.path.expanduser("~/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5")
COMPLETENESS = os.path.join(REPO, "data", "delve.exposures.completeness.fits")
CORNERS = os.path.join(REPO, "data", "delve.ccdcorners.fits")
RUN_EXPOSURES = os.path.join(DECAMPM, "PMSculptor_8deg_garyb", "pmsculptor8deg_exposures.npy")

MATCH_ARCSEC = 1.0
EXT_MASH_MAX = 1  # gold star/galaxy class: 0-1 = stars
SHRINK = 0.97  # pull CCD polygons toward their centers to stay off the edges

GROUPS = {  # label -> exptime range [lo, hi], seconds
    "30-60s": (25, 65),
    "90s": (90, 90),
    "140-200s": (140, 200),
    "250-300s": (250, 300),
}


def logistic(m, m50, k, c):
    return c / (1 + np.exp(k * (m - m50)))


def fit_logistic(mag, det):
    def nll(p):
        pr = np.clip(logistic(mag, *p), 1e-9, 1 - 1e-9)
        return -np.sum(det * np.log(pr) + (~det) * np.log(1 - pr))
    res = minimize(nll, [np.median(mag[det]) + 1, 5.0, 0.95], method="Nelder-Mead",
                   options=dict(maxiter=4000, xatol=1e-4, fatol=1e-4))
    return res.x


def measure(expnum, band, corners, ext_mash_max=EXT_MASH_MAX):
    b, B = band, band.upper()
    skim = fitsio.read(SKIM.format(B=B, e=expnum, b=b), columns=["RA", "DEC"])
    cc = corners[corners["expnum"] == expnum]
    pix = np.unique(hp.ang2pix(32, cc["ra"][:, :4].ravel(), cc["dec"][:, :4].ravel(), lonlat=True))
    mcol = f"PSF_MAG_APER_8_{B}"
    cols = [mcol, "EXT_MASH", "FLAGS_GOLD", "ALPHAWIN_J2000", "DELTAWIN_J2000"]
    missing = [p for p in pix if not os.path.exists(GOLD.format(p))]
    if missing:
        print(f"  warning: {expnum} missing Y6 Gold pixels {missing}")
    ref = np.concatenate([fitsio.read(GOLD.format(p), columns=cols)
                          for p in pix if os.path.exists(GOLD.format(p))])
    ref = ref[(ref["EXT_MASH"] >= 0) & (ref["EXT_MASH"] <= ext_mash_max)
              & (ref["FLAGS_GOLD"] == 0) & (ref[mcol] > 16) & (ref[mcol] < 27)]
    ra, dec = ref["ALPHAWIN_J2000"], ref["DELTAWIN_J2000"]

    inside = np.zeros(len(ref), bool)
    for row in cc:
        v = np.c_[row["ra"][:4], row["dec"][:4]]
        v = v.mean(0) + SHRINK * (v - v.mean(0))
        inside |= Path(v).contains_points(np.c_[ra, dec])
    ref, ra, dec = ref[inside], ra[inside], dec[inside]

    # Flat-sky match is fine at 1" on a single exposure's footprint.
    cosd = np.cos(np.radians(np.mean(dec)))
    tree = cKDTree(np.c_[skim["RA"] * cosd, skim["DEC"]])
    d, _ = tree.query(np.c_[ra * cosd, dec], distance_upper_bound=MATCH_ARCSEC / 3600)
    return ref[mcol].astype(float), np.isfinite(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--out", default=os.path.join(REPO, "dev", "completeness_vs_exptime.npz"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    with h5py.File(EXPOSURES, "r") as f:
        t = f["__astropy_table__"]
        ex, xt = t["expnum"][:].astype(int), t["exptime"][:].astype(float)
        bd, te = np.char.decode(t["band"][:]), t["t_eff"][:].astype(float)
    info = {e: (x, b, tt) for e, x, b, tt in zip(ex, xt, bd, te)}
    comp = fitsio.read(COMPLETENESS)
    model = {int(r["expnum"]): r for r in comp}
    run = np.load(RUN_EXPOSURES)

    pools = defaultdict(list)
    for e in run.tolist():
        if e not in info or e not in model:
            continue
        x, b, _ = info[e]
        if b not in "griz" or not os.path.exists(SKIM.format(B=b.upper(), e=e, b=b)):
            continue
        for g, (lo, hi) in GROUPS.items():
            if lo <= x <= hi:
                # split 90 s into measured vs t_eff-estimated
                key = g if g != "90s" else ("90s est" if model[e]["estimated"] else "90s meas")
                pools[key].append(e)

    corners = fitsio.read(CORNERS, ext=1, columns=["expnum", "ccdnum", "ra", "dec"])
    rows = []
    for g in ["90s meas", "90s est", "30-60s", "140-200s", "250-300s"]:
        pick = rng.permutation(pools[g])[: args.per_group]
        print(f"== {g}: {len(pools[g])} available, measuring {len(pick)}")
        for e in pick:
            x, b, tt = info[e]
            mag, det = measure(int(e), b, corners[np.isin(corners["expnum"], [e])])
            m50, k, c = fit_logistic(mag, det)
            mm = model[e]
            rows.append((g, e, b, x, tt, mm["m50"], mm["k"], mm["c"], bool(mm["estimated"]),
                         m50, k, c, len(mag)))
            print(f"  {e} {b} {x:5.0f}s teff={tt:.2f}  model m50={mm['m50']:.2f}  "
                  f"emp m50={m50:.2f} k={k:.1f} c={c:.2f}  d={m50 - mm['m50']:+.2f}  N={len(mag)}")

    arr = np.array(rows, dtype=[("group", "U10"), ("expnum", "i8"), ("band", "U1"),
                                ("exptime", "f8"), ("teff", "f8"), ("model_m50", "f8"),
                                ("model_k", "f8"), ("model_c", "f8"), ("estimated", "?"),
                                ("emp_m50", "f8"), ("emp_k", "f8"), ("emp_c", "f8"), ("nref", "i8")])
    np.savez(args.out, rows=arr)
    print(f"wrote {args.out}")
    d = arr["emp_m50"] - arr["model_m50"]
    print("\nmedian (empirical - model) m50 by group:")
    for g in dict.fromkeys(arr["group"]):
        s = arr["group"] == g
        dd = d[s]
        err = 1.4826 * np.median(np.abs(dd - np.median(dd))) / np.sqrt(s.sum())
        print(f"  {g:10s} {np.median(dd):+.2f} +- {err:.2f}")


if __name__ == "__main__":
    main()
