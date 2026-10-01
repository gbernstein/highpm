"""
Calibrate the S/N detection threshold that maps per-CCD depth (m1, the magnitude at
S/N = 1 from scripts/measure_exposure_depth.py) to a completeness curve:

    P(detected | m) = c / (1 + exp(k (m - m50))),
    m50 = m1_ccd - 2.5 log10(nu) - seeing_loss(fwhm, kernel)

with nu, k and c fit per band and per processing tag (DES and DELVE finalcut use
different detection thresholds). seeing_loss (scripts/build_completeness_table.py) is
the S/N a fixed Gaussian detection filter of FWHM kernel loses against the PSF-fit S/N
behind m1, 2.5 log10((kernel^2 + fwhm^2) / (2 kernel fwhm)); kernel is fit only for
KERNEL_TAGS. Both DES and DELVE finalcut lose depth at both seeing extremes, as a fixed
filter does: DELVE's kernel is ~0.95" (~3.6 px), DES's ~1.7" (~6.4 px), so DES only shows
the loss in its seeing tails (dev/snr_residual_correlations.png). DECADE has too few
exposures and too narrow a seeing range to constrain one, so it gets no correction.
The truth catalog is DES Y6 Gold, so the fit uses exposures lying entirely in Y6 Gold,
stratified by exposure time, plus TAIL_PER_BIN extra exposures per band in each seeing
tail of the KERNEL_TAGS (a random draw rarely reaches them, and they fix the kernel).

Reference: isolated Y6 Gold point sources (EXT_MASH 0-1, FLAGS_GOLD == 0, no other
gold object within ISOLATION_ARCSEC; EXT_MASH == 0 alone runs out below i ~ 23.5, short
of deep exposures' m50) inside the exposure's CCDs, away from edges.
The fakes pipeline removes FOV, charge-trap and blended fakes itself, so those losses
are kept out of c. Detected: matched within 1" to a skim detection passing clean_cat's
flag cuts (FLAGS < 4, IMAFLAGS_ISO == 0). clean_cat's single-epoch star/galaxy cut is
deliberately left out: it rejects a large, S/N-dependent fraction of faint true stars
that isn't a detection threshold and doesn't follow this logistic (see
dev/check_star_class_loss.py).

Gold magnitude errors and k are combined in quadrature (logistic sigma = 1.814/k), so
the fitted k is that of exact magnitudes, as the fakes have. Each exposure carries equal
weight in the fit (stars weighted by 1 / its star count), as the table is used per
exposure; star weighting lets dense fields, which here are mostly sharp-seeing ones,
set the seeing term.

Writes data/snr_threshold.fits and dev/snr_threshold_calibration.npz (per-exposure
diagnostics: free logistic fits vs. the calibrated prediction).
"""

import argparse
import os
import sys
from multiprocessing import Pool

import fitsio
import healpy as hp
import numpy as np
from matplotlib.path import Path
from scipy.optimize import minimize
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_completeness_table import FWHM_DEFAULT, seeing_loss  # noqa: E402
from measure_exposure_depth import NSIDE, SKIM, match  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GOLD = "/home2/dchgomes/Turbulence_GPR/DES_DELVE/coadd_skims/dr3_gold_{:05d}.fits"
DEPTH = os.path.join(REPO, "data", "delve.exposures.depth.fits")
CORNERS = os.path.join(REPO, "data", "delve.ccdcorners.fits")
OUT = os.path.join(REPO, "data", "snr_threshold.fits")
DIAG = os.path.join(REPO, "dev", "snr_threshold_calibration.npz")

SHRINK = 0.97  # pull CCD polygons toward their centers to stay off the edges
ISOLATION_ARCSEC = 2.0
FIT_MAG = (18.0, 26.0)  # bright limit keeps saturation out of c
EXPTIME_BINS = [0, 45, 75, 105, 160, 250, 1000]
KERNEL_TAGS = ("DES", "DELVE")
SEEING_TAILS = [(0.0, 1.3), (2.4, 9.0)]  # arcsec
TAIL_PER_BIN = 20
LOGISTIC_SIGMA = np.pi / np.sqrt(3)  # std of a unit logistic, ~1.814


def logistic(m, m50, k, c):
    return c / (1 + np.exp(np.clip(k * (m - m50), -50, 50)))


def reference_stars(expnum, band, cc, mash_max=1):
    """Gold (mag, magerr, ccdnum, ra, dec) of isolated point sources on the CCDs."""
    B = band.upper()
    pix = np.unique(hp.ang2pix(NSIDE, cc["ra"][:, :4].ravel(), cc["dec"][:, :4].ravel(), lonlat=True))
    cols = ["ALPHAWIN_J2000", "DELTAWIN_J2000", "EXT_MASH", "FLAGS_GOLD", "SOURCE",
            f"PSF_MAG_APER_8_{B}", f"PSF_MAG_ERR_APER_8_{B}"]
    g = np.concatenate([fitsio.read(GOLD.format(p), columns=cols) for p in pix])
    ra, dec = g["ALPHAWIN_J2000"], g["DELTAWIN_J2000"]

    ccd = np.zeros(len(g), "i2")
    for row in cc:
        v = np.c_[row["ra"][:4], row["dec"][:4]]
        v = v.mean(0) + SHRINK * (v - v.mean(0))
        ccd[Path(v).contains_points(np.c_[ra, dec]) & (ccd == 0)] = row["ccdnum"]
    on = ccd > 0

    # isolation against every gold object (any class), not just stars
    cosd = np.cos(np.radians(np.median(dec[on])))
    tree = cKDTree(np.c_[ra[on] * cosd, dec[on]])
    nn = tree.query(np.c_[ra[on] * cosd, dec[on]], k=2)[0][:, 1]
    isolated = np.zeros(len(g), bool)
    isolated[on] = nn > ISOLATION_ARCSEC / 3600

    mag = g[f"PSF_MAG_APER_8_{B}"]
    keep = (isolated & (g["EXT_MASH"] >= 0) & (g["EXT_MASH"] <= mash_max) & (g["FLAGS_GOLD"] == 0)
            & (np.char.strip(g["SOURCE"].astype(str)) == "Y6_GOLD")
            & (mag > FIT_MAG[0]) & (mag < FIT_MAG[1]))
    return mag[keep], g[f"PSF_MAG_ERR_APER_8_{B}"][keep], ccd[keep], ra[keep], dec[keep]


def detected(expnum, band, ra, dec):
    s = fitsio.read(SKIM.format(B=band.upper(), e=expnum, b=band),
                    columns=["RA", "DEC", "FLAGS", "IMAFLAGS_ISO"])
    idx = match(ra, dec, s["RA"], s["DEC"])
    passes = (s["FLAGS"] < 4) & (s["IMAFLAGS_ISO"] == 0)
    return (idx >= 0) & passes[np.maximum(idx, 0)]


def fit_free(mag, det):
    """Unconstrained per-exposure logistic, for diagnostics."""
    def nll(p):
        pr = np.clip(logistic(mag, *p), 1e-9, 1 - 1e-9)
        return -np.sum(np.where(det, np.log(pr), np.log(1 - pr)))
    return minimize(nll, [np.median(mag[det]) + 1, 5.0, 0.95], method="Nelder-Mead",
                    options=dict(maxiter=4000, xatol=1e-4, fatol=1e-4)).x


def stars_for(args):
    row, cc = args
    e, b = int(row["expnum"]), str(row["band"])
    mag, err, ccd, ra, dec = reference_stars(e, b, cc)
    m1 = row["m1_ccd"][ccd]
    ok = np.isfinite(m1) & np.isfinite(err)
    det = detected(e, b, ra[ok], dec[ok])
    fwhm = np.full(ok.sum(), np.nan_to_num(row["fwhm"], nan=FWHM_DEFAULT))
    return e, mag[ok], err[ok], m1[ok], det, fwhm


def fit_threshold(mag, err, m1, det, fwhm, weight, fit_kernel):
    """Joint fit of (nu, k, c[, kernel]) over all stars of one band/tag."""
    def unpack(p):
        return 10 ** p[0], np.exp(p[1]), 1 / (1 + np.exp(-p[2])), (np.exp(p[3]) if fit_kernel else 0.0)

    def nll(p):
        nu, k, c, kernel = unpack(p)
        keff = LOGISTIC_SIGMA / np.sqrt((LOGISTIC_SIGMA / k) ** 2 + err ** 2)
        m50 = m1 - 2.5 * np.log10(nu) - seeing_loss(fwhm, kernel)
        pr = np.clip(logistic(mag, m50, keff, c), 1e-9, 1 - 1e-9)
        return -np.sum(weight * np.where(det, np.log(pr), np.log(1 - pr)))

    p0 = [np.log10(5.0), np.log(6.0), 3.0] + ([np.log(1.2)] if fit_kernel else [])
    res = minimize(nll, p0, method="Nelder-Mead",
                   options=dict(maxiter=8000, xatol=1e-5, fatol=1e-3))
    return unpack(res.x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bin", type=int, default=10, help="exposures per band/tag/exptime bin")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-j", "--jobs", type=int, default=32)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    depth = fitsio.read(DEPTH)
    usable = (np.isfinite(depth["m1"]) & (depth["y6_frac"] == 1) & (depth["n_ccd"] >= 50)
              & (depth["n_zp"] >= 50))
    pick = []
    for b in "griz":
        for tag in np.unique(depth["tag"]):
            for lo, hi in zip(EXPTIME_BINS[:-1], EXPTIME_BINS[1:]):
                sel = np.where(usable & (depth["band"] == b) & (depth["tag"] == tag)
                               & (depth["exptime"] >= lo) & (depth["exptime"] < hi))[0]
                pick.extend(rng.permutation(sel)[: args.per_bin].tolist())
    for b in "griz":
        for tag in KERNEL_TAGS:
            for lo, hi in SEEING_TAILS:
                sel = np.where(usable & (depth["band"] == b) & (depth["tag"] == tag)
                               & (depth["fwhm"] >= lo) & (depth["fwhm"] < hi))[0]
                sel = np.setdiff1d(sel, pick)
                pick.extend(rng.permutation(sel)[:TAIL_PER_BIN].tolist())
    sample = depth[np.sort(pick)]
    print(f"calibration sample: {len(sample)} exposures")

    corners = fitsio.read(CORNERS, ext=1, columns=["expnum", "ccdnum", "ra", "dec"])
    corners = corners[np.isin(corners["expnum"], sample["expnum"])]
    jobs = [(r, corners[corners["expnum"] == r["expnum"]]) for r in sample]
    with Pool(args.jobs) as pool:
        stars = {e: rest for e, *rest in pool.imap_unordered(stars_for, jobs)}

    fits, diag = [], []
    for b in "griz":
        for tag in np.unique(sample["tag"]):
            rows = sample[(sample["band"] == b) & (sample["tag"] == tag)]
            if len(rows) == 0:
                continue
            mag, err, m1, det, fwhm = (np.concatenate([stars[e][i] for e in rows["expnum"]])
                                       for i in range(5))
            n_e = [len(stars[e][0]) for e in rows["expnum"]]
            weight = np.repeat(np.mean(n_e) / np.maximum(n_e, 1), n_e)
            nu, k, c, kernel = fit_threshold(mag, err, m1, det, fwhm, weight, tag in KERNEL_TAGS)
            fits.append((b, tag, nu, k, c, kernel, len(rows), len(mag)))
            print(f"{b} {tag:6s} nu={nu:.2f} k={k:.2f} c={c:.3f} kernel={kernel:.2f}\"  "
                  f"({len(rows)} exposures, {len(mag)} stars)")
            for r in rows:
                mg, er, m1s, dt, fw = stars[r["expnum"]]
                m50, kf, cf = fit_free(mg, dt)
                pred = r["m1"] - 2.5 * np.log10(nu) - seeing_loss(r["fwhm"], kernel)
                diag.append((r["expnum"], b, tag, r["exptime"], r["m1"], r["m1_ccd_std"],
                             r["fwhm"], pred, m50, kf, cf, len(mg)))

    out = np.array(fits, dtype=[("band", "U1"), ("tag", "U8"), ("nu", "f8"), ("k", "f8"),
                                ("c", "f8"), ("kernel", "f8"), ("n_exp", "i4"),
                                ("n_star", "i8")])
    fitsio.write(OUT, out, clobber=True)
    d = np.array(diag, dtype=[("expnum", "i4"), ("band", "U1"), ("tag", "U8"), ("exptime", "f4"),
                              ("m1", "f8"), ("m1_ccd_std", "f4"), ("fwhm", "f4"), ("pred_m50", "f8"),
                              ("emp_m50", "f8"), ("emp_k", "f8"), ("emp_c", "f8"), ("nref", "i8")])
    np.savez(DIAG, rows=d)
    print(f"wrote {OUT} and {DIAG}")

    print("\nper-exposure free-fit m50 minus calibrated prediction, median [robust std]:")
    for b in "griz":
        for tag in np.unique(d["tag"]):
            line = []
            for lo, hi in zip(EXPTIME_BINS[:-1], EXPTIME_BINS[1:]):
                s = (d["band"] == b) & (d["tag"] == tag) & (d["exptime"] >= lo) & (d["exptime"] < hi)
                if s.sum():
                    r = d["emp_m50"][s] - d["pred_m50"][s]
                    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
                    line.append(f"{lo}-{hi}s: {np.median(r):+.2f} [{mad:.2f}] n={s.sum()}")
            if line:
                print(f"  {b} {tag:6s} " + "; ".join(line))


if __name__ == "__main__":
    main()
