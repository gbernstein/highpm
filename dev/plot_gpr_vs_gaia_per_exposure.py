"""Per-EXPOSURE offset between GPR-corrected detection positions and Gaia,
as a function of time.

For every position-corrected exposure, match its detections to Gaia stars
(Gaia propagated to that exposure's epoch with Gaia's own PM and the
exposure's parallax factors), and record the median (GPR - Gaia) offset in
RA*cos(dec) and Dec. A constant PM offset between the GPR frame and Gaia
shows up as a linear drift of this per-exposure offset with MJD; a constant
zero-point shows up as a flat, nonzero offset; a per-epoch problem shows up
as exposure-to-exposure scatter.

The per-exposure table is cached (--cache) so replotting is instant.
"""

import argparse
import glob
import os
import sys
import multiprocessing as mp
import re

import fitsio
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy.lib.recfunctions as rfn
from healpy import ang2pix
from scipy.stats import norm
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from highpm.position_correction import getGPRFile
from healpix_region_data import load_coadd, load_movers

GAIA_REF_MJD = 57388.5  # J2016.0
YEAR = 365.25
MAS = 3.6e6             # mas per degree

_G = {}


def robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def ref_keys(expnum, gprdirs):
    """int64 keys expnum*1e10 + ccd*1e8 + object_number of the stars GPR flagged ref=True in this exposure."""
    f = getGPRFile(expnum, gprdirs)
    if f is None:
        return np.zeros(0, dtype=np.int64)
    g = fitsio.read(f, ext=1, columns=["id", "ref"])
    ids = [s.split("_") for s in g["id"][g["ref"].astype(bool)]]
    if not ids:
        return np.zeros(0, dtype=np.int64)
    ids = np.array(ids, dtype=np.int64)
    return expnum * 10**10 + ids[:, 0] * 10**8 + ids[:, 1]


def _expnum(path):
    return int(re.search(r"(\d{8})\.fits$", path).group(1))


def per_exposure(path):
    try:
        d = fitsio.read(path, ext="DATA", columns=["BEST_RA", "BEST_DEC", "MJD", "PAR_XI", "PAR_ETA",
                                                   "TRAP_FLAG", "FLAGS", "CCDNUM", "OBJECT_NUMBER"])
    except Exception:
        return None
    d = d[~d["TRAP_FLAG"] & (d["FLAGS"] == 0)]
    if _G["ref_only"]:
        keys = ref_keys(_expnum(path), _G["gprdirs"]) - _expnum(path) * 10**10
        d = d[np.isin(d["CCDNUM"].astype(np.int64) * 10**8 + d["OBJECT_NUMBER"].astype(np.int64), keys)]
    if len(d) == 0 or d["MJD"][0] < 0:
        return None
    ra, dec = d["BEST_RA"], d["BEST_DEC"]
    dist, j = _G["tree"].query(np.column_stack([ra * _G["cos0"], dec]), distance_upper_bound=_G["cand_deg"])
    ok = np.isfinite(dist)
    if ok.sum() < 20:
        return None
    d, j = d[ok], j[ok]
    g = _G["gaia"]
    dt = (d["MJD"] - GAIA_REF_MJD) / YEAR
    cosd = np.cos(np.radians(g["DEC"][j]))
    g_ra = g["RA"][j] + (g["PMRA"][j] * dt + g["PARALLAX"][j] * d["PAR_XI"]) / MAS / cosd
    g_dec = g["DEC"][j] + (g["PMDEC"][j] * dt + g["PARALLAX"][j] * d["PAR_ETA"]) / MAS
    dra = (d["BEST_RA"] - g_ra) * cosd * MAS
    ddec = (d["BEST_DEC"] - g_dec) * MAS
    close = np.hypot(dra, ddec) < _G["match_mas"]
    dra, ddec = dra[close], ddec[close]
    if len(dra) < 20:
        return None
    # one more clip against the exposure's own median to drop residual mismatches
    for _ in range(2):
        keep = (np.abs(dra - np.median(dra)) < 4 * robust_std(dra)) & (np.abs(ddec - np.median(ddec)) < 4 * robust_std(ddec))
        dra, ddec = dra[keep], ddec[keep]
    n = len(dra)
    expnum = _expnum(path)
    return (expnum, d["MJD"][0], n, np.median(dra), np.median(ddec),
            1.2533 * robust_std(dra) / np.sqrt(n), 1.2533 * robust_std(ddec) / np.sqrt(n),
            robust_std(dra), robust_std(ddec))


def fit_line(t, y, err):
    w = 1.0 / err ** 2
    A = np.vstack([np.ones_like(t), t]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    p = cov @ (A.T @ (w * y))
    chi2 = np.sum(w * (y - A @ p) ** 2) / (len(t) - 2)
    return p, np.sqrt(np.diag(cov) * max(chi2, 1.0)), chi2


def pm_pulls(pixels, args, ref_only):
    """Match our modest1 movers to Gaia; return (dpmra, dpmdec, pull_ra, pull_dec)."""
    movers = load_movers(args.pmcatalog_dir, 32, pixels, pattern="real_PM_hp*.fits", exts=("modest1_movers",))
    gaia = load_coadd(args.gaia_dir, pixels, pattern="GaiaSource_*.fits")
    gaia = gaia[np.isfinite(gaia["PMRA"]) & ~((gaia["PMRA"] == 0) & (gaia["PMDEC"] == 0) & (gaia["PARALLAX"] == 0))]
    gmin, gmax = args.gmag_range
    gaia = gaia[(gaia["RUWE"] <= args.max_ruwe) & (gaia["PHOT_G_MEAN_MAG"] >= gmin) & (gaia["PHOT_G_MEAN_MAG"] <= gmax)]
    if ref_only:
        files = sorted(glob.glob(os.path.join(args.exposure_dir, "position_corrected_*.fits")))
        with mp.get_context("fork").Pool(args.nproc) as pool:
            allkeys = np.sort(np.concatenate(pool.starmap(ref_keys, [(_expnum(f), args.gpr_dirs) for f in files])))
        frac = {}
        for hp in pixels:
            pf = os.path.join(args.pmcatalog_dir, f"real_PM_hp{hp:05d}.fits")
            hdc = os.path.join(os.path.dirname(args.pmcatalog_dir.rstrip("/")), "HealpixDetectionCatalog",
                               f"cleaned_detections_hp{hp:05d}.fits")
            det = fitsio.read(pf, ext="modest1_detections")
            det = det[~det["clipped"]]
            c = fitsio.read(hdc, columns=["EXPNUM", "CCDNUM", "OBJECT_NUMBER"])[det["detections"]]
            k = (c["EXPNUM"].astype(np.int64) * 10**10 + c["CCDNUM"].astype(np.int64) * 10**8
                 + c["OBJECT_NUMBER"].astype(np.int64))
            isref = np.isin(k, allkeys).astype(float)
            n = int(det["idx"].max()) + 1
            frac[hp] = np.bincount(det["idx"], isref, n) / np.maximum(np.bincount(det["idx"], minlength=n), 1)
        hpix = ang2pix(32, movers["ra"], movers["dec"], lonlat=True)
        movers = rfn.append_fields(movers, "ref_frac",
                                   np.array([frac[int(h)][int(i)] if int(h) in frac and int(i) < len(frac[int(h)]) else np.nan
                                             for h, i in zip(hpix, movers["idx"])]), usemask=False)
        movers = movers[movers["ref_frac"] >= args.min_ref_frac]
        print(f"[ref] {len(movers)} movers with ref fraction >= {args.min_ref_frac}")
    idx, d2d, _ = SkyCoord(ra=movers["ra"] * u.degree, dec=movers["dec"] * u.degree).match_to_catalog_sky(
        SkyCoord(ra=gaia["RA"] * u.degree, dec=gaia["DEC"] * u.degree))
    close = d2d.arcsecond < 0.3
    m = rfn.merge_arrays([movers[close], gaia[idx[close]]], flatten=True, usemask=False)
    dra, ddec = m["pmra"] - m["PMRA"], m["pmdec"] - m["PMDEC"]
    pra = dra / np.hypot(1000 * np.sqrt(m["c_vxvx"]), m["PMRA_ERROR"])
    pdec = ddec / np.hypot(1000 * np.sqrt(m["c_vyvy"]), m["PMDEC_ERROR"])
    ok = np.isfinite(pra) & np.isfinite(pdec)
    print(f"[pm match] {ok.sum()} movers matched to Gaia")
    return dra[ok], ddec[ok], pra[ok], pdec[ok]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--healpix-npy", default="/data8/shared/decampm/PMSculptor_8deg_garyb/pmsculptor8deg_healpix.npy")
    parser.add_argument("--exposure-dir",
                         default="/data8/shared/decampm/PMSculptor_8deg_garyb/PositionCorrectedExposureCatalog/")
    parser.add_argument("--pmcatalog-dir", default="/data8/shared/decampm/PMSculptor_8deg_garyb/PMCatalog/")
    parser.add_argument("--gpr-dirs", nargs="+", default=[f"/data8/shared/decampm/GPR2/CAT/{b}/" for b in "griz"])
    parser.add_argument("--all-stars", action="store_true", help="Do not restrict to GPR reference (ref=True) stars")
    parser.add_argument("--min-ref-frac", type=float, default=0.5,
                         help="A catalog star counts as a GPR reference star if at least this fraction of its "
                              "detections were ref=True")
    parser.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia/")
    parser.add_argument("--gmag-range", type=float, nargs=2, default=(15.0, 19.0))
    parser.add_argument("--max-ruwe", type=float, default=1.4)
    parser.add_argument("--match-mas", type=float, default=300.0, help="Final match radius after propagation")
    parser.add_argument("--candidate-arcsec", type=float, default=2.0,
                         help="Candidate radius against J2016 Gaia positions (must cover PM * baseline)")
    parser.add_argument("--nproc", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N exposures (testing)")
    parser.add_argument("--cache", default=None)
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()
    ref_only = not args.all_stars
    if args.out is None:
        args.out = "gpr_vs_gaia_per_exposure_ref.png" if ref_only else "gpr_vs_gaia_per_exposure.png"
    if args.cache is None:
        args.cache = "gpr_vs_gaia_per_exposure.npy" if not ref_only else "gpr_vs_gaia_per_exposure_ref.npy"

    if not os.path.exists(args.cache):
        pixels = set(int(h) for h in np.load(args.healpix_npy))
        gaia = load_coadd(args.gaia_dir, pixels, pattern="GaiaSource_*.fits")
        gaia = gaia[np.isfinite(gaia["PMRA"]) & ~((gaia["PMRA"] == 0) & (gaia["PMDEC"] == 0) & (gaia["PARALLAX"] == 0))]
        gmin, gmax = args.gmag_range
        gaia = gaia[(gaia["RUWE"] <= args.max_ruwe) & (gaia["PHOT_G_MEAN_MAG"] >= gmin) & (gaia["PHOT_G_MEAN_MAG"] <= gmax)]
        print(f"[gaia] {len(gaia)} reference stars, G in [{gmin}, {gmax}], RUWE <= {args.max_ruwe}")
        cos0 = np.cos(np.radians(np.median(gaia["DEC"])))
        _G.update(gaia=gaia, cos0=cos0, cand_deg=args.candidate_arcsec / 3600.0, match_mas=args.match_mas, ref_only=ref_only, gprdirs=args.gpr_dirs,
                  tree=cKDTree(np.column_stack([gaia["RA"] * cos0, gaia["DEC"]])))

        files = sorted(glob.glob(os.path.join(args.exposure_dir, "position_corrected_*.fits")))
        if args.limit:
            files = files[:args.limit]
        print(f"[exposures] {len(files)} files")
        with mp.get_context("fork").Pool(args.nproc) as pool:
            rows = pool.map(per_exposure, files, chunksize=8)
        rows = [r for r in rows if r is not None]
        tab = np.array(rows, dtype=[("expnum", "i8"), ("mjd", "f8"), ("n", "i8"), ("dra", "f8"), ("ddec", "f8"),
                                     ("dra_err", "f8"), ("ddec_err", "f8"), ("dra_std", "f8"), ("ddec_std", "f8")])
        np.save(args.cache, tab)
        print(f"[cache] wrote {len(tab)} exposures to {args.cache}")
    else:
        tab = np.load(args.cache)
        print(f"[cache] loaded {len(tab)} exposures from {args.cache}")

    tab = tab[np.argsort(tab["mjd"])]
    tyr = 2016.0 + (tab["mjd"] - GAIA_REF_MJD) / YEAR
    print(f"[median matched stars/exposure] {np.median(tab['n']):.0f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for col, (key, label) in enumerate((("dra", "RA*cos(dec)"), ("ddec", "Dec"))):
        y, err = tab[key], np.maximum(tab[key + "_err"], 1e-3)
        p, perr, chi2 = fit_line(tyr - 2016.0, y, err)
        print(f"{label:12s}: offset at J2016 = {p[0]:+.4f} +/- {perr[0]:.4f} mas ; slope = {p[1]:+.4f} +/- {perr[1]:.4f} mas/yr "
              f"(reduced chi2 {chi2:.1f}); median = {np.median(y):+.4f} mas; scatter (robust) = {robust_std(y):.3f} mas")
        ax = axes[0, col]
        ax.errorbar(tyr, y, yerr=err, fmt=".", color="darkorange", alpha=0.5, ms=4, lw=0.5)
        xx = np.array([tyr.min(), tyr.max()])
        ax.plot(xx, p[0] + p[1] * (xx - 2016.0), color="firebrick", lw=2,
                label=f"slope {p[1]:+.3f} +/- {perr[1]:.3f} mas/yr")
        ax.axhline(0.0, color="k", ls="--", lw=1)
        lim = 5 * robust_std(y)
        ax.set_ylim(np.median(y) - lim, np.median(y) + lim)
        ax.set_xlabel("year")
        ax.set_ylabel(f"GPR - Gaia, {label} (mas), per-exposure median")
        ax.set_title(f"{label}: {len(tab)} exposures")
        ax.legend(fontsize=9)
        ax.grid()

    pixels = set(int(h) for h in np.load(args.healpix_npy))
    dpm = pm_pulls(pixels, args, ref_only)
    for col, (label, dv, pull) in enumerate((("pmra", dpm[0], dpm[2]), ("pmdec", dpm[1], dpm[3]))):
        ax = axes[1, col]
        rng = 8.0
        bins = np.linspace(-rng, rng, 121)
        rstd, med = robust_std(pull), np.median(pull)
        ax.hist(pull, bins=bins, density=True, color="steelblue", alpha=0.7,
                label=f"mean PM offset (ours - Gaia) = {np.mean(dv[np.abs(dv - np.median(dv)) < 5 * robust_std(dv)]):+.3f} mas/yr\n"
                      f"median pull {med:+.2f}, robust std {rstd:.2f}, n={len(pull)}")
        x = np.linspace(-rng, rng, 400)
        ax.plot(x, norm.pdf(x, 0, 1), "k--", lw=2, label=r"$\mathcal{N}(0,1)$")
        ax.set_yscale("log"); ax.set_ylim(1e-5, 1)
        ax.set_xlabel(f"{label} pull " + r"$(\mu_{our}-\mu_{gaia})/\sqrt{\sigma_{our}^2+\sigma_{gaia}^2}$")
        ax.set_ylabel("density")
        ax.legend(fontsize=9)
        ax.grid()

    fig.suptitle("Sculptor 8 degree: per-exposure GPR position - Gaia (propagated to exposure epoch)" + (", GPR reference stars only" if ref_only else ""), fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")
