"""Per-exposure GPR - Gaia offset, split by the GPR `ref` flag.

Reads the RAW GPR files (bypassing position_correction/detection packing) and,
for each exposure, matches GPR new_rd positions to Gaia (propagated to the
exposure epoch with Gaia's PM and the exposure's parallax factors), separately
for stars GPR used as astrometric references (ref=True) and those it did not
(ref=False). If the time-dependent drift seen in
dev/plot_gpr_vs_gaia_per_exposure.py exists only for non-reference stars, the
references are pinned correctly and the drift is in the GPR model away from
them; if it exists for reference stars too, the reference positions
themselves are being propagated to the exposure epoch inconsistently with the
Gaia PMs used here.
"""

import argparse
import glob
import multiprocessing as mp
import os
import re
import sys

import fitsio
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from healpix_region_data import load_coadd
from plot_gpr_vs_gaia_per_exposure import fit_line, robust_std, GAIA_REF_MJD, YEAR, MAS
from highpm.position_correction import getGPRFile

_G = {}


def stats(dra, ddec):
    if len(dra) < 20:
        return (len(dra), np.nan, np.nan, np.nan, np.nan)
    for _ in range(2):
        keep = (np.abs(dra - np.median(dra)) < 4 * robust_std(dra)) & (np.abs(ddec - np.median(ddec)) < 4 * robust_std(ddec))
        dra, ddec = dra[keep], ddec[keep]
    n = len(dra)
    return (n, np.median(dra), np.median(ddec),
            1.2533 * robust_std(dra) / np.sqrt(n), 1.2533 * robust_std(ddec) / np.sqrt(n))


def per_exposure(path):
    expnum = int(re.search(r"(\d{8})\.fits$", path).group(1))
    gprfile = getGPRFile(expnum, _G["gprdirs"])
    if gprfile is None:
        return None
    try:
        meta = fitsio.read(path, ext="DATA", columns=["MJD", "PAR_XI", "PAR_ETA"], rows=[0])
        g = fitsio.read(gprfile, ext=1, columns=["ref", "new_rd", "cov_model"])
    except Exception:
        return None
    mjd, par_xi, par_eta = meta["MJD"][0], meta["PAR_XI"][0], meta["PAR_ETA"][0]
    if mjd < 0:
        return None
    g = g[(g["cov_model"][:, 0, 0] > 0) & (g["cov_model"][:, 1, 1] > 0)]
    ra, dec = g["new_rd"][:, 0], g["new_rd"][:, 1]
    dist, j = _G["tree"].query(np.column_stack([ra * _G["cos0"], dec]), distance_upper_bound=_G["cand_deg"])
    ok = np.isfinite(dist)
    if ok.sum() < 20:
        return None
    ra, dec, j, isref = ra[ok], dec[ok], j[ok], g["ref"][ok].astype(bool)
    gaia = _G["gaia"]
    dt = (mjd - GAIA_REF_MJD) / YEAR
    cosd = np.cos(np.radians(gaia["DEC"][j]))
    g_ra = gaia["RA"][j] + (gaia["PMRA"][j] * dt + gaia["PARALLAX"][j] * par_xi) / MAS / cosd
    g_dec = gaia["DEC"][j] + (gaia["PMDEC"][j] * dt + gaia["PARALLAX"][j] * par_eta) / MAS
    dra = (ra - g_ra) * cosd * MAS
    ddec = (dec - g_dec) * MAS
    close = np.hypot(dra, ddec) < _G["match_mas"]
    out = [mjd]
    for sel in (close & isref, close & ~isref):
        out.extend(stats(dra[sel], ddec[sel]))
    return tuple(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--healpix-npy", default="/data8/shared/decampm/PMSculptor_8deg_garyb/pmsculptor8deg_healpix.npy")
    parser.add_argument("--exposure-dir",
                         default="/data8/shared/decampm/PMSculptor_8deg_garyb/PositionCorrectedExposureCatalog/",
                         help="Used only to define the exposure list and read each exposure's MJD/parallax factors")
    parser.add_argument("--gpr-dirs", nargs="+", default=[f"/data8/shared/decampm/GPR2/CAT/{b}/" for b in "griz"])
    parser.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia/")
    parser.add_argument("--gmag-range", type=float, nargs=2, default=(15.0, 19.0))
    parser.add_argument("--max-ruwe", type=float, default=1.4)
    parser.add_argument("--match-mas", type=float, default=300.0)
    parser.add_argument("--candidate-arcsec", type=float, default=2.0)
    parser.add_argument("--nproc", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache", default="gpr_ref_vs_nonref.npy")
    parser.add_argument("-o", "--out", default="gpr_ref_vs_nonref.png")
    args = parser.parse_args()

    names = ["mjd"] + [f"{s}_{k}" for s in ("ref", "non") for k in ("n", "dra", "ddec", "dra_err", "ddec_err")]
    if not os.path.exists(args.cache):
        pixels = set(int(h) for h in np.load(args.healpix_npy))
        gaia = load_coadd(args.gaia_dir, pixels, pattern="GaiaSource_*.fits")
        gaia = gaia[np.isfinite(gaia["PMRA"]) & ~((gaia["PMRA"] == 0) & (gaia["PMDEC"] == 0) & (gaia["PARALLAX"] == 0))]
        gmin, gmax = args.gmag_range
        gaia = gaia[(gaia["RUWE"] <= args.max_ruwe) & (gaia["PHOT_G_MEAN_MAG"] >= gmin) & (gaia["PHOT_G_MEAN_MAG"] <= gmax)]
        print(f"[gaia] {len(gaia)} reference stars, G in [{gmin}, {gmax}], RUWE <= {args.max_ruwe}")
        cos0 = np.cos(np.radians(np.median(gaia["DEC"])))
        _G.update(gaia=gaia, cos0=cos0, cand_deg=args.candidate_arcsec / 3600.0, match_mas=args.match_mas,
                  gprdirs=args.gpr_dirs, tree=cKDTree(np.column_stack([gaia["RA"] * cos0, gaia["DEC"]])))
        files = sorted(glob.glob(os.path.join(args.exposure_dir, "position_corrected_*.fits")))
        if args.limit:
            files = files[:args.limit]
        with mp.get_context("fork").Pool(args.nproc) as pool:
            rows = [r for r in pool.map(per_exposure, files, chunksize=8) if r is not None]
        tab = np.array(rows, dtype=[(n, "i8" if n.endswith("_n") else "f8") for n in names])
        np.save(args.cache, tab)
        print(f"[cache] wrote {len(tab)} exposures to {args.cache}")
    else:
        tab = np.load(args.cache)

    tab = tab[np.argsort(tab["mjd"])]
    tyr = 2016.0 + (tab["mjd"] - GAIA_REF_MJD) / YEAR
    print(f"[median matched stars/exposure] ref={np.median(tab['ref_n']):.0f}, non-ref={np.median(tab['non_n']):.0f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for col, (key, label) in enumerate((("dra", "RA*cos(dec)"), ("ddec", "Dec"))):
        ax = axes[0, col]
        axh = axes[1, col]
        for tag, color, name in (("ref", "firebrick", "GPR reference stars (ref=True)"),
                                 ("non", "steelblue", "non-reference stars (ref=False)")):
            y, err = tab[f"{tag}_{key}"], np.maximum(tab[f"{tag}_{key}_err"], 1e-3)
            good = np.isfinite(y) & np.isfinite(err)
            p, perr, chi2 = fit_line(tyr[good] - 2016.0, y[good], err[good])
            print(f"{label:12s} {tag:3s}: offset at J2016 = {p[0]:+.4f} +/- {perr[0]:.4f} mas ; "
                  f"slope = {p[1]:+.4f} +/- {perr[1]:.4f} mas/yr (n_exp={good.sum()}, reduced chi2 {chi2:.1f}); "
                  f"median = {np.median(y[good]):+.4f} mas")
            ax.plot(tyr[good], y[good], ".", color=color, alpha=0.3, ms=4)
            xx = np.array([tyr.min(), tyr.max()])
            ax.plot(xx, p[0] + p[1] * (xx - 2016.0), color=color, lw=2.5,
                    label=f"{name}: slope {p[1]:+.3f} +/- {perr[1]:.3f} mas/yr")
            axh.hist(y[good], bins=np.linspace(-4, 4, 100), color=color, alpha=0.5, label=name)
        ax.axhline(0.0, color="k", ls="--", lw=1)
        ax.set_ylim(-3, 3.5)
        ax.set_xlabel("year")
        ax.set_ylabel(f"GPR - Gaia, {label} (mas), per-exposure median")
        ax.legend(fontsize=8)
        ax.grid()
        axh.axvline(0.0, color="k", ls="--", lw=1)
        axh.set_xlabel(f"per-exposure median GPR - Gaia, {label} (mas)")
        axh.set_ylabel("exposures")
        axh.legend(fontsize=8)
        axh.grid()

    fig.suptitle("Sculptor 8 degree: per-exposure raw GPR position - Gaia, reference vs non-reference stars", fontsize=13)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")
