"""Per-DETECTION (single-exposure) astrometric residual vs Gaia magnitude.

Checks whether the magnitude-dependent bias seen in the star-level position
and PM comparisons against Gaia (dev/plot_gaia_position_error_pull.py,
dev/plot_gaia_pm_error_pull.py) is already present in individual detections,
BEFORE the multi-epoch PM fit (highpm.pmfit.fit5d/singleFit) averages them
together. If the same magnitude trend shows up here, the bias enters at (or
before) the per-exposure measurement/GPR stage, not in the PM-fit/covariance
machinery.

No per-detection residual is persisted anywhere in the pipeline outputs
(highpm.fits_writer only writes summed chisqTotal at the star level, and the
*_detections extension is a pure idx/detections/clipped join table -- see
highpm/fits_writer.py:193-231). So this script reconstructs the residual
itself: for every non-clipped detection that entered a star's final fit,

    resid = observed(XI, ETA) - model(x0, y0, vx, vy, pi, [color])

using that star's own fitted parameters from modest1_movers (highpm/pmfit.py
singleFit's model, mirrored exactly here: highpm/pmfit.py:88-101, 335-337,
424-428) and the detection's own XI/ETA/MJD/PAR_XI/PAR_ETA/[DXI_DCOLOR,
DETA_DCOLOR] from the HealpixDetectionCatalog file (cleaned_detections_hp*).

Detections are binned by their STAR's Gaia G magnitude (not an uncalibrated
per-detection instrumental flux) so bins are directly comparable to the
star-level plots -- every detection of a star shares that star's G.
"""

import argparse
import os
import sys

import fitsio
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy import units as u
from astropy.coordinates import SkyCoord

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from healpix_region_data import load_coadd, _hp_index

DEGREE = 3600.0          # arcsec per degree
DAY = 1.0 / 365.2425     # years per day


def robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def binned_stats(x, y, w, bins):
    edges = np.linspace(*bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(x, edges) - 1
    wmean = np.full(len(centers), np.nan)
    wmean_err = np.full(len(centers), np.nan)
    rstd = np.full(len(centers), np.nan)
    n = np.zeros(len(centers), dtype=int)
    for i in range(len(centers)):
        sel = idx == i
        n[i] = sel.sum()
        if n[i] >= 50:
            wsel = w[sel]
            wmean[i] = np.sum(wsel * y[sel]) / np.sum(wsel)
            wmean_err[i] = 1.0 / np.sqrt(np.sum(wsel))
            rstd[i] = robust_std(y[sel])
    return centers, wmean, wmean_err, rstd, n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--healpix", type=int, nargs="+", required=True)
    parser.add_argument("--pmcatalog-dir", default="/data8/shared/decampm/PMSculptor_8deg_garyb/PMCatalog/")
    parser.add_argument("--detections-dir",
                         default="/data8/shared/decampm/PMSculptor_8deg_garyb/HealpixDetectionCatalog/")
    parser.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia/")
    parser.add_argument("--pm-pattern", default="real_PM_hp*.fits")
    parser.add_argument("--detections-pattern", default="cleaned_detections_hp*.fits")
    parser.add_argument("--gaia-pattern", default="GaiaSource_*.fits")
    parser.add_argument("--mjd-ref", type=float, default=57388.0)
    parser.add_argument("--match-arcsec", type=float, default=0.3)
    parser.add_argument("--max-ruwe", type=float, default=1.4)
    parser.add_argument("-o", "--out", default="detection_residual_vs_mag.png")
    parser.add_argument("--title", default="Sculptor 8 degree radius (46 GPR2-complete healpix): "
                                            "per-detection residuals")
    args = parser.parse_args()

    pixels = set(args.healpix)

    # Star-level Gaia match (same recipe as the other scripts) to get one G
    # magnitude per star, shared by all of that star's detections.
    movers_parts = []
    resid_parts = []
    for h in pixels:
        pm_file = os.path.join(args.pmcatalog_dir, f"real_PM_hp{h:05d}.fits")
        det_file = os.path.join(args.detections_dir, f"cleaned_detections_hp{h:05d}.fits")
        if not (os.path.exists(pm_file) and os.path.exists(det_file)):
            print(f"[skip] hp{h:05d}: missing pm_file or det_file")
            continue

        movers = fitsio.read(pm_file, ext="modest1_movers")
        dets = fitsio.read(pm_file, ext="modest1_detections")
        dets = dets[~dets["clipped"]]

        cat = fitsio.read(det_file, columns=["XI", "ETA", "MJD", "PAR_XI", "PAR_ETA",
                                              "DXI_DCOLOR", "DETA_DCOLOR"])

        rows = cat[dets["detections"]]
        star = movers[dets["idx"]]

        t = (rows["MJD"] - args.mjd_ref) * DAY
        x0 = star["xi"] * DEGREE
        y0 = star["eta"] * DEGREE
        vx = star["pmra"] / 1000.0
        vy = star["pmdec"] / 1000.0
        pi = star["parallax"]

        pred_x = x0 + vx * t + pi * rows["PAR_XI"]
        pred_y = y0 + vy * t + pi * rows["PAR_ETA"]

        has_color = star["color_err"] > 0
        pred_x[has_color] += star["color"][has_color] * rows["DXI_DCOLOR"][has_color] * DEGREE
        pred_y[has_color] += star["color"][has_color] * rows["DETA_DCOLOR"][has_color] * DEGREE

        obs_x = rows["XI"] * DEGREE
        obs_y = rows["ETA"] * DEGREE

        resid_x_mas = (obs_x - pred_x) * 1000.0
        resid_y_mas = (obs_y - pred_y) * 1000.0

        out = np.zeros(len(dets), dtype=[
            ("star_idx", "i8"), ("star_ra", "f8"), ("star_dec", "f8"),
            ("resid_x_mas", "f8"), ("resid_y_mas", "f8")])
        out["star_idx"] = dets["idx"].astype(np.int64) + (int(h) * 10_000_000)  # globally-unique-enough star key
        out["star_ra"] = star["ra"]
        out["star_dec"] = star["dec"]
        out["resid_x_mas"] = resid_x_mas
        out["resid_y_mas"] = resid_y_mas
        resid_parts.append(out)

        uniq_idx = np.unique(dets["idx"])
        star_out = np.zeros(len(uniq_idx), dtype=[("star_idx", "i8"), ("ra", "f8"), ("dec", "f8")])
        star_out["star_idx"] = uniq_idx.astype(np.int64) + (int(h) * 10_000_000)
        star_out["ra"] = movers["ra"][uniq_idx]
        star_out["dec"] = movers["dec"][uniq_idx]
        movers_parts.append(star_out)

    resid = np.concatenate(resid_parts)
    star_pos = np.concatenate(movers_parts)
    print(f"[detections] {len(resid)} non-clipped detections from {len(pixels)} healpix, "
          f"{len(star_pos)} distinct stars")

    gaia = load_coadd(args.gaia_dir, pixels, pattern=args.gaia_pattern)
    gaia = gaia[~np.isnan(gaia["PMRA"])]
    two_param = (gaia["PMRA"] == 0) & (gaia["PMDEC"] == 0) & (gaia["PARALLAX"] == 0)
    gaia = gaia[~two_param]
    if args.max_ruwe is not None:
        gaia = gaia[gaia["RUWE"] <= args.max_ruwe]

    star_coords = SkyCoord(ra=star_pos["ra"] * u.degree, dec=star_pos["dec"] * u.degree)
    gaia_coords = SkyCoord(ra=gaia["RA"] * u.degree, dec=gaia["DEC"] * u.degree)
    idx, d2d, _ = star_coords.match_to_catalog_sky(gaia_coords)
    close = d2d.arcsecond < args.match_arcsec
    star_gmag = np.full(len(star_pos), np.nan)
    star_gmag[close] = gaia["PHOT_G_MEAN_MAG"][idx[close]]
    gmag_by_star = dict(zip(star_pos["star_idx"], star_gmag))

    resid_gmag = np.array([gmag_by_star.get(s, np.nan) for s in resid["star_idx"]])
    keep = np.isfinite(resid_gmag)
    resid = resid[keep]
    resid_gmag = resid_gmag[keep]
    print(f"[match] {len(resid)} detections with a Gaia-matched star magnitude")

    weight = np.ones(len(resid))  # unweighted robust binning; formal per-detection
                                   # errors aren't reconstructed here, only the trend shape

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, resid_c, label in zip(axes[0], (resid["resid_x_mas"], resid["resid_y_mas"]), ("xi", "eta")):
        rng = 5 * robust_std(resid_c)
        bins = np.linspace(-rng, rng, 121)
        ax.hist(resid_c, bins=bins, color="steelblue", alpha=0.7)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_xlabel(f"{label} residual (mas): observed - PM-fit-predicted")
        ax.set_ylabel("count")
        ax.set_title(f"{label}: median={np.median(resid_c):+.3f} mas, robust std={robust_std(resid_c):.2f} mas, "
                     f"n={len(resid_c)}")
        ax.grid()

    for ax, resid_c, label in zip(axes[1], (resid["resid_x_mas"], resid["resid_y_mas"]), ("xi", "eta")):
        centers, wmean, wmean_err, rstd, n = binned_stats(resid_gmag, resid_c, weight, (12, 21, 37))
        ax.errorbar(centers, wmean, yerr=wmean_err, fmt="o-", color="darkorange", label="mean residual")
        ax.axhline(0.0, color="k", ls="--", lw=1.5)
        ax.set_xlabel("Gaia G magnitude (of the detection's own star)")
        ax.set_ylabel(f"{label} mean detection residual (mas)")
        ax.legend(fontsize=9)
        ax.grid()

    fig.suptitle(args.title, fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
