"""Per-DETECTION (single-exposure) astrometric residual vs time (MJD).

Same per-detection residual reconstruction as dev/plot_detection_residual_vs_mag.py
(see that script's docstring for the full derivation), but binned by the
detection's own MJD instead of its star's Gaia G magnitude. This asks a
different question than the magnitude trend: does the systematic offset seen
in individual exposures drift over the multi-year baseline (e.g. a slow GPR
solution drift, a camera-shape/optics change, or a Gaia-frame proper-motion
mismatch that grows with |t - mjd_ref|), or is it a constant, time-independent
zero-point?

A residual that is flat in G but trends with time points at something in the
per-exposure astrometric solution (GPR) itself rather than a magnitude-
dependent centroiding bias.

Produces:
  (1) Mean detection residual (xi, eta) vs MJD, binned, with a linear fit
      overlaid to quantify any drift (mas/yr).
  (2) The same vs time, restricted to a single magnitude range (--gmag-range)
      so a magnitude-dependent effect doesn't masquerade as -- or hide -- a
      time trend.
"""

import argparse
import os
import sys

import fitsio
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from healpix_region_data import load_coadd

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


def weighted_linfit(x, y, w):
    """Weighted least-squares slope/intercept, returns (slope, slope_err, intercept)."""
    W = np.sum(w)
    xbar = np.sum(w * x) / W
    ybar = np.sum(w * y) / W
    sxx = np.sum(w * (x - xbar) ** 2)
    sxy = np.sum(w * (x - xbar) * (y - ybar))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    slope_err = np.sqrt(1.0 / sxx)
    return slope, slope_err, intercept


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
    parser.add_argument("--gmag-range", type=float, nargs=2, default=None,
                         help="Restrict panel (2) to detections whose star has GMAG_MIN <= G <= GMAG_MAX")
    parser.add_argument("--time-bins", type=int, default=40, help="Number of MJD bins")
    parser.add_argument("-o", "--out", default="detection_residual_vs_time.png")
    parser.add_argument("--title", default="Sculptor 8 degree radius (46 GPR2-complete healpix): "
                                            "per-detection residuals vs time")
    args = parser.parse_args()

    pixels = set(args.healpix)

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
            ("star_idx", "i8"), ("star_ra", "f8"), ("star_dec", "f8"), ("mjd", "f8"),
            ("resid_x_mas", "f8"), ("resid_y_mas", "f8")])
        out["star_idx"] = dets["idx"].astype(np.int64) + (int(h) * 10_000_000)  # globally-unique-enough star key
        out["star_ra"] = star["ra"]
        out["star_dec"] = star["dec"]
        out["mjd"] = rows["MJD"]
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
    print(f"[mjd] range {resid['mjd'].min():.1f} - {resid['mjd'].max():.1f} "
          f"({(resid['mjd'].max() - resid['mjd'].min()) / 365.25:.2f} yr baseline)")

    star_gmag = None
    if args.gmag_range is not None:
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

    weight = np.ones(len(resid))  # unweighted robust binning; formal per-detection
                                   # errors aren't reconstructed here, only the trend shape

    tmin, tmax = resid["mjd"].min(), resid["mjd"].max()
    tbins = (tmin, tmax, args.time_bins + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for ax, resid_c, label in zip(axes[0], (resid["resid_x_mas"], resid["resid_y_mas"]), ("xi", "eta")):
        centers, wmean, wmean_err, rstd, n = binned_stats(resid["mjd"], resid_c, weight, tbins)
        good = np.isfinite(wmean)
        slope, slope_err, intercept = weighted_linfit(
            centers[good], wmean[good], 1.0 / wmean_err[good] ** 2)
        slope_yr = slope * 365.2425
        slope_err_yr = slope_err * 365.2425

        ax.errorbar(centers, wmean, yerr=wmean_err, fmt="o", color="darkorange", label="mean residual")
        xx = np.linspace(tmin, tmax, 2)
        ax.plot(xx, intercept + slope * xx, color="firebrick", ls="--", lw=1.5,
                label=f"linear fit: {slope_yr:+.3f} +/- {slope_err_yr:.3f} mas/yr")
        ax.axhline(0.0, color="k", ls=":", lw=1.0)
        ax.set_xlabel("MJD")
        ax.set_ylabel(f"{label} mean detection residual (mas)")
        ax.set_title(f"{label}: all magnitudes")
        ax.legend(fontsize=9)
        ax.grid()

    for ax, resid_c, label in zip(axes[1], (resid["resid_x_mas"], resid["resid_y_mas"]), ("xi", "eta")):
        if args.gmag_range is not None:
            gmin, gmax = args.gmag_range
            sel = (resid_gmag >= gmin) & (resid_gmag <= gmax)
            centers, wmean, wmean_err, rstd, n = binned_stats(resid["mjd"][sel], resid_c[sel], weight[sel], tbins)
            title_suffix = f"{gmin:.1f} <= G <= {gmax:.1f}, n={sel.sum()}"
        else:
            centers, wmean, wmean_err, rstd, n = binned_stats(resid["mjd"], resid_c, weight, tbins)
            title_suffix = "all magnitudes (repeat of top row)"
        ax.errorbar(centers, wmean, yerr=wmean_err, fmt="o-", color="steelblue", label="mean residual")
        ax.axhline(0.0, color="k", ls="--", lw=1.5)
        ax.set_xlabel("MJD")
        ax.set_ylabel(f"{label} mean detection residual (mas)")
        ax.set_title(f"{label}: {title_suffix}")
        ax.legend(fontsize=9)
        ax.grid()

    fig.suptitle(args.title, fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
