"""Compare our formal proper-motion errors (from modest1_movers, i.e.
confirmed slow movers) against Gaia's formal PMRA_ERROR/PMDEC_ERROR for the
same stars, for an explicit list of healpixels.

Loads deduped modest1_movers from real_PM_hp*.fits, matches them to Gaia
sources (GaiaSource_hpXXXXX.fits, same nside pixelization) by position, and
makes:
  (1) sigma_pm (mas/yr) vs Gaia G magnitude: our hexbin density + Gaia's
      running-median error curve + DECam/LSST/Gaia forecast curves.
  (2) direct 1:1 comparison hexbin of our sigma_pm vs Gaia's sigma_pm for
      the matched stars.
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
from matplotlib.ticker import ScalarFormatter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from healpix_region_data import load_coadd, load_movers

DEFAULT_FORECAST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "attic", "pm_plot.npz")


def running_median(x, y, bins):
    edges = np.linspace(*bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(x, edges) - 1
    med = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        sel = idx == i
        if sel.sum() >= 5:
            med[i] = np.median(y[sel])
    return centers, med


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healpix", type=int, nargs="+", required=True, help="Explicit healpix indices to plot")
    parser.add_argument("--pmcatalog-dir", default="/data8/shared/decampm/PMSculptor_8deg_garyb/PMCatalog/")
    parser.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia/")
    parser.add_argument("--forecast", default=DEFAULT_FORECAST)
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--match-arcsec", type=float, default=0.3)
    parser.add_argument("--pm-pattern", default="real_PM_hp*.fits")
    parser.add_argument("--gaia-pattern", default="GaiaSource_*.fits")
    parser.add_argument("-o", "--out", default="gaia_pm_error_comparison.png")
    parser.add_argument("--title", default="Sculptor 8 degree radius (46 GPR2-complete healpix)")
    args = parser.parse_args()

    pixels = set(args.healpix)
    movers = load_movers(args.pmcatalog_dir, args.nside, pixels,
                          pattern=args.pm_pattern, exts=("modest1_movers",))
    gaia = load_coadd(args.gaia_dir, pixels, pattern=args.gaia_pattern)
    gaia = gaia[~np.isnan(gaia["PMRA"])]
    print(f"{len(gaia)} Gaia sources with valid PM in this region")

    mover_coords = SkyCoord(ra=movers["ra"] * u.degree, dec=movers["dec"] * u.degree)
    gaia_coords = SkyCoord(ra=gaia["RA"] * u.degree, dec=gaia["DEC"] * u.degree)
    idx, d2d, _ = mover_coords.match_to_catalog_sky(gaia_coords)
    close = d2d.arcsecond < args.match_arcsec
    matched = rfn.merge_arrays([movers[close], gaia[idx[close]]], flatten=True, usemask=False)
    print(f"[match] {len(movers)} modest1 movers, {len(matched)} matched to Gaia within {args.match_arcsec} arcsec")

    our_sigma_pmra = 1000 * np.sqrt(matched["c_vxvx"])   # mas/yr
    our_sigma_pmdec = 1000 * np.sqrt(matched["c_vyvy"])  # mas/yr
    our_sigma_pm = np.hypot(our_sigma_pmra, our_sigma_pmdec) / np.sqrt(2)

    gaia_sigma_pmra = matched["PMRA_ERROR"]    # already mas/yr
    gaia_sigma_pmdec = matched["PMDEC_ERROR"]  # already mas/yr
    gaia_sigma_pm = np.hypot(gaia_sigma_pmra, gaia_sigma_pmdec) / np.sqrt(2)

    gmag = matched["PHOT_G_MEAN_MAG"]

    forecast = np.load(args.forecast, allow_pickle=True)
    imag = forecast["imag"]
    med_dec = forecast["med_dec"]
    med_lsst = forecast["med_lsst"]
    dr5_G = forecast["dr5_G"]
    Gi = forecast["Gi"]
    dr5_sigpm = forecast["dr5_sigpm"]
    dr3_sigpm = forecast["dr3_sigpm"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: our formal error density vs Gaia G mag, with Gaia's own
    # running-median formal error and the static forecast curves overlaid.
    ax = axes[0]
    pos = our_sigma_pm > 0
    ax.hexbin(
        gmag[pos], our_sigma_pm[pos],
        extent=(12, 21, np.log10(0.01), np.log10(50.0)),
        yscale="log", mincnt=5, cmap="managua", gridsize=100, edgecolors="none",
        label="_nolegend_",
    )
    gc, gmed = running_median(gmag, gaia_sigma_pm, (12, 21, 60))
    ax.semilogy(gc, gmed, "-", color="orange", lw=3, label="Gaia (median formal error)")

    ax.semilogy(imag, med_dec[:, 0], "c-", lw=2, label="DECam")
    ax.semilogy(imag, med_lsst[:, 1], "m--", lw=2, label="LSST 1yr")
    ax.semilogy(imag, med_dec[:, 1], "m-", lw=2, label="LSST 1yr + DECam")
    ax.semilogy(imag, med_dec[:, 10], "k-", lw=2, label="LSST 10yr + DECam")
    ax.semilogy(dr5_G - Gi, dr5_sigpm, "g:", lw=4, label="Gaia DR5 (forecast)")
    ax.semilogy(dr5_G - Gi, dr3_sigpm, "r:", lw=4, label="Gaia DR3 (forecast)")

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(12, 21)
    ax.set_ylim(0.01, 50)
    ax.set_xlabel("Gaia G magnitude", fontsize=16)
    ax.set_ylabel(r"$\sigma_{\mu}$ (mas/yr)", fontsize=16)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid()

    # Panel 2: direct 1:1 comparison of our formal error vs Gaia's, for the
    # same matched stars.
    ax = axes[1]
    lo, hi = 1e-2, 1e2
    finite = (gaia_sigma_pm > 0) & (our_sigma_pm > 0) & np.isfinite(gaia_sigma_pm) & np.isfinite(our_sigma_pm)
    hb = ax.hexbin(
        gaia_sigma_pm[finite], our_sigma_pm[finite],
        xscale="log", yscale="log",
        extent=(np.log10(lo), np.log10(hi), np.log10(lo), np.log10(hi)),
        mincnt=1, cmap="managua", gridsize=100, edgecolors="none", bins="log",
    )
    fig.colorbar(hb, ax=ax, label="log$_{10}$(stars / bin)")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Gaia $\sigma_{\mu}$ (mas/yr)", fontsize=16)
    ax.set_ylabel(r"Our $\sigma_{\mu}$ (mas/yr)", fontsize=16)
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    ax.grid()

    fig.suptitle(args.title, fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
