"""Precision plot (measured sigma_pm vs magnitude) for an explicit list of
healpixels, rather than everything within a disc radius. Same hexbin +
forecast-curve overlay as healpix_precision_plot.py, useful when only a
handful of specific pixels (e.g. fully-covered ones) should be shown.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib.ticker import ScalarFormatter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from healpix_region_data import load_coadd, load_movers

DEFAULT_FORECAST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "attic", "pm_plot.npz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healpix", type=int, nargs="+", required=True, help="Explicit healpix indices to plot")
    parser.add_argument("--pmcatalog-dir", default="/data8/shared/decampm/PMSculptor/PMCatalog/")
    parser.add_argument("--coadd-dir", default="/data8/shared/decampm/COADDS/")
    parser.add_argument("--forecast", default=DEFAULT_FORECAST)
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--match-arcsec", type=float, default=0.1)
    # COADDS/*.fits is a DELVE quick-object catalog: RA/DEC (not ALPHAWIN_J2000/
    # DELTAWIN_J2000) and EXTENDED_CLASS_<band> (0=star..3=galaxy, not EXT_MASH),
    # so match_to_coadd's hardcoded DES-Gold column names don't apply here --
    # matching is done directly below instead.
    parser.add_argument("--mag-col", default="MAG_PSF_I")
    parser.add_argument("--class-col", default="EXTENDED_CLASS_I")
    parser.add_argument("--pm-pattern", default="real_PM_hp*.fits")
    parser.add_argument("--coadd-pattern", default="*.fits")
    parser.add_argument("-o", "--out", default="precision_plot_selected_healpix.png")
    parser.add_argument("--title", default="Sculptor 8 degree radius")
    args = parser.parse_args()

    pixels = set(args.healpix)
    movers = load_movers(args.pmcatalog_dir, args.nside, pixels, pattern=args.pm_pattern)
    coadd = load_coadd(args.coadd_dir, pixels, pattern=args.coadd_pattern)
    coadd = coadd[(coadd[args.mag_col] < 90) & (coadd[args.class_col] >= 0)]

    mover_coords = SkyCoord(ra=movers["ra"] * u.degree, dec=movers["dec"] * u.degree)
    coadd_coords = SkyCoord(ra=coadd["RA"] * u.degree, dec=coadd["DEC"] * u.degree)
    idx, d2d, _ = mover_coords.match_to_catalog_sky(coadd_coords)
    close = d2d.arcsecond < args.match_arcsec
    matched = rfn.merge_arrays([movers[close], coadd[idx[close]]], flatten=True, usemask=False)
    print(f"[match] {len(movers)} movers, {len(matched)} matched within {args.match_arcsec} arcsec")

    stars_matched = matched[matched[args.class_col] <= 1]
    sigma_pm = 1000 * np.sqrt(stars_matched["c_vxvx"])

    forecast = np.load(args.forecast, allow_pickle=True)
    imag = forecast["imag"]
    med_dec = forecast["med_dec"]
    med_lsst = forecast["med_lsst"]
    dr5_G = forecast["dr5_G"]
    Gi = forecast["Gi"]
    dr5_sigpm = forecast["dr5_sigpm"]
    dr3_sigpm = forecast["dr3_sigpm"]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hexbin(
        stars_matched[args.mag_col], sigma_pm,
        extent=(16, 24, np.log10(0.02), np.log10(50.0)),
        yscale="log", mincnt=5, cmap="managua", gridsize=100, edgecolors="none",
    )

    ax.semilogy(imag, med_dec[:, 0], "c-", lw=3)
    ax.semilogy(imag, med_lsst[:, 1], "m--", lw=2)
    ax.semilogy(imag, med_dec[:, 1], "m-", lw=3)
    ax.semilogy(imag, med_dec[:, 10], "k-", lw=3)
    ax.semilogy(dr5_G - Gi, dr5_sigpm, "g:", lw=5)
    ax.semilogy((dr5_G - Gi)[-1], dr5_sigpm[-1], "g*", ms=16)
    ax.semilogy(dr5_G - Gi, dr3_sigpm, "r:", lw=5)
    ax.semilogy((dr5_G - Gi)[-1], dr3_sigpm[-1], "r*", ms=16)

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(16, 24)
    ax.set_ylim(0.02, 50)
    ax.set_xlabel(f"{args.mag_col} magnitude", fontsize=20)
    ax.set_ylabel(r"$\sigma_{\mu_{\alpha*}}$ (mas/yr)", fontsize=20)
    ax.set_title(args.title, fontsize=14)
    ax.grid()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
