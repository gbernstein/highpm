"""Precision plot: measured PM precision (sigma_pm vs magnitude) for stars
within radius degrees of a center on the sky, as a hexbin density, overlaid
on the static DECam/LSST/Gaia forecast curves from attic/pm_plot.npz.

Pulls only the PMCatalog/coadd_skims healpixel files needed for the region.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from healpix_region_data import disc_pixels, load_coadd, load_movers, match_to_coadd

DEFAULT_FORECAST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "attic", "pm_plot.npz")


def measured_precision(
    ra0, dec0, radius, pmcatalog_dir, coadd_dir, nside=32, match_arcsec=0.1,
    pm_pattern="*.fits", coadd_pattern="*.fits",
):
    pixels = disc_pixels(nside, ra0, dec0, radius)
    movers = load_movers(pmcatalog_dir, nside, pixels, pattern=pm_pattern)
    coadd = load_coadd(coadd_dir, pixels, pattern=coadd_pattern)
    matched = match_to_coadd(movers, coadd, match_arcsec)
    return matched[matched["EXT_MASH"] <= 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ra0", type=float, help="Center RA in degrees")
    parser.add_argument("dec0", type=float, help="Center Dec in degrees")
    parser.add_argument("radius", type=float, help="Radius in degrees")
    parser.add_argument("pmcatalog_dir", help="Dir of PM mover FITS files (one per healpixel)")
    parser.add_argument("coadd_dir", help="Dir of coadd skim FITS files (one per healpixel)")
    parser.add_argument("--forecast", default=DEFAULT_FORECAST, help="Static forecast .npz (default: attic/pm_plot.npz)")
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--match-arcsec", type=float, default=0.1)
    parser.add_argument("--mag-col", default="PSF_MAG_APER_8_I")
    parser.add_argument("--pm-pattern", default="*.fits",
                         help="Glob restricting which files count as PM movers, e.g. 'real_PM_hp*.fits' "
                              "to exclude fake-star injection test files in the same dir")
    parser.add_argument("--coadd-pattern", default="*.fits", help="Glob restricting which coadd files to read")
    parser.add_argument("-o", "--out", default="precision_plot.png")
    args = parser.parse_args()

    stars_matched = measured_precision(
        args.ra0, args.dec0, args.radius, args.pmcatalog_dir, args.coadd_dir,
        nside=args.nside, match_arcsec=args.match_arcsec,
        pm_pattern=args.pm_pattern, coadd_pattern=args.coadd_pattern,
    )
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
        extent=(18, 24, np.log10(0.02), np.log10(50.0)),
        yscale="log", mincnt=5, cmap="managua", gridsize=100, edgecolors="none",
    )

    ax.semilogy(imag, med_dec[:, 0], "c-", lw=3, label="DECam")
    ax.semilogy(imag, med_lsst[:, 1], "m--", lw=2, label="LSST 1yr")
    ax.semilogy(imag, med_dec[:, 1], "m-", lw=3, label="LSST 1yr + DECam")
    ax.semilogy(imag, med_dec[:, 10], "k-", lw=3, label="LSST 10yr + DECam")
    ax.semilogy(dr5_G - Gi, dr5_sigpm, "g:", lw=5, label="Gaia DR5")
    ax.semilogy((dr5_G - Gi)[-1], dr5_sigpm[-1], "g*", ms=16)
    ax.semilogy(dr5_G - Gi, dr3_sigpm, "r:", lw=5, label="Gaia DR3")
    ax.semilogy((dr5_G - Gi)[-1], dr3_sigpm[-1], "r*", ms=16)

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(18, 24)
    ax.set_ylim(0.02, 50)
    ax.set_xlabel(f"{args.mag_col} magnitude", fontsize=20)
    ax.set_ylabel(r"$\sigma_{\mu_{\alpha*}}$ (mas/yr)", fontsize=20)
    ax.legend(fontsize=10)
    ax.grid()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
