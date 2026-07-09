"""Completeness plot: fraction of coadd stars recovered as PM movers (and
optionally Gaia), within radius degrees of a center on the sky.

Pulls only the PMCatalog/coadd_skims healpixel files needed for the region.
"""

import argparse
import os
import sys

import fitsio
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from healpix_region_data import disc_pixels, load_coadd, load_movers, match_to_coadd


def completeness(
    ra0, dec0, radius, pmcatalog_dir, coadd_dir, gaia_file=None, nside=32,
    match_arcsec=0.1, mag_col="PSF_MAG_APER_8_G", mag_range=(18, 28), bins=100,
    pm_pattern="*.fits", coadd_pattern="*.fits",
):
    pixels = disc_pixels(nside, ra0, dec0, radius)
    movers = load_movers(pmcatalog_dir, nside, pixels, pattern=pm_pattern)
    coadd = load_coadd(coadd_dir, pixels, pattern=coadd_pattern)

    mover_matched = match_to_coadd(movers, coadd, match_arcsec)
    stars_matched = mover_matched[mover_matched["EXT_MASH"] <= 1]
    coadd_stars = coadd[coadd["EXT_MASH"] <= 1]

    coadd_hist, edges = np.histogram(coadd_stars[mag_col], bins=bins, range=mag_range)
    mover_hist, _ = np.histogram(stars_matched[mag_col], bins=bins, range=mag_range)
    centers = 0.5 * (edges[:-1] + edges[1:])

    gaia_hist = None
    if gaia_file:
        gaia = fitsio.read(gaia_file)
        gaia = gaia[~np.isnan(gaia["pmra"])]
        gaia_matched = match_to_coadd(gaia, coadd, match_arcsec)
        gaia_stars = gaia_matched[gaia_matched["EXT_MASH"] <= 1]
        gaia_hist, _ = np.histogram(gaia_stars[mag_col], bins=bins, range=mag_range)

    return centers, coadd_hist, mover_hist, gaia_hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ra0", type=float, help="Center RA in degrees")
    parser.add_argument("dec0", type=float, help="Center Dec in degrees")
    parser.add_argument("radius", type=float, help="Radius in degrees")
    parser.add_argument("pmcatalog_dir", help="Dir of PM mover FITS files (one per healpixel)")
    parser.add_argument("coadd_dir", help="Dir of coadd skim FITS files (one per healpixel)")
    parser.add_argument("--gaia", help="Gaia result FITS (ra/dec/pmra) for a second completeness curve")
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--match-arcsec", type=float, default=0.1)
    parser.add_argument("--mag-col", default="PSF_MAG_APER_8_G")
    parser.add_argument("--mag-min", type=float, default=18.0)
    parser.add_argument("--mag-max", type=float, default=28.0)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--pm-pattern", default="*.fits",
                         help="Glob restricting which files count as PM movers, e.g. 'real_PM_hp*.fits' "
                              "to exclude fake-star injection test files in the same dir")
    parser.add_argument("--coadd-pattern", default="*.fits", help="Glob restricting which coadd files to read")
    parser.add_argument("-o", "--out", default="completeness_plot.png")
    args = parser.parse_args()

    centers, coadd_hist, mover_hist, gaia_hist = completeness(
        args.ra0, args.dec0, args.radius, args.pmcatalog_dir, args.coadd_dir,
        gaia_file=args.gaia, nside=args.nside, match_arcsec=args.match_arcsec,
        mag_col=args.mag_col, mag_range=(args.mag_min, args.mag_max), bins=args.bins,
        pm_pattern=args.pm_pattern, coadd_pattern=args.coadd_pattern,
    )

    plt.figure()
    plt.plot(centers, mover_hist / coadd_hist, label="DECam")
    if gaia_hist is not None:
        plt.plot(centers, gaia_hist / coadd_hist, label="Gaia")
    plt.xlabel(f"{args.mag_col} magnitude", fontsize=20)
    plt.ylabel("Completeness", fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.grid()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
