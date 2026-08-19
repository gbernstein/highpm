"""Plot detection density (deg^-2) in a healpix detections file, binned into
fine healpix pixels and rendered as a zoomed (gnomonic) sky map centered on
the file's RA0/DEC0 header values.
"""

import argparse
import os
import sys

import fitsio
import healpy as hp
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from highpm.cat_reader import clean_cat, read_cat_data, read_cat_header

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", help="Path to a cleaned_detections_hp*.fits file")
    parser.add_argument("--nside", type=int, default=4096, help="Fine healpix nside for density binning")
    parser.add_argument("--reso", type=float, default=None,
                         help="gnomview pixel resolution in arcmin (default: auto-fit to the data extent)")
    parser.add_argument("--xsize", type=int, default=800, help="gnomview image size in pixels")
    parser.add_argument("-o", "--out", default="healpix_density.png")
    args = parser.parse_args()

    cat = read_cat_data(args.catalog)
    # Same quality cuts PM.py applies before feeding detections to the fitter
    # (FLAGS/IMAFLAGS_ISO/SPREAD_MODEL) -- otherwise the density map includes
    # detections the PM code never sees.
    cat = cat[clean_cat(cat)]
    ra, dec = cat["BEST_RA"], cat["BEST_DEC"]

    header = read_cat_header(args.catalog)
    ra0 = header.get("RA0", np.median(ra))
    dec0 = header.get("DEC0", np.median(dec))

    pix = hp.ang2pix(args.nside, ra, dec, lonlat=True)
    npix = hp.nside2npix(args.nside)
    counts = np.bincount(pix, minlength=npix)
    pix_area_deg2 = hp.nside2pixarea(args.nside, degrees=True)
    density = np.where(counts > 0, counts / pix_area_deg2, hp.UNSEEN)

    print(f"{len(cat)} detections, {np.sum(counts > 0)} occupied pixels at nside={args.nside}")
    print(f"density: median={np.median(density[counts > 0]):.1f} deg^-2, "
          f"max={density[counts > 0].max():.1f} deg^-2")

    reso = args.reso
    if reso is None:
        radius_deg = np.hypot(ra - ra0, dec - dec0).max()
        reso = 60.0 * 2.2 * radius_deg / args.xsize  # arcmin/pix, ~10% margin

    hp.gnomview(
        density, rot=(ra0, dec0), reso=reso, xsize=args.xsize,
        title=f"Detection density near ({ra0:.3f}, {dec0:.3f})",
        unit="deg$^{-2}$", cmap="viridis",
    )
    hp.graticule()
    import matplotlib.pyplot as plt
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")
