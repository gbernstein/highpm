"""Deduped modest1_movers density plot across all healpixels in a region,
gnomonically projected around the region center and rendered as a
log-scaled hexbin.

Loads real_PM_hp*.fits movers for each healpixel in the region (via
healpix_region_data.load_movers, which dedupes movers within 1 arcsec that
appear in more than one file), keeps only the modest1_movers, projects them
into xi/eta about the center, and hexbins with a log color scale.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from healpix_region_data import disc_pixels, load_movers
from highpm.gnomonic_converter import projectGnomonic

SCULPTOR_RA, SCULPTOR_DEC = 15.038750, -33.709000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmcatalog-dir", default="/data8/shared/decampm/PMSculptor_8deg_garyb/PMCatalog/")
    parser.add_argument("--ra0", type=float, default=SCULPTOR_RA)
    parser.add_argument("--dec0", type=float, default=SCULPTOR_DEC)
    parser.add_argument("--radius", type=float, default=8.0, help="Region radius in degrees")
    parser.add_argument("--nside", type=int, default=32, help="Nside used to pixelize the PMCatalog files")
    parser.add_argument("--gridsize", type=int, default=200, help="hexbin gridsize")
    parser.add_argument("--pm-pattern", default="real_PM_hp*.fits",
                         help="Glob restricting which files to read (excludes injection files by default)")
    parser.add_argument("-o", "--out", default="modest1_movers_density.png")
    args = parser.parse_args()

    pixels = disc_pixels(args.nside, args.ra0, args.dec0, args.radius)
    movers = load_movers(
        args.pmcatalog_dir, args.nside, pixels,
        pattern=args.pm_pattern, exts=("modest1_movers",),
    )
    ra, dec = movers["ra"], movers["dec"]

    zeros = np.zeros_like(ra)
    xi, eta, *_ = projectGnomonic(ra, dec, zeros, zeros, args.ra0, args.dec0)

    print(f"{len(movers)} deduped modest1 movers")

    fig, ax = plt.subplots(figsize=(8, 7))
    hb = ax.hexbin(xi, eta, gridsize=args.gridsize, cmap="viridis", mincnt=1,
                    edgecolor="none", bins="log")
    fig.colorbar(hb, ax=ax, label="log$_{10}$(modest1 movers / bin)")

    ax.set_xlabel(r"$\xi$ (deg)", fontsize=14)
    ax.set_ylabel(r"$\eta$ (deg)", fontsize=14)
    ax.set_title(f"Deduped modest1 mover density near ({args.ra0:.3f}, {args.dec0:.3f})", fontsize=12)
    ax.set_aspect("equal")
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
