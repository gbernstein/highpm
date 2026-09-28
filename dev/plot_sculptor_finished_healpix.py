"""Scatter plot of sky positions for finished (real_PM_hp*.fits) healpixels
within a radius of Sculptor, deduped and downsampled for a quick coverage look.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from healpix_region_data import disc_pixels, load_movers

SCULPTOR_RA, SCULPTOR_DEC = 15.038750, -33.709000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmcatalog-dir", default="/data8/shared/decampm/PMSculptor/PMCatalog/")
    parser.add_argument("--radius", type=float, default=5.0, help="Radius in degrees around Sculptor")
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--downsample", type=int, default=10, help="Keep 1 out of every N points")
    parser.add_argument("-o", "--out", default="sculptor_finished_healpix.png")
    args = parser.parse_args()

    pixels = disc_pixels(args.nside, SCULPTOR_RA, SCULPTOR_DEC, args.radius)
    movers = load_movers(args.pmcatalog_dir, args.nside, pixels, pattern="real_PM_hp*.fits")

    ra = movers["ra"][:: args.downsample]
    dec = movers["dec"][:: args.downsample]

    fig, ax = plt.subplots()
    ax.invert_xaxis()
    ax.scatter(ra, dec, s=1, alpha=0.1)
    ax.scatter(SCULPTOR_RA, SCULPTOR_DEC, c="r", s=10)
    circle = plt.Circle((SCULPTOR_RA, SCULPTOR_DEC), radius=args.radius, fill=False, color="r", linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.set_aspect("equal")
    ax.grid()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
