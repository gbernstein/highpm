import argparse
import sys

import fitsio
import numpy as np
from scipy.spatial import KDTree

sys.path.append("/home/vwetzell/gitrepos/highpm")

from highpm.gnomonic_converter import projectGnomonic


def complete_exposure_clip(cat, ra0, dec0, radius):

    xi, eta, *_ = projectGnomonic(
        cat["BEST_RA"],
        cat["BEST_DEC"],
        np.zeros_like(cat["BEST_RA"]),
        np.zeros_like(cat["BEST_RA"]),
        ra0,
        dec0,
    )

    tree = KDTree(np.vstack([xi, eta]).T)
    idx = tree.query_ball_point([0, 0], radius)

    return cat[idx]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input FITS catalog file")
    parser.add_argument("outfile", type=str, help="Output FITS catalog file")
    parser.add_argument("ra0", type=float, help="Center RA in degrees")
    parser.add_argument("dec0", type=float, help="Center DEC in degrees")
    parser.add_argument("radius", type=float, help="Radius in degrees")

    args = parser.parse_args()

    cat = fitsio.read(args.infile)
    headers = fitsio.read_header(args.infile, ext=1)

    print(f"Read {len(cat)} objects from {args.infile}")

    clipped_cat = complete_exposure_clip(cat, args.ra0, args.dec0, args.radius)

    fitsio.write(args.outfile, clipped_cat, header=headers, clobber=True)
    print(f"Wrote {len(clipped_cat)} objects to {args.outfile}")
