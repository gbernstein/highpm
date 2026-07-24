import argparse
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

DEFAULT_REGEX = r"(\d{7})"


def scan_expnums(directory, pattern, regex):
    expnums = set()
    for f in glob(os.path.join(directory, pattern)):
        m = re.search(regex, os.path.basename(f))
        if m:
            expnums.add(int(m.group(1)))
    return np.array(sorted(expnums))


def match_pointings(expnums, pointing_file):
    cat = Table.read(pointing_file, path="__astropy_table__")
    mask = np.isin(np.asarray(cat["expnum"]), expnums)
    return cat["pole"][mask, 0], cat["pole"][mask, 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scatter-plot the sky positions of exposures found in a directory."
    )
    parser.add_argument("--dir", required=True, help="Directory to scan for exposures.")
    parser.add_argument("--pattern", default="*", help="Glob pattern for filenames (default: '*').")
    parser.add_argument(
        "--regex",
        default=DEFAULT_REGEX,
        help=f"Regex whose first group is the expnum (default: '{DEFAULT_REGEX}').",
    )
    parser.add_argument(
        "--pointing-file", required=True, help="Path to delveExposures.hdf5 (astropy Table)."
    )
    parser.add_argument("--output", default="exposure_pointings.png", help="Output image path.")
    args = parser.parse_args()

    expnums = scan_expnums(args.dir, args.pattern, args.regex)
    print(f"Found {len(expnums)} exposures in {args.dir}.")

    ra, dec = match_pointings(expnums, args.pointing_file)
    print(f"Matched {len(ra)} pointings in {args.pointing_file}.")

    plt.scatter(ra, dec, s=2, alpha=0.5)
    plt.gca().invert_xaxis()
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.grid()
    plt.savefig(args.output)
    print(f"Saved plot to {args.output}")
