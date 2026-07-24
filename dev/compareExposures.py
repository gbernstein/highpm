import argparse
import os
import re
from glob import glob

import numpy as np

DEFAULT_REGEX = r"(\d{8})"


def scan_expnums(directories, pattern, regex):
    expnums = set()
    for directory in directories:
        for f in glob(os.path.join(directory, pattern)):
            m = re.search(regex, os.path.basename(f))
            if m:
                expnums.add(int(m.group(1)))
    return np.array(sorted(expnums))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate exposure numbers from filenames in one or more "
        "directories and compare against a .npy array of exposure ids."
    )
    parser.add_argument("--dirs", nargs="+", required=True, help="Directories to scan.")
    parser.add_argument(
        "--pattern", default="*", help="Glob pattern for filenames (default: '*')."
    )
    parser.add_argument(
        "--regex",
        default=DEFAULT_REGEX,
        help=f"Regex whose first group is the expnum (default: '{DEFAULT_REGEX}').",
    )
    parser.add_argument("--npy-file", required=True, help="Path to .npy file of expnums.")
    args = parser.parse_args()

    found = scan_expnums(args.dirs, args.pattern, args.regex)
    reference = np.unique(np.load(args.npy_file))

    missing = np.setdiff1d(reference, found)
    extra = np.setdiff1d(found, reference)
    common = np.intersect1d(found, reference)

    print(f"Found {len(found)} exposures across {len(args.dirs)} dir(s).")
    print(f"Reference file has {len(reference)} exposures.")
    print(f"Common: {len(common)}")
    print(f"In reference but not found: {len(missing)}")
    if len(missing):
        print(missing)
    print(f"Found but not in reference: {len(extra)}")
    if len(extra):
        print(extra)
