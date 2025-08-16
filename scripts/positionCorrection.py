import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

# Ensure project root is importable when running this script directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import implementation from package to ensure picklable targets in multiprocessing
from highpm.position_correction import process_exposure

"""
Command-line wrapper that orchestrates argument parsing and parallel mapping.
The implementation details live in highpm.position_correction to keep the worker
function picklable for multiprocessing.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Match skim and GPR catalogs, cross-match to coadds, apply pixmappy "
            "coordinate corrections, and write updated FITS tables per exposure."
        )
    )

    # Required I/O paths
    parser.add_argument(
        "--skims-path",
        required=True,
        help=(
            "Directory or glob prefix for skim FITS files. getSkimFile will search "
            "for 'D*{expnum:08d}_*.fits' under this path/prefix."
        ),
    )
    parser.add_argument(
        "--gpr-path",
        required=True,
        help=(
            "Directory or glob prefix for GPR FITS files named 'gpr_*{expnum:07d}.fits'."
        ),
    )
    parser.add_argument(
        "--coadd-path",
        required=True,
        help=("Directory containing coadd files named 'y6_gold_2_2_{pix:05d}.fits'."),
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Directory to write updated FITS files (one per exposure).",
    )

    # Exposure selection (mutually exclusive: explicit list vs .npy file)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--expnum",
        type=int,
        nargs="+",
        help="One or more exposure numbers to process.",
    )
    group.add_argument(
        "--exposures-npy",
        help=("Path to a .npy file containing a list/array of exposure numbers."),
    )

    # Optional environment/file settings
    parser.add_argument(
        "--des-exposures",
        default=None,
        help=(
            "Path to DES exposures FITS table; if provided, sets the DES_EXPOSURES "
            "environment variable used by pixmappy DESMaps."
        ),
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel processes to use (default: CPU count).",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Optionally set environment for DES_EXPOSURES
    if args.des_exposures:
        os.environ["DES_EXPOSURES"] = args.des_exposures
    elif "DES_EXPOSURES" not in os.environ:
        parser.error(
            "DES_EXPOSURES is not set. Provide --des-exposures or set the environment variable."
        )

    # Gather exposures
    if args.expnum is not None:
        exposures = np.array(args.expnum, dtype=int)
    else:
        exposures = np.load(args.exposures_npy, allow_pickle=True)

    print(f"Skim Path: {args.skims_path}")
    print(f"GPR Path: {args.gpr_path}")
    print(f"Coadd Path: {args.coadd_path}")
    print(f"Output Path: {args.output_path}")
    print(f"Number of processes: {args.processes}")
    print(f"Processing exposures: {exposures[:10]}... (total {len(exposures)})")

    # Build argument tuples and use starmap to avoid pickling partials defined in __main__
    task_args = [
        (int(expnum), args.skims_path, args.gpr_path, args.coadd_path, args.output_path)
        for expnum in exposures
    ]

    with Pool(processes=args.processes) as pool:
        pool.starmap(process_exposure, task_args)
