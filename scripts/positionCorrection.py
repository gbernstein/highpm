import argparse
import os
import re
import sys
from glob import glob
from multiprocessing import Pool

import numpy as np

# Ensure project root is importable when running this script directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

"""
Command-line wrapper that orchestrates argument parsing and parallel mapping.
The implementation details live in highpm.position_correction to keep the worker
function picklable for multiprocessing.
"""


if __name__ == "__main__":
    # Import implementation from package to ensure picklable targets in multiprocessing
    from highpm.position_correction import process_exposure

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
            "Directory or glob prefix for GPR FITS files named like 'gpr_*{expnum:07d}_<band>.fits'"
            " (e.g., gpr_something_1234567_r.fits)."
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
    # If neither is provided, we will auto-discover all available exposures.
    group = parser.add_mutually_exclusive_group(required=False)
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
    parser.add_argument(
        "--index",
        type=int,
        help=(
            "Index of exposure number in --exposures-npy."
        ),
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

    # Helper: discover exposures from files when none provided
    def _discover_exposures(skims_path: str, gpr_path: str) -> np.ndarray:
        # Skims look like: D*{expnum:08d}_*.fits -> extract 8 digits before underscore
        skim_files = glob(os.path.join(skims_path, "D*.fits"))
        skim_expnums = set()
        for f in skim_files:
            bn = os.path.basename(f)
            m = re.search(r"(\d{8})(?=_)", bn)
            if m:
                try:
                    skim_expnums.add(int(m.group(1)))
                except ValueError:
                    pass

        # GPR files look like: gpr_*{expnum:07d}_<band>.fits
        # Extract 7-digit expnum immediately before the _<band>.fits suffix
        # and accept common DES bands [g,r,i,z].
        gpr_files = glob(os.path.join(gpr_path, "gpr_*_*.fits"))
        gpr_expnums = set()
        for f in gpr_files:
            bn = os.path.basename(f)
            m = re.search(r"(\d{7})(?=_[griz]\.fits$)".replace(" ", ""), bn)
            if m:
                try:
                    gpr_expnums.add(int(m.group(1)))
                except ValueError:
                    pass

        if not skim_expnums:
            raise SystemExit(
                "No skim exposures found. Checked pattern 'D*.fits' under skims-path."
            )
        if not gpr_expnums:
            raise SystemExit(
                "No GPR exposures found. Checked pattern 'gpr_*_<band>.fits' under gpr-path."
            )

        # Intersection to ensure both inputs exist per exposure
        inter = skim_expnums & gpr_expnums
        if not inter:
            raise SystemExit(
                f"No overlapping exposures between skim ({len(skim_expnums)}) and GPR ({len(gpr_expnums)})."
            )
        exposures = sorted(inter)

        return np.array(exposures, dtype=int)

    # Gather exposures from args or discover automatically
    if args.expnum is not None:
        exposures = np.array(args.expnum, dtype=int)
        source = "--expnum"
    elif args.exposures_npy is not None:
        exposures = np.load(args.exposures_npy, allow_pickle=True)
        source = "--exposures-npy"
    else:
        exposures = _discover_exposures(args.skims_path, args.gpr_path)
        source = "auto-discovered"

    if args.index is not None and args.exposures_npy is not None:
        print(1*args.index, 1*(1+args.index))
        exposures = np.array([exposures[1*args.index:(1*(1+args.index))]], dtype=int)

    print(f"Skim Path: {args.skims_path}")
    print(f"GPR Path: {args.gpr_path}")
    print(f"Coadd Path: {args.coadd_path}")
    print(f"Output Path: {args.output_path}")
    print(f"Number of processes: {args.processes}")
    print(
        f"Processing exposures ({source}): {exposures[0][:10]}... (total {len(exposures[0])})"
    )

    # Build argument tuples and use starmap to avoid pickling partials defined in __main__
    task_args = [
        (expnum, args.skims_path, args.gpr_path, args.coadd_path, args.output_path)
        for expnum in exposures[0]
    ]

    with Pool(processes=args.processes) as pool:
        pool.starmap(process_exposure, task_args)
