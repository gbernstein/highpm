import argparse
import os
import re
import sys
from glob import glob

import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn

# Ensure project root is importable when running this script directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from highpm.detection_packaging import (
    clean_err_detections,
    clean_healpix_detections,
    clean_snr_detections,
    concatenate_detections,
    get_exposures_near_healpix,
    get_healpix_center,
)
from highpm.gnomonic_converter import projectGnomonic

"""
Command-line wrapper that orchestrates argument parsing and calls the detection
packaging functions in highpm.detection_packaging.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate detection files, clean them, and write out a single "
            "cleaned detection file per healpixel."
        )
    )

    parser.add_argument(
        "--healpix",
        type=int,
        required=True,
        help="HEALPix pixel number to process.",
    )

    parser.add_argument(
        "--exposures-file",
        required=True,
        help=(
            "FITS file containing exposure metadata with columns 'EXPNUM', 'RA', "
            "'DEC'."
        ),
    )

    parser.add_argument(
        "--detection-path",
        required=True,
        help=(
            "Directory or glob prefix for detection FITS files. The script will "
            "search for 'detections_*.fits' under this path/prefix."
        ),
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Directory to write cleaned detection FITS files (one per healpixel).",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=32,
        help="HEALPix nside parameter for sky tiling (default: 32).",
    )
    parser.add_argument(
        "--subside",
        type=int,
        default=16,
        help=(
            "Subdivide each HEALPix pixel into subsquares of size (nside*subside) "
            "(default: 16)."
        ),
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=5.0,
        help="Signal-to-noise ratio threshold for cleaning detections (default: 5.0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if they exist.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    exposure_data = fitsio.read(args.exposures_file)
    expnum = np.array(exposure_data["expnum"], dtype="i8")
    epxra = np.array(exposure_data["ra"], dtype="f8")
    expdec = np.array(exposure_data["dec"], dtype="f8")

    exposures = get_exposures_near_healpix(
        args.healpix, epxra, expdec, expnum, nside=args.nside
    )

    detection_files = sorted(
        glob(os.path.join(args.detection_path, "updated_skim_gpr_coadd_*.fits"))
    )
    if len(detection_files) == 0:
        print(f"No detection files found in {args.detection_path}")
        sys.exit(1)

    detection_files = [
        f
        for f in detection_files
        if (
            (match := re.search(r"_(\d+)\.fits$", f)) is not None
            and int(match.group(1)) in exposures
        )
    ]
    if len(detection_files) == 0:
        print(f"No detection files match exposures for HEALPix {args.healpix}")
        sys.exit(1)

    print(f"Found {len(detection_files)} detection files. Concatenating...")
    detections = concatenate_detections(detection_files)
    print(f"Total detections before cleaning: {len(detections)}")

    detections = clean_err_detections(detections)
    print(f"Detections after error cleaning: {len(detections)}")

    detections = clean_snr_detections(detections, snr_threshold=args.snr_threshold)
    print(f"Detections after SNR cleaning: {len(detections)}")

    detections = clean_healpix_detections(
        detections, args.healpix, nside=args.nside, subside=args.subside
    )
    print(f"Detections after HEALPix cleaning: {len(detections)}")

    if len(detections) == 0:
        print("No detections remain after cleaning. Exiting.")
        sys.exit(0)

    ra0, dec0 = get_healpix_center(args.healpix, nside=args.nside)
    xi, eta, dxi, deta = projectGnomonic(
        detections["BEST_RA"],
        detections["BEST_DEC"],
        detections["DRA_DCOLOR"],
        detections["DDEC_DCOLOR"],
        ra0,
        dec0,
    )

    detections = rfn.append_fields(
        base=detections,
        names=["ID", "XI", "ETA", "DXI_DCOLOR", "DETA_DCOLOR"],
        data=[np.arange(len(detections), dtype="i8"), xi, eta, dxi, deta],
        dtypes=[
            np.dtype("i8"),
            np.dtype("f8"),
            np.dtype("f8"),
            np.dtype("f8"),
            np.dtype("f8"),
        ],
        usemask=False,
        asrecarray=True,
    )

    hdr = {"ra0": ra0, "dec0": dec0, "nside": args.nside, "subside": args.subside}

    outfilename = os.path.join(
        args.output_path, f"cleaned_detections_hp{args.healpix:05d}.fits"
    )
    if os.path.exists(outfilename) and not args.overwrite:
        print(f"Output file {outfilename} exists and --overwrite not set. Exiting.")
        sys.exit(1)

    fitsio.write(outfilename, detections, header=hdr, clobber=True)
    print(f"Wrote cleaned detections to {outfilename}")
