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
    concatenate_detections,
    get_exposures_near_healpix,
    get_healpix_center,
    healpix_membership_mask,
    parallax_factors as _parallax_factors,
    rotate_covariances_to_healpix_frame,
)
from highpm.gnomonic_converter import projectGnomonic

"""
Command-line wrapper that orchestrates argument parsing and calls the detection
packaging functions in highpm.detection_packaging.
"""


def _radec_from_table(tab):
    """(expnum, ra_deg, dec_deg) from an exposure table, case-insensitively.

    Old y6a1 FITS stores ra/dec columns; pixmappy v2 delveExposures.hdf5 stores
    the pointing as 'pole' = [ra, dec] (deg). astropy reads both file formats.
    """
    col = {c.lower(): c for c in tab.colnames}
    expnum = np.array(tab[col["expnum"]], dtype="i8")
    if "ra" in col and "dec" in col:
        ra = np.array(tab[col["ra"]], dtype="f8")
        dec = np.array(tab[col["dec"]], dtype="f8")
    else:
        pole = np.array(tab[col["pole"]], dtype="f8")  # (N, 2) = [ra, dec] deg
        ra, dec = pole[:, 0], pole[:, 1]
    return expnum, ra, dec


def _self_test():
    from astropy.table import Table

    fits_like = Table({"expnum": [1, 2], "ra": [10.0, 20.0], "dec": [-30.0, -40.0]})
    hdf5_like = Table({"expnum": [1, 2], "pole": [[10.0, -30.0], [20.0, -40.0]]})
    for t in (fits_like, hdf5_like):
        e, r, d = _radec_from_table(t)
        assert list(e) == [1, 2] and list(r) == [10.0, 20.0] and list(d) == [-30.0, -40.0]

    # Observatory along +x (ra=0, dec=0): no parallax shift at the (0, 0)
    # tangent point; at (90, 0) it's -obs . e_xi = -(-1) = +1 in xi.
    par_tab = Table({"expnum": [7, 3], "obsicrs": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]})
    px, pe = _parallax_factors(par_tab, np.array([7, 7, 3]), 0.0, 0.0)
    assert np.allclose(px, [0.0, 0.0, 0.0]) and np.allclose(pe, [0.0, 0.0, -1.0])
    px, pe = _parallax_factors(par_tab, np.array([7]), 90.0, 0.0)
    assert np.allclose(px, [1.0]) and np.allclose(pe, [0.0])
    print("self-test ok: ra/dec schemas and parallax factors")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description=(
            "Concatenate detection files, clean them, and write out a single "
            "cleaned detection file per healpixel."
        )
    )

    parser.add_argument(
        "--exposures-file",
        required=True,
        help=(
            "Exposure metadata table (FITS or HDF5). Needs 'expnum' plus either "
            "'ra'/'dec' columns or a 'pole' = [ra, dec] column (pixmappy v2)."
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

    parser.add_argument(
        "--healpix",
        type=int,
        help="HEALPix pixel number to process.",
    )
    parser.add_argument(
        "--index",
        type=int,
    )
    parser.add_argument(
        "--healpix-npy",
    )

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    if args.healpix is None:
        healpixels = np.load(args.healpix_npy,allow_pickle=True)
        healpix = healpixels[args.index]
    else:
        healpix = args.healpix

    from astropy.table import Table

    exposure_table = Table.read(args.exposures_file)
    expnum, expra, expdec = _radec_from_table(exposure_table)

    exposures = get_exposures_near_healpix(
        healpix, expra, expdec, expnum, nside=args.nside
    )

    detection_files = sorted(
        glob(os.path.join(args.detection_path, "position_corrected_*.fits"))
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
        print(f"No detection files match exposures for HEALPix {healpix}")
        sys.exit(1)

    def row_filter(data):
        # Same three cuts as clean_err/clean_snr/clean_healpix_detections, applied
        # per-file so rows outside this healpixel never get held in memory
        # alongside the rest of a dense pixel's overlapping exposures. MJD <= 0
        # drops exposures positionCorrection wrote before it started skipping
        # those missing from the exposure table (old files carry MJD = -1).
        err_ok = (data["BEST_RA_ERR"] > 0.0) & (data["BEST_DEC_ERR"] > 0.0)
        err_ok &= data["MJD"] > 0.0
        snr_ok = (data["FLUX_PSF"] / data["FLUXERR_PSF"]) >= args.snr_threshold
        healpix_ok = healpix_membership_mask(
            data["BEST_RA"], data["BEST_DEC"], healpix, nside=args.nside, subside=args.subside
        )
        return err_ok & snr_ok & healpix_ok

    print(f"Found {len(detection_files)} detection files. Concatenating...")
    detections = concatenate_detections(detection_files, row_filter=row_filter)
    print(f"Detections after cleaning: {len(detections)}")

    if len(detections) == 0:
        print("No detections remain after cleaning. Exiting.")
        sys.exit(0)

    ra0, dec0 = get_healpix_center(healpix, nside=args.nside)

    detections = rotate_covariances_to_healpix_frame(detections, ra0, dec0)

    # PAR_XI/PAR_ETA came out of positionCorrection in the GPR tangent frame;
    # recompute them about the healpixel center, the frame XI/ETA and the
    # rotated covariance are in (and fake_detections' parallax factors use).
    detections["PAR_XI"], detections["PAR_ETA"] = _parallax_factors(
        exposure_table, detections["EXPNUM"], ra0, dec0
    )

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
        args.output_path, f"cleaned_detections_hp{healpix:05d}.fits"
    )
    if os.path.exists(outfilename) and not args.overwrite:
        print(f"Output file {outfilename} exists and --overwrite not set. Exiting.")
        sys.exit(1)

    fitsio.write(outfilename, detections, header=hdr, clobber=True) 
    print(f"Wrote cleaned detections to {outfilename}")
