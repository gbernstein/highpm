import argparse
import os
import sys

import numpy as np

# Ensure project root is importable when running this script directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from highpm.detection_packaging import get_exposures_near_healpix


def _radec_from_table(tab):
    """(expnum, ra_deg, dec_deg) from an exposure table, same convention as scripts/detectionPacking.py."""
    col = {c.lower(): c for c in tab.colnames}
    expnum = np.array(tab[col["expnum"]], dtype="i8")
    if "ra" in col and "dec" in col:
        ra = np.array(tab[col["ra"]], dtype="f8")
        dec = np.array(tab[col["dec"]], dtype="f8")
    else:
        pole = np.array(tab[col["pole"]], dtype="f8")  # (N, 2) = [ra, dec] deg
        ra, dec = pole[:, 0], pole[:, 1]
    return expnum, ra, dec


def exposures_for_healpixels(healpixels, expnum, ra, dec, nside):
    """Union of get_exposures_near_healpix() over every pixel in healpixels."""
    found = [get_exposures_near_healpix(hp_i, ra, dec, expnum, nside=nside) for hp_i in healpixels]
    return np.unique(np.concatenate(found)) if found else np.array([], dtype=expnum.dtype)


def _self_test():
    expnum = np.array([1, 2, 3, 4])
    ra = np.array([0.0, 0.0, 180.0, 180.0])
    dec = np.array([0.0, 0.0, 0.0, 0.0])
    import healpy as hp

    nside = 32
    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    result = exposures_for_healpixels(np.unique(pix), expnum, ra, dec, nside)
    assert set(result) == {1, 2, 3, 4}, result
    print("self-test ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Collect the union of exposures (expnums) near every healpixel listed in a "
            "--healpix-npy file, using the same neighbor-based matching as "
            "scripts/detectionPacking.py's get_exposures_near_healpix()."
        )
    )
    parser.add_argument(
        "--exposures-file",
        help=(
            "Exposure metadata table (FITS or HDF5). Needs 'expnum' plus either "
            "'ra'/'dec' columns or a 'pole' = [ra, dec] column (pixmappy v2)."
        ),
    )
    parser.add_argument("--healpix-npy", help="Path to .npy file with an array of healpixel numbers.")
    parser.add_argument("--nside", type=int, default=32, help="HEALPix nside (default 32).")
    parser.add_argument("--output-file", help="Path to save the union of expnums as a .npy file.")
    parser.add_argument("--self-test", action="store_true", help="Run a self-check and exit.")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        sys.exit(0)

    if not args.exposures_file or not args.healpix_npy or not args.output_file:
        parser.error("--exposures-file, --healpix-npy, and --output-file are required")

    from astropy.table import Table

    healpixels = np.load(args.healpix_npy, allow_pickle=True)
    print(f"{len(healpixels)} healpixels in {args.healpix_npy}")

    expnum, ra, dec = _radec_from_table(Table.read(args.exposures_file))
    exposures = exposures_for_healpixels(healpixels, expnum, ra, dec, args.nside)

    np.save(args.output_file, exposures)
    print(f"{len(exposures)} exposure numbers saved to {args.output_file}")
