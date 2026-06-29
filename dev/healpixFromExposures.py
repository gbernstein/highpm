import argparse
import os
import re
import sys
from glob import glob

import numpy as np


def _disc_pixels(nside: int, ra: float, dec: float, radius_deg: float) -> np.ndarray:
    """Healpixels overlapping a disc of radius_deg centered on (ra, dec) [deg]."""
    import healpy as hp

    vec = hp.ang2vec(np.radians(90.0 - dec), np.radians(ra))
    # ponytail: inclusive=True so a pixel counts as covered if any part falls in the
    # focal plane; slightly over-counts at edges, which is the safe side for "covered".
    return hp.query_disc(nside, vec, np.radians(radius_deg), inclusive=True)


def _expnums_in_dir(exposure_dir: str, pattern: str) -> set:
    expnums = set()
    for f in glob(os.path.join(exposure_dir, pattern)):
        m = re.search(r"(\d+)", os.path.basename(f))
        if m:
            expnums.add(int(m.group(1)))
    return expnums


def _self_test() -> None:
    import healpy as hp

    nside, ra, dec = 32, 45.0, -20.0
    pix = _disc_pixels(nside, ra, dec, 1.1)
    center = hp.ang2pix(nside, np.radians(90.0 - dec), np.radians(ra))
    assert center in pix, "center pixel missing from disc"
    assert len(pix) == len(np.unique(pix)) > 0
    print(f"self-test ok: {len(pix)} pixels for a 1.1 deg disc at nside {nside}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Accumulate the healpixels covered by a set of exposures: read expnums "
            "from FITS filenames in a directory, look up each pointing (pole) in the "
            "pixmappy exposure table, and union all healpixels within --radius of it."
        )
    )
    parser.add_argument(
        "--exposure-dir",
        help="Directory of position-corrected exposure FITS files.",
    )
    parser.add_argument(
        "--exposure-table",
        default=os.environ.get("DES_EXPOSURES"),
        help="pixmappy exposure table (delveExposures.hdf5). Defaults to $DES_EXPOSURES.",
    )
    parser.add_argument(
        "--output-file",
        help="Path to output .npy file containing the covered healpixels.",
    )
    parser.add_argument("--nside", type=int, default=32, help="Healpix nside (default 32).")
    parser.add_argument(
        "--radius",
        type=float,
        default=1.1,
        help="Disc radius in degrees (~DES focal plane radius, default 1.1).",
    )
    parser.add_argument(
        "--pattern",
        default="position_corrected_*.fits",
        help="Glob for exposure files (expnum = first integer in the filename).",
    )
    parser.add_argument("--self-test", action="store_true", help="Run a geometry sanity check and exit.")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        sys.exit(0)

    if not args.exposure_dir or not args.output_file:
        parser.error("--exposure-dir and --output-file are required")
    if not args.exposure_table:
        raise SystemExit("No exposure table: pass --exposure-table or set $DES_EXPOSURES.")

    from astropy.table import Table

    expnums = _expnums_in_dir(args.exposure_dir, args.pattern)
    if not expnums:
        raise SystemExit(f"No exposures matched '{args.pattern}' under {args.exposure_dir}.")
    print(f"{len(expnums)} exposures found in {args.exposure_dir}")

    tab = Table.read(args.exposure_table)
    table_expnum = np.asarray(tab["expnum"])
    pole = np.asarray(tab["pole"])  # (N, 2) = [ra, dec] in degrees

    mask = np.isin(table_expnum, list(expnums))
    found = table_expnum[mask]
    missing = expnums - set(found.tolist())
    if missing:
        print(f"WARNING: {len(missing)} expnums not in exposure table, skipped: "
              f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")

    discs = [_disc_pixels(args.nside, ra, dec, args.radius) for ra, dec in pole[mask]]
    healpix = np.unique(np.concatenate(discs)) if discs else np.array([], dtype=int)

    np.save(args.output_file, healpix)
    print(f"{len(healpix)} healpixels (nside {args.nside}) saved to {args.output_file}")
