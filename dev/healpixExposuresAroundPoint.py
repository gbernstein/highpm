import argparse
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from healpixFromExposures import _disc_pixels  # noqa: E402
from exposuresFromHealpixList import _radec_from_table  # noqa: E402


def exposures_near_pixel_centers(pix_ra, pix_dec, expnum, exp_ra, exp_dec, radius_deg):
    """expnums whose pointing is within radius_deg of any (pix_ra, pix_dec) center."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    centers = SkyCoord(ra=pix_ra * u.degree, dec=pix_dec * u.degree)
    exposures = SkyCoord(ra=exp_ra * u.degree, dec=exp_dec * u.degree)
    _, d2d, _ = exposures.match_to_catalog_sky(centers)
    return expnum[d2d.degree < radius_deg]


def _self_test():
    import healpy as hp

    nside, ra, dec = 32, 45.0, -20.0
    pix = _disc_pixels(nside, ra, dec, 1.1)
    pix_ra, pix_dec = hp.pix2ang(nside, pix, lonlat=True)

    expnum = np.array([1, 2])
    exp_ra = np.array([45.0, 200.0])
    exp_dec = np.array([-20.0, 10.0])
    found = exposures_near_pixel_centers(pix_ra, pix_dec, expnum, exp_ra, exp_dec, 1.2)
    assert set(found) == {1}, found
    print("self-test ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "From a center (--ra, --dec) and --search-radius, list the covered healpixels, "
            "then list every exposure whose pointing lies within --exposure-radius of any of "
            "those healpixel centers."
        )
    )
    parser.add_argument("--ra", type=float, help="Center RA in degrees.")
    parser.add_argument("--dec", type=float, help="Center Dec in degrees.")
    parser.add_argument("--search-radius", type=float, default=1.1,
                         help="Disc radius in degrees around (ra, dec) for the healpixel list (default 1.1).")
    parser.add_argument("--exposure-radius", type=float, default=1.2,
                         help="Max distance in degrees from a healpixel center to keep an exposure (default 1.2).")
    parser.add_argument("--nside", type=int, default=32, help="HEALPix nside (default 32).")
    parser.add_argument("--exposures-file",
                         help="Exposure metadata table (FITS/HDF5) with expnum + ra/dec or pole.")
    parser.add_argument("--healpix-output", help="Path to save the healpixel list (.npy).")
    parser.add_argument("--exposures-output", help="Path to save the matched expnums (.npy).")
    parser.add_argument("--self-test", action="store_true", help="Run a self-check and exit.")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        sys.exit(0)

    if args.ra is None or args.dec is None or not args.exposures_file \
            or not args.healpix_output or not args.exposures_output:
        parser.error("--ra, --dec, --exposures-file, --healpix-output, and --exposures-output are required")

    import healpy as hp
    from astropy.table import Table

    healpixels = _disc_pixels(args.nside, args.ra, args.dec, args.search_radius)
    np.save(args.healpix_output, healpixels)
    print(f"{len(healpixels)} healpixels (nside {args.nside}) saved to {args.healpix_output}")

    pix_ra, pix_dec = hp.pix2ang(args.nside, healpixels, lonlat=True)
    read_kwargs = {"path": "__astropy_table__"} if args.exposures_file.endswith((".hdf5", ".h5")) else {}
    expnum, exp_ra, exp_dec = _radec_from_table(Table.read(args.exposures_file, **read_kwargs))
    exposures = exposures_near_pixel_centers(pix_ra, pix_dec, expnum, exp_ra, exp_dec, args.exposure_radius)

    np.save(args.exposures_output, exposures)
    print(f"{len(exposures)} exposures saved to {args.exposures_output}")
