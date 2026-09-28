"""List the healpixels within --radius of Sculptor and the exposures needed
to cover them, restricted to exposures actually available in both the skim
and GPR reprocessing directories.

Unlike build_pmsculptor_exposures.py (which takes the *entire* skim/GPR
overlap), this also intersects with the region around Sculptor, since
--gpr-path here (delve_processed_exposures/cat) is a full-footprint
reprocessing, not a Sculptor-only one like GPR2 was.

Run on the HPC (needs /data8 access):
    python build_pmsculptor8deg_exposures.py \
        --skims-path "/data8/shared/decampm/[GRIZ]/" \
        --gpr-path "/data8/shared/decampm/delve_processed_exposures/cat/[griz]/" \
        --des-exposures /home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5 \
        --radius 8.0 \
        --healpix-output /data8/shared/decampm/PMSculptor_8deg/pmsculptor8deg_healpix.npy \
        --exposures-output /data8/shared/decampm/PMSculptor_8deg/pmsculptor8deg_exposures.npy
"""
import argparse
import os
import sys
import tempfile

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from healpixFromExposures import _disc_pixels  # noqa: E402
from exposuresFromHealpixList import _radec_from_table, exposures_for_healpixels  # noqa: E402
from build_pmsculptor_exposures import _intersect_exposures, _skimmed_expnums  # noqa: E402

# Center of the Sculptor dwarf spheroidal, matching dev/plot_sculptor_finished_healpix.py.
SCULPTOR_RA, SCULPTOR_DEC = 15.038750, -33.709000


def _self_test() -> None:
    import healpy as hp
    from astropy.table import Table

    with tempfile.TemporaryDirectory() as skims, tempfile.TemporaryDirectory() as gpr:
        nside = 32
        pix = _disc_pixels(nside, SCULPTOR_RA, SCULPTOR_DEC, 1.0)
        ra_in, dec_in = hp.pix2ang(nside, int(pix[0]), lonlat=True)

        des_exposures = os.path.join(gpr, "exposures.hdf5")
        Table({
            "expnum": [111, 222, 333],
            "pole": [[ra_in, dec_in], [200.0, 10.0], [ra_in, dec_in]],
        }).write(des_exposures, path="__astropy_table__")

        for f in ("DECam_00000111_r.fits", "DECam_00000333_r.fits"):
            open(os.path.join(skims, f), "w").close()
        # 222 is in the region but has no skim; 333 has a skim/GPR match but
        # isn't in the region -- neither should survive the intersection.
        for f in ("gpr_x_0000111_r.fits", "gpr_x_0000222_r.fits"):
            open(os.path.join(gpr, f), "w").close()

        healpix, exposures = _build(skims, gpr, des_exposures, SCULPTOR_RA, SCULPTOR_DEC, 1.0, nside)
        assert int(pix[0]) in healpix, healpix
        assert list(exposures) == [111], exposures
    print("self-test ok: sculptor8deg healpix/exposure selection")


def _build(skims_path, gpr_path, des_exposures, ra, dec, radius_deg, nside, complete_only=False):
    healpixels = _disc_pixels(nside, ra, dec, radius_deg)

    from astropy.table import Table

    read_kwargs = {"path": "__astropy_table__"} if des_exposures.endswith((".hdf5", ".h5")) else {}
    expnum, exp_ra, exp_dec = _radec_from_table(Table.read(des_exposures, **read_kwargs))
    available = _intersect_exposures(skims_path, gpr_path)

    if complete_only:
        # Keep only healpixels whose every skimmed exposure has a GPR file, and
        # only the exposures those healpixels need. Exposures with no skim
        # can't be processed by anyone, so they don't count against a healpix.
        from highpm.detection_packaging import get_exposures_near_healpix

        skimmed = _skimmed_expnums(skims_path)
        avail = set(available.tolist())
        kept, needed_sets = [], []
        for h in healpixels:
            need = set(get_exposures_near_healpix(int(h), exp_ra, exp_dec, expnum, nside=nside).tolist()) & skimmed
            if need and need <= avail:
                kept.append(int(h))
                needed_sets.append(need)
        healpixels = np.array(kept, dtype=healpixels.dtype)
        exposures = np.array(sorted(set().union(*needed_sets)) if needed_sets else [], dtype=int)
        return healpixels, exposures

    needed = exposures_for_healpixels(healpixels, expnum, exp_ra, exp_dec, nside)
    exposures = np.array(sorted(set(needed.tolist()) & set(available.tolist())), dtype=int)
    return healpixels, exposures


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
        sys.exit(0)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skims-path", required=True)
    parser.add_argument("--gpr-path", required=True)
    parser.add_argument("--des-exposures", required=True,
                         help="Exposure metadata table (FITS/HDF5) with expnum + ra/dec or pole.")
    parser.add_argument("--ra", type=float, default=SCULPTOR_RA, help="Center RA in degrees (default: Sculptor).")
    parser.add_argument("--dec", type=float, default=SCULPTOR_DEC, help="Center Dec in degrees (default: Sculptor).")
    parser.add_argument("--radius", type=float, default=8.0, help="Disc radius in degrees (default 8.0).")
    parser.add_argument("--nside", type=int, default=32, help="HEALPix nside (default 32).")
    parser.add_argument("--healpix-output", required=True)
    parser.add_argument("--exposures-output", required=True)
    parser.add_argument("--complete-only", action="store_true",
                        help="Only keep healpixels whose exposures are all in --gpr-path, "
                             "and only the exposures those healpixels need.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    healpixels, exposures = _build(
        args.skims_path, args.gpr_path, args.des_exposures,
        args.ra, args.dec, args.radius, args.nside, args.complete_only,
    )
    if len(exposures) == 0:
        raise SystemExit(
            f"No exposures within {args.radius} deg of ({args.ra}, {args.dec}) overlap "
            f"both {args.skims_path} and {args.gpr_path}."
        )

    os.makedirs(os.path.dirname(args.healpix_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.exposures_output), exist_ok=True)
    np.save(args.healpix_output, healpixels)
    np.save(args.exposures_output, exposures)
    print(f"{len(healpixels)} healpixels (nside {args.nside}, {args.radius} deg around "
          f"({args.ra}, {args.dec})) saved to {args.healpix_output}")
    print(f"{len(exposures)} exposures saved to {args.exposures_output}")
