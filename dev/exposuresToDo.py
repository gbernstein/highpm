import argparse
import os
import re
import sys
from glob import glob

import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Create a .npy file with an array of exposure numbers that have yet to be processes."
        )
    )

    parser.add_argument(
        "--gpr-path",
        default=None,
        help=(
            "Directory or glob prefix for GPR FITS files names like 'gpr_*{expnum:07d}_<band>.fits'"
        ),
    )

    parser.add_argument(
        "--output-path",
        default=None,
        help=("Directory or glob prefix for finished exposures."),
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help=("Path to output .npy file containing an array of exposures left to do."),
    )

    parser.add_argument(
        "--healpix",
        type=int,
        default=None,
        help="(Optional) Healpix number to limit the search to.",
    )

    parser.add_argument(
        "--nside",
        type=int,
        default=32,
        help="(Optional) Nside for the healpix number provided.",
    )

    parser.add_argument(
        "--pointing-file",
        type=str,
        default=None,
        help="(Optional) Path to delveExposures.hdf5 (astropy Table) with pointing info.",
    )

    parser.add_argument("--ra", type=float, default=None, help="Cone-search center RA (deg).")
    parser.add_argument("--dec", type=float, default=None, help="Cone-search center Dec (deg).")
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Cone-search radius (deg). With --ra/--dec, save all exposures (all bands) within it.",
    )

    args = parser.parse_args()

    cone_mode = args.ra is not None and args.dec is not None and args.radius is not None
    if cone_mode:
        if args.pointing_file is None:
            raise SystemExit("--ra/--dec/--radius cone search requires --pointing-file.")
    else:
        if args.gpr_path is None or args.output_path is None:
            raise SystemExit(
                "--gpr-path and --output-path are required (unless doing a --ra/--dec/--radius cone search)."
            )
        if args.healpix is not None and args.pointing_file is None:
            raise SystemExit(
                "If --healpix is provided, --pointing-file must also be provided."
            )

    def _scan_expnums(directory, pattern, regex):
        expnums = set()
        for f in glob(os.path.join(directory, pattern)):
            m = re.search(regex, os.path.basename(f))
            if m:
                expnums.add(int(m.group(1)))
        return expnums

    def _cone_search(pointing_file, ra0, dec0, radius_deg):
        import healpy as hp
        from astropy.table import Table

        cat = Table.read(pointing_file, path="__astropy_table__")
        center = hp.ang2vec(ra0, dec0, lonlat=True)
        vecs = hp.ang2vec(cat["pole"][:, 0], cat["pole"][:, 1], lonlat=True)
        sep = np.degrees(np.arccos(np.clip(vecs @ center, -1.0, 1.0)))
        keep = sep <= radius_deg
        return np.unique(np.asarray(cat["expnum"])[keep]), cat["pole"][:,0][keep], cat["pole"][:,1][keep]

    def _discover_exposures(
        gpr_path: str,
        output_path: str,
        healpix: int = None,
        nside: int = 32,
        pointing_file: str = None,
    ) -> np.ndarray:

        if healpix is not None and pointing_file is not None:
            import healpy as hp
            from astropy.table import Table

            # delveExposures.hdf5: astropy Table; pointing center is the "pole" (ra, dec) column
            exposure_cat = Table.read(pointing_file, path="__astropy_table__")
            ra = exposure_cat["pole"][:, 0]
            dec = exposure_cat["pole"][:, 1]

            # Determine which exposures have centers in the healpix or its neighbors
            neighbors = hp.get_all_neighbours(nside, healpix)
            hp_indices = [healpix, *neighbors[neighbors >= 0]]

            expo_healpix = hp.ang2pix(nside, ra, dec, lonlat=True)

            valid_exposures = exposure_cat[np.isin(expo_healpix, hp_indices)]["expnum"]

            print(
                f"Found {len(valid_exposures)} exposures in healpix {healpix} and neighbors."
            )

        # position_corrected_*.fits -> first digit run is the expnum
        output_expnums = _scan_expnums(output_path, "position_corrected_*.fits", r"(\d+)")

        # gpr_*{expnum:07d}_<band>.fits -> 7 digits before the _<band>.fits suffix (DES griz)
        gpr_expnums = _scan_expnums(gpr_path, "gpr_*_*.fits", r"(\d{7})(?=_[griz]\.fits$)")

        if not gpr_expnums:
            raise SystemExit(
                "No GPR exposures found. Checked pattern 'gpr_*_<band>.fits' under gpr-path."
            )

        print(list(gpr_expnums)[:10])
        print(list(output_expnums)[:10])
        # Intersection to ensure both inputs exist per exposure
        if healpix is not None and pointing_file is not None:
            inter = np.setdiff1d(
                np.array(list(valid_exposures)),
                np.array(list(output_expnums)),
            )
        else:
            inter = np.setdiff1d(
                np.array(list(gpr_expnums)), np.array(list(output_expnums))
            )
        if len(inter) == 0:
            raise SystemExit(
                f"No exposures left to do between output ({len(output_expnums)}) and GPR ({len(gpr_expnums)})."
            )

        exposures = inter

        return np.array(list(exposures))

    if cone_mode:
        expos, expo_ra, expo_dec = _cone_search(args.pointing_file, args.ra, args.dec, args.radius)
        print(
            f"Found {len(expos)} exposures within {args.radius} deg of "
            f"({args.ra}, {args.dec})."
        )
        # ponytail: GPR/output filtering only when both paths given; else save the raw cone.
        if args.gpr_path is not None:
            gpr_expnums = _scan_expnums(
                args.gpr_path, "gpr_*_*.fits", r"(\d{7})(?=_[griz]\.fits$)"
            )
            if not gpr_expnums:
                raise SystemExit(
                    "No GPR exposures found. Checked pattern 'gpr_*_<band>.fits' under gpr-path."
                )
            expos = np.intersect1d(expos, list(gpr_expnums))
        if args.output_path is not None:
            output_expnums = _scan_expnums(
                args.output_path, "position_corrected_*.fits", r"(\d+)"
            )
            expos = np.setdiff1d(expos, list(output_expnums))
            print(
                f"{len(expos)} left after dropping {len(output_expnums)} already-done exposures."
            )
    else:
        print(f"GPR Path: {args.gpr_path}")
        print(f"Output Path: {args.output_path}")
        expos = _discover_exposures(
            args.gpr_path, args.output_path, args.healpix, args.nside, args.pointing_file
        )

    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(
            expo_ra,
            expo_dec,
            s=1,
            c="r",
    )
    plt.savefig(args.output_file+".png")
    plt.show()

    np.save(args.output_file, expos)

    print(f"{len(expos)} exposure numbers saved to {args.output_file}")
