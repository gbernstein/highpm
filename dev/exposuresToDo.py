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
        required=True,
        help=(
            "Directory or glob prefix for GPR FITS files names like 'gpr_*{expnum:07d}_<band>.fits'"
        ),
    )

    parser.add_argument(
        "--output-path",
        required=True,
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
        help="(Optional) Path to a file containing pointing information.",
    )

    args = parser.parse_args()

    if args.healpix is not None and args.pointing_file is None:
        raise SystemExit(
            "If --healpix is provided, --pointing-file must also be provided."
        )

    def _discover_exposures(
        gpr_path: str,
        output_path: str,
        healpix: int = None,
        nside: int = 32,
        pointing_file: str = None,
    ) -> np.ndarray:

        if healpix is not None and pointing_file is not None:
            import healpy as hp
            import fitsio

            # Load pointing file
            exposure_cat = fitsio.read(pointing_file, ext=1)

            # Determine which exposures have centers in the healpix or its neighbors
            hp_indices = []
            hp_indices.append(healpix)
            neighbors = hp.get_all_neighbours(
                nside,
                healpix,
            )
            for n in neighbors:
                if n >= 0:
                    hp_indices.append(n)

            expo_healpix = hp.ang2pix(
                nside,
                np.radians(90.0 - exposure_cat["dec"]),
                np.radians(exposure_cat["ra"]),
            )

            valid_exposures = exposure_cat[np.isin(expo_healpix, list(hp_indices))][
                "expnum"
            ]

            print(
                f"Found {len(valid_exposures)} exposures in healpix {healpix} and neighbors."
            )

        # Skims look like: D*{expnum:08d}_*.fits -> extract 8 digits before underscore
        output_files = glob(os.path.join(output_path, "position_corrected_*.fits"))
        output_expnums = set()
        for f in output_files:
            bn = os.path.basename(f)
            m = re.search(r"(\d+)", bn)
            if m:
                try:
                    output_expnums.add(int(m.group(1)))
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

    print(f"GPR Path: {args.gpr_path}")
    print(f"Output Path: {args.output_path}")

    expos = _discover_exposures(
        args.gpr_path, args.output_path, args.healpix, args.nside, args.pointing_file
    )
    np.save(args.output_file, expos)

    print(f"{len(expos)} exposure numbers saved to {args.output_file}")
