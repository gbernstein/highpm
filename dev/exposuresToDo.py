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
        help=(
            "Directory or glob prefix for finished exposures."
        ),
    )

    parser.add_argument(
         "--output-file",
         required=True,
         help=(
             "Path to output .npy file containing an array of exposures left to do."
         ),
    )

    args = parser.parse_args()

    def _discover_exposures(gpr_path: str,output_path: str) -> np.ndarray:
        # Skims look like: D*{expnum:08d}_*.fits -> extract 8 digits before underscore
        output_files = glob(os.path.join(output_path, "position_corrected_*.fits"))
        print(len(output_files))
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
        inter = np.setdiff1d(np.array(list(gpr_expnums)), np.array(list(output_expnums)))
        if len(inter)==0:
            raise SystemExit(
                f"No exposures left to do between output ({len(skim_expnums)}) and GPR ({len(gpr_expnums)})."
            )

        exposures = inter

        return np.array(list(exposures))

    print(f"GPR Path: {args.gpr_path}")
    print(f"Output Path: {args.output_path}")

    expos = _discover_exposures(args.gpr_path,args.output_path)
    np.save(args.output_file,expos)

    print(f"{len(expos)} exposure numbers saved to {args.output_file}")
