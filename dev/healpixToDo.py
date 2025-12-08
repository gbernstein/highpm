import argparse
import os
import re
import sys
from glob import glob

import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Create a .npy file with an array of exposure numbers that have yet to be proce"
        )
    )

    parser = argparse.ArgumentParser(
        description=(
            "Create a .npy file with an array of healpixel numbers that have yet to be processes."
        )
    )

    parser.add_argument(
        "--healpix-path",
        required=True,
        help=(
            "Directory or glob prefix for Coadd FITS files names like ''"
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

    def _discover_healpixels(healpix_path: str,output_path: str) -> np.ndarray:
        output_files = glob(os.path.join(output_path, "*.fits"))
        print(len(output_files), " healpixels completed.")
        output_healpix = set()
        for f in output_files:
            bn = os.path.basename(f)
            m = re.search(r"(\d+)", bn)
            if m:
                try:
                    output_healpix.add(int(m.group(1)))
                except ValueError:
                    pass

        healpix_files = glob(os.path.join(healpix_path, "*.fits"))
        print(len(healpix_files), " total healpixels.")
        all_healpix = set()
        for f in healpix_files:
            bn = os.path.basename(f)
            m = re.search(r"(\d+)", bn)
            if m:
                try:
                    all_healpix.add(int(m.group(1)))
                except ValueError:
                    pass


        if len(all_healpix)==0:
            raise SystemExit(
                "No coadd healpixels found."
            )

        # Intersection to ensure both inputs exist per exposure
        inter = np.setdiff1d(np.array(list(all_healpix)), np.array(list(output_healpix)))
        if len(inter)==0:
            raise SystemExit(
                f"No exposures left to do."
            )

        healpixels = inter

        return np.array(list(healpixels))

    print(f"Coadd Path: {args.healpix_path}")
    print(f"Output Path: {args.output_path}")

    healpix = _discover_healpixels(args.healpix_path,args.output_path)
    np.save(args.output_file,healpix)

    print(f"{len(healpix)} exposure numbers saved to {args.output_file}")


