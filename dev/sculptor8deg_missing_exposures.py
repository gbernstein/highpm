"""Find sculptor8deg position-correction exposures with no output file (e.g. after preemption)."""
import argparse
import os
import re
from glob import glob

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exposures-npy",
        default="/home2/vwetzell/ProperMotion_Sculptor/sculptor8deg_exposures.npy",
        help="Full list of exposures the array job was meant to process.",
    )
    parser.add_argument(
        "--output-path",
        default="/home2/vwetzell/ProperMotion_Sculptor/PositionCorrectedExposureCatalog_8deg/",
        help="Directory where position_corrected_*.fits files are written.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to save the .npy of missing exposure numbers.",
    )
    args = parser.parse_args()

    exposures = np.load(args.exposures_npy, allow_pickle=True).astype(int)

    done = {
        int(m.group(1))
        for f in glob(os.path.join(args.output_path, "position_corrected_*.fits"))
        if (m := re.search(r"(\d+)\.fits$", os.path.basename(f)))
    }

    missing = np.array(sorted(set(exposures) - done), dtype=int)

    np.save(args.output_file, missing)
    print(f"{len(missing)} of {len(exposures)} exposures missing an output file.")
    print(f"Saved to {args.output_file}")
