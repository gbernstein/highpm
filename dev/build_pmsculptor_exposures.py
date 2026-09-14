"""One-off: list reprocessed Sculptor exposures for the PMSculptor pipeline.

Intersects skim exposures with the new GPR2 reprocessing, same matching
positionCorrection.py's _discover_exposures does, and saves the array for
job-array chunking (positionCorrection.py --exposures-npy needs a file on
disk to slice by --index/--chunk-size).

Run on the HPC (needs /data8 access):
    python build_pmsculptor_exposures.py \
        --skims-path "/data8/shared/decampm/[GRIZ]/" \
        --gpr-path "/data8/shared/decampm/GPR2/CAT/[GRIZ]/" \
        --output-file /data8/shared/decampm/PMSculptor/pmsculptor_exposures.npy
"""
import argparse
import os
import re
import sys
import tempfile
from glob import glob

import numpy as np


def _intersect_exposures(skims_path: str, gpr_path: str) -> np.ndarray:
    skim_files = glob(os.path.join(skims_path, "D*.fits"))
    skim_expnums = {
        int(m.group(1))
        for f in skim_files
        if (m := re.search(r"(\d{8})(?=_)", os.path.basename(f)))
    }
    gpr_files = glob(os.path.join(gpr_path, "gpr_*_*.fits"))
    gpr_expnums = {
        int(m.group(1))
        for f in gpr_files
        if (m := re.search(r"(\d{7})(?=_[griz]\.fits$)", os.path.basename(f)))
    }
    if not skim_expnums:
        raise SystemExit(
            f"No skim exposures found. Checked pattern 'D*.fits' under {skims_path!r} "
            f"({len(skim_files)} files matched glob, 0 parsed an expnum)."
        )
    if not gpr_expnums:
        raise SystemExit(
            f"No GPR exposures found. Checked pattern 'gpr_*_<band>.fits' under {gpr_path!r} "
            f"({len(gpr_files)} files matched glob, 0 parsed an expnum)."
        )
    return np.array(sorted(skim_expnums & gpr_expnums), dtype=int)


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as skims, tempfile.TemporaryDirectory() as gpr:
        for f in ("DECam_00123456_r.fits", "DECam_00999999_r.fits"):
            open(os.path.join(skims, f), "w").close()
        for f in ("gpr_x_0123456_r.fits", "gpr_x_0555555_r.fits"):
            open(os.path.join(gpr, f), "w").close()
        result = _intersect_exposures(skims, gpr)
        assert list(result) == [123456], result
    print("self-test ok: skim/GPR2 exposure intersection")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
        sys.exit(0)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skims-path", required=True)
    parser.add_argument("--gpr-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    exposures = _intersect_exposures(args.skims_path, args.gpr_path)
    if len(exposures) == 0:
        raise SystemExit(
            f"No overlapping exposures under {args.skims_path} and {args.gpr_path}."
        )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.save(args.output_file, exposures)
    print(f"{len(exposures)} exposures saved to {args.output_file}")
