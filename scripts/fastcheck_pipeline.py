"""
Wrapper: run the fast checker on the real / fake / overlay PM outputs that
fake_pipeline.py produced for a healpixel. Any run can be toggled via --runs.

The fast checker's `fastcat` is a PM mover file, so this consumes the *_PM_hp*.fits
(and fake_detections) that fake_pipeline wrote -- it does not re-run PM. Run
fake_pipeline first with the same --output-name.
"""

import argparse
import os
import sys
from glob import glob

# Make the project root importable when run directly (mirrors PM.py).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from scripts.PM import select_hp
from scripts.fastChecks import run_fast_checks
from scripts.fake_pipeline import parse_runs  # shared run-toggle parsing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to YAML config.")
    parser.add_argument(
        "--detections", help="Glob of detection catalogs (cleaned_detections_hp*.fits)."
    )
    parser.add_argument("--healpix-npy", help="Path to .npy array of healpixels.")
    parser.add_argument("--index", type=int, help="Index into --healpix-npy.")
    parser.add_argument(
        "--output-name", default="", help="Base path/prefix used by fake_pipeline."
    )
    parser.add_argument(
        "--runs", default="real,fake,overlay", help="Comma list: real,fake,overlay."
    )
    parser.add_argument(
        "--search-radius-arcsec", type=float, default=None, help="Fast-check search radius."
    )
    parser.add_argument("--self-test", action="store_true", help="Run self-check and exit.")
    args = parser.parse_args()

    if args.self_test:
        assert parse_runs("real,overlay") == ["real", "overlay"]
        print("self-test OK")
        return 0

    if not args.config:
        parser.error("--config is required")

    runs = parse_runs(args.runs)

    healpix = int(np.load(args.healpix_npy, allow_pickle=True)[args.index])
    catalog = select_hp(glob(args.detections), healpix)
    if catalog is None:
        print(f"No cleaned_detections_hp{healpix:05d}.fits matching {args.detections!r}.")
        return 0

    base = args.output_name
    fake_det = f"{base}fake_detections_hp{healpix:05d}.fits"

    # (detections to search, PM mover file to check, injection) per run -- must
    # match how fake_pipeline built each PM run.
    plan = {
        "real": (catalog, f"{base}real_PM_hp{healpix:05d}.fits", None),
        "fake": (fake_det, f"{base}fake_PM_hp{healpix:05d}.fits", None),
        "overlay": (catalog, f"{base}overlay_PM_hp{healpix:05d}.fits", fake_det),
    }

    for run in runs:
        detections, fastcat, injection = plan[run]
        missing = [
            p for p in (detections, fastcat, injection) if p and not os.path.exists(p)
        ]
        if missing:
            print(f"Skipping {run}: missing {missing}. Run fake_pipeline first.")
            continue
        print(f"=== fast check: {run} ===")
        run_fast_checks(
            args.config,
            detections,
            fastcat,
            output_prefix=f"{base}{run}_hp{healpix:05d}",
            search_radius_arcsec=args.search_radius_arcsec,
            injection_file=injection,
        )

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
