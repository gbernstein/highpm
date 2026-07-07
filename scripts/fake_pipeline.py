"""
Wrapper: given a healpixel, generate fake detections and run the PM fitter on up
to three detection sets -- real, fake, and overlay (real + fake injected). Any of
the three runs can be toggled via --runs. Thin orchestration over existing code.
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

import fitsio
import healpy as hp
import numpy as np

from scripts.PM import run_pm, select_hp
from highpm.cat_reader import read_cat_header
from highpm.fake_star import generate_fake_star_catalog
from highpm.fake_detections import generate_fake_detections

ALL_RUNS = ("real", "fake", "overlay")

# ponytail: empirically ~15 recovered fake detections per injected star (epochs x
# completeness x FOV survival). Only sets the auto-density scale; --density overrides.
DET_PER_STAR = 15.0


def default_density(n_real, nside):
    """Fake-star density (stars/deg^2) giving ~1/8 the real detection density."""
    real_det_density = n_real / hp.nside2pixarea(nside, degrees=True)
    return real_det_density / 8.0 / DET_PER_STAR


def parse_runs(runs_str):
    runs = [r.strip() for r in runs_str.split(",") if r.strip()]
    bad = [r for r in runs if r not in ALL_RUNS]
    if bad:
        raise ValueError(f"Unknown run(s) {bad}; choose from {ALL_RUNS}.")
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to YAML config.")
    parser.add_argument(
        "--detections", help="Glob of detection catalogs (cleaned_detections_hp*.fits)."
    )
    parser.add_argument("--healpix-npy", help="Path to .npy array of healpixels.")
    parser.add_argument("--index", type=int, help="Index into --healpix-npy.")
    parser.add_argument(
        "--output-name", default="", help="Base path/prefix for all outputs."
    )
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help="Fake star density (stars/deg^2). Default: ~1/8 the real detection density.",
    )
    parser.add_argument(
        "--nside", type=int, default=None, help="Override nside (else read from header)."
    )
    parser.add_argument(
        "--runs",
        default="real,fake,overlay",
        help="Comma list of runs to execute: real,fake,overlay.",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run toggle-parsing self-check and exit."
    )
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if not args.config:
        parser.error("--config is required")

    runs = parse_runs(args.runs)

    healpix = int(np.load(args.healpix_npy, allow_pickle=True)[args.index])
    catalog = select_hp(glob(args.detections), healpix)
    if catalog is None:
        print(f"No cleaned_detections_hp{healpix:05d}.fits matching {args.detections!r}.")
        return 0

    nside = args.nside or int(read_cat_header(catalog)["NSIDE"])
    base = args.output_name

    # Fakes are needed for the fake and overlay runs; generate once.
    fake_det = f"{base}fake_detections_hp{healpix:05d}.fits"
    if "fake" in runs or "overlay" in runs:
        density = args.density
        if density is None:
            n_real = fitsio.FITS(catalog)[1].get_nrows()
            density = default_density(n_real, nside)
            print(f"Auto fake-star density: {density:.0f}/deg^2 (~1/8 real det density)")
        star_cat = f"{base}fake_stars_hp{healpix:05d}.fits"
        generate_fake_star_catalog(density, healpix, nside, star_cat)
        generate_fake_detections(star_cat, catalog, healpix, nside, fake_det)

    if "real" in runs:
        run_pm(args.config, catalog, f"{base}real_PM_hp{healpix:05d}.fits")
    if "fake" in runs:
        run_pm(args.config, fake_det, f"{base}fake_PM_hp{healpix:05d}.fits")
    if "overlay" in runs:
        run_pm(
            args.config,
            catalog,
            f"{base}overlay_PM_hp{healpix:05d}.fits",
            injection_file=fake_det,
        )

    print("Done!")
    return 0


def _self_test():
    assert parse_runs("real,overlay") == ["real", "overlay"]
    assert "fake" not in parse_runs("real,overlay")  # fake generation would be skipped
    assert parse_runs("real,fake,overlay") == list(ALL_RUNS)
    try:
        parse_runs("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("parse_runs should reject unknown runs")
    # ~4.48M real dets in an nside=32 pixel -> ~1/8 real det density in fake dets.
    d = default_density(4_477_687, 32)
    assert 9000 < d < 13000, d
    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
