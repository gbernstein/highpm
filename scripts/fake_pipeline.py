"""
Wrapper: given a healpixel, generate fake detections and run the PM fitter on
two detection sets -- real, and injection (real + fake injected). Either run
can be toggled via --runs. Thin orchestration over existing code.
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
import yaml

from scripts.PM import run_pm, select_hp
from highpm.cat_reader import clean_cat, read_cat_header
from highpm.fake_star import generate_fake_star_catalog
from highpm.fake_detections import generate_fake_detections

ALL_RUNS = ("real", "injection")

# ponytail: empirically ~15 recovered fake detections per injected star (epochs x
# completeness x FOV survival). Only sets the auto-density scale; --density overrides.
DET_PER_STAR = 15.0


def default_density(n_real, nside):
    """Injection star density (stars/deg^2) giving ~1/32 the real detection density.

    n_real should be the count *after* clean_cat, not the raw row count --
    clean_cat rejects the majority of rows (FLAGS/IMAFLAGS_ISO/SPREAD_MODEL),
    so a raw-row-count input overstates the real detection density target.
    """
    real_det_density = n_real / hp.nside2pixarea(nside, degrees=True)
    return real_det_density / 32.0 / DET_PER_STAR


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
        help="Injection star density (stars/deg^2). Default: ~1/32 the real detection density.",
    )
    parser.add_argument(
        "--nside", type=int, default=None, help="Override nside (else read from header)."
    )
    parser.add_argument(
        "--runs",
        default="real,injection",
        help="Comma list of runs to execute: real,injection.",
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

    # Injected fakes (with blending-with-real removed, see generate_fake_detections)
    # are only needed for the injection run; generate once.
    fake_det = f"{base}fake_detections_hp{healpix:05d}.fits"
    if "injection" in runs:
        density = args.density
        if density is None:
            # Density target is "~1/32 the real DETECTION density" -- use the count
            # after clean_cat (FLAGS/IMAFLAGS_ISO/SPREAD_MODEL/TRAP_FLAG + exposure
            # thinning), not the raw row count, or a target based on the raw count
            # is inflated by however many rows clean_cat would have rejected.
            with open(args.config, "r") as f:
                density_config = yaml.safe_load(f)
            cat_header = read_cat_header(catalog)
            density_config.setdefault("ra0", cat_header["ra0"])
            density_config.setdefault("dec0", cat_header["dec0"])
            wanted = ["FLAGS", "IMAFLAGS_ISO", "SPREAD_MODEL", "SPREADERR_MODEL",
                      "TRAP_FLAG", "EXPNUM", "MJD"]
            available = fitsio.FITS(catalog)[1].get_colnames()
            clean_cols = fitsio.read(
                catalog, columns=[c for c in wanted if c in available], ext=1
            )
            n_real = int(np.sum(clean_cat(clean_cols, density_config)))
            density = default_density(n_real, nside)
            print(f"Auto injection density: {density:.0f}/deg^2 (~1/32 cleaned real det density)")
        star_cat = f"{base}fake_stars_hp{healpix:05d}.fits"
        generate_fake_star_catalog(density, healpix, nside, star_cat)
        generate_fake_detections(star_cat, catalog, healpix, nside, fake_det)

    if "real" in runs:
        run_pm(args.config, catalog, f"{base}real_PM_hp{healpix:05d}.fits")
    if "injection" in runs:
        run_pm(
            args.config,
            catalog,
            f"{base}injection_PM_hp{healpix:05d}.fits",
            injection_file=fake_det,
        )

    print("Done!")
    return 0


def _self_test():
    assert parse_runs("real") == ["real"]
    assert parse_runs("real,injection") == list(ALL_RUNS)
    try:
        parse_runs("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("parse_runs should reject unknown runs")
    # ~2.81M cleaned real dets in hp09413 (nside=32) -> ~1/32 real det density in injected dets.
    d = default_density(2_807_296, 32)
    assert 1500 < d < 2000, d
    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
