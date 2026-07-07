"""
Command-line wrapper to run the fast-checker against a catalog of detections.

This script:
- Loads a YAML config
- Reads and cleans the detections catalog (FITS)
- Reads a catalog of objects to check (FITS) with columns: XI, ETA, RA, DEC, PMRA, PMDEC, PARALLAX
- Runs highpm.fast_checks.fast_checker
- Writes results to FITS using highpm.fits_writer.output_fits

Usage:
  python -m scripts.fastChecks CONFIG_YAML DETECTIONS_FITS FASTCAT_FITS [--output-prefix PATH] [--search-radius-arcsec 1.0]
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial

import numpy.lib.recfunctions as rfn

# Ensure project root (package parent) is importable when running this script directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import fitsio
import numpy as np
import yaml

from highpm.fast_checks import fast_checker
from highpm.fits_writer import output_fits
from highpm.pmfit import fit5d

# Only the columns the fast checker + fitter actually touch (fast_checks,
# modest.new_modest_mover, pmfit.fit5d/err2cov). Reading just these avoids
# loading/copying the full ~1 GB detection catalog.
FASTCHECK_COLUMNS = [
    "XI", "ETA", "MJD", "PAR_XI", "PAR_ETA", "EXPNUM",
    "ERRAWIN_WORLD", "NEW_RA_ERR", "NEW_DEC_ERR", "BAND",
    "MAG_AUTO_G", "MAG_AUTO_R", "MAG_AUTO_I", "MAG_AUTO_Z",
    "SPREAD_MODEL", "SPREADERR_MODEL", "DXI_DCOLOR", "DETA_DCOLOR",
]


def _read_fastcheck_cat(path):
    # Force column order (fitsio returns file order) and pack so the detections
    # and injection arrays share an identical dtype for np.concatenate.
    arr = fitsio.read(path, columns=FASTCHECK_COLUMNS, ext=1)
    return rfn.repack_fields(arr[FASTCHECK_COLUMNS])


def _validate_config(config: dict):
    missing: list[str] = []
    # Fitting block requirements
    if "fitting" not in config:
        missing.append("fitting")
    else:
        for k in (
            "time_sep",
            "minSeasons",
            "chisqClip",
            "parallax_prior",
            "color_prior",
            "colorFrac",
            "pm_prior",
            "additional_error",
        ):
            if k not in config["fitting"]:
                missing.append(f"fitting.{k}")

    # Core/global requirements used elsewhere
    for k in ("n_detections", "mjd_ref"):
        if k not in config:
            missing.append(k)

    # Fast-check specific
    if "fastcheck" not in config or "search_radius" not in config["fastcheck"]:
        # We'll allow the wrapper to provide a default; don't fail here, just warn via return
        pass

    if missing:
        raise ValueError("Missing required config keys: " + ", ".join(missing))


def _validate_fastcat_columns(arr: np.ndarray):
    required = ["xi", "eta", "ra", "dec", "pmra", "pmdec", "parallax"]
    missing_cols = [c for c in required if c not in arr.dtype.names]
    if missing_cols:
        raise ValueError(
            "FAST catalog is missing required columns: " + ", ".join(missing_cols)
        )


def run_fast_checks(
    config_path: str,
    detections_path: str,
    fastcat_path: str,
    output_prefix: str | None = None,
    search_radius_arcsec: float | None = None,
    injection_file: str | None = None,
) -> int:
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Search radius is in arcsec throughout (trees/queries use 3600*XI coords).
    if "fastcheck" not in config:
        config["fastcheck"] = {}
    if search_radius_arcsec is not None:
        config["fastcheck"]["search_radius"] = float(search_radius_arcsec)
    # Default if still missing
    if "search_radius" not in config["fastcheck"]:
        config["fastcheck"]["search_radius"] = 1.0  # arcsec

    _validate_config(config)

    if not (np.isin("ra0", config) or np.isin("dec0", config)):
        from highpm.cat_reader import read_cat_header

        cat_header = read_cat_header(detections_path)

        config["ra0"] = cat_header["ra0"]
        config["dec0"] = cat_header["dec0"]

    # Load detections (only the columns the checker/fitter need).
    print(f"Loading detections: {detections_path}")
    cat = _read_fastcheck_cat(detections_path)

    if injection_file is not None:
        print(f"Loading injected fake stars from: {injection_file}")
        injected_stars = _read_fastcheck_cat(injection_file)
        cat = np.concatenate([cat, injected_stars])
        print(f"Catalog size after adding injections: {len(cat)}")


    print(f"Detections loaded: {len(cat)}")

    # completeness_cat = fitsio.read(
    #     "/home/vwetzell/gitrepos/highpm/data/y6a1c.exposures.completeness.fits"
    # )

    # cleanmask = clean_cat(cat, completeness_cat, 0.1)
    cat_idx = np.arange(len(cat))  # [cleanmask]
    # cat = cat[cleanmask]
    print(f"Detections after cleaning: {len(cat)}")

    # Load fast catalog to check
    print(f"Loading fast-check catalog: {fastcat_path}")
    # slow_cat = [
    #     fitsio.read(fastcat_path, ext="modest{i}_movers".format(i=i))
    #     for i in range(1, config["n_modests"] + 1)
    # ]

    slow_cat = []
    for i in range(1, config["n_modests"] + 1):
        try:
            slow_cat.append(
                fitsio.read(fastcat_path, ext="modest{i}_movers".format(i=i))
            )
        except Exception as e:
            print(
                f"Warning: Could not read modest{i}_movers extension from {fastcat_path}: {e}"
            )
            print("This may be expected if no modest movers were found for that fit.")
            print("Continuing without this modest movers extension.")

    slow_cat = np.concatenate(slow_cat)
    fast_cat = fitsio.read(fastcat_path, ext="fast_movers")
    _validate_fastcat_columns(fast_cat)
    print(f"Fast-check objects loaded: {len(fast_cat)}")

    overwrite = False
    if "fastcheck" in fitsio.FITS(fastcat_path)[-1].get_extname():
        overwrite = True
        print("Overwriting existing fastcheck extensions from fastcat.")

    # Build fitter from config
    fit_cfg = config["fitting"]
    part_fit5d = partial(
        fit5d,
        time_sep=fit_cfg["time_sep"],
        minSeasons=fit_cfg["minSeasons"],
        chisqClip=fit_cfg["chisqClip"],
        parallax_prior=fit_cfg["parallax_prior"],
        color_prior=fit_cfg["color_prior"],
        colorFrac=fit_cfg["colorFrac"],
        pm_prior=fit_cfg["pm_prior"],
        additional_error=fit_cfg["additional_error"],
    )

    # Run fast checker
    print(
        "Running fast checker with search radius (deg):",
        config["fastcheck"]["search_radius"],
    )
    pm_arr = fast_checker(cat, fast_cat, slow_cat, part_fit5d, config)

    if pm_arr is None or len(pm_arr) == 0:
        print("No fast-check movers found.")
        return 0

    # Write outputs
    outname = None if output_prefix is None else f"{output_prefix}_fastcheck"
    tbl = output_fits(
        pm_arr,
        cat_idx,
        fastcat_path,
        "fastcheck",
        config=config,
        outputname=fastcat_path if outname is None else f"{outname}_movers.fits",
    )
    print(f"Wrote {len(tbl)} fast-check movers.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a fast-checker against a detections catalog using known object parameters."
        )
    )
    parser.add_argument("config", help="Path to YAML configuration file.")
    parser.add_argument("catalog", help="Path to input detections FITS file.")
    parser.add_argument(
        "fastcat",
        help=(
            "Path to a FITS table of objects to check. Must contain columns: "
            "XI, ETA (deg), RA, DEC (deg), PMRA, PMDEC (arcsec/yr), PARALLAX (deg)."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Optional base path/name for outputs. If omitted, filenames are derived "
            "from the input catalog name."
        ),
    )
    parser.add_argument(
        "--search-radius-arcsec",
        type=float,
        default=None,
        help=(
            "Override search radius (in arcseconds) for matching detections around the "
            "propagated positions. Defaults to 1.0 arcsec if not set and not in config."
        ),
    )

    parser.add_argument(
        "--injection-file",
        default=None,
        help=(
            "Optional path to a FITS file containing injected fake stars to add to the "
            "detections catalog before running the fast checker. Must have same format "
            "as the main detections catalog, with columns including at least RA, DEC, "
            "XI, ETA, PMRA, PMDEC, PARALLAX."
        ),
    )

    args = parser.parse_args()
    # try:
    if True:
        return run_fast_checks(
            args.config,
            args.catalog,
            args.fastcat,
            args.output_prefix,
            args.search_radius_arcsec,
            args.injection_file,
        )
    # except Exception as e:
    #     print(f"Error: {e}")
    #     return 1


if __name__ == "__main__":
    sys.exit(main())
