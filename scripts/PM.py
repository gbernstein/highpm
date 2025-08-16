"""
This module provides functions for detecting and fitting proper motion from
single epoch detections from ground based surveys.
"""

import argparse
import os
import sys
from functools import partial

# Ensure project root (package parent) is importable when running this script directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import yaml

from highpm.cat_reader import clean_cat, read_cat_data
from highpm.fast import fast_movers
from highpm.fits_writer import output_fits, output_fits_mask
from highpm.modest import new_modest_fitter
from highpm.pmfit import fit5d
from highpm.utils import detections_for_removal


def _validate_config(config: dict):
    missing = []
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
    if "n_modests" not in config:
        missing.append("n_modests")
    if missing:
        raise ValueError("Missing required config keys: " + ", ".join(missing))


def run_pm(config_path: str, catname: str, output_prefix: str | None = None) -> int:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _validate_config(config)

    print(f"Loading catalog: {catname}")
    cat = read_cat_data(catname)
    print(f"Detections loaded: {len(cat)}")

    cat = clean_cat(cat)
    print(f"Detections after cleaning: {len(cat)}")

    fitting = config["fitting"]

    part_fit5d = partial(
        fit5d,
        time_sep=fitting["time_sep"],
        minSeasons=fitting["minSeasons"],
        chisqClip=fitting["chisqClip"],
        parallax_prior=fitting["parallax_prior"],
        color_prior=fitting["color_prior"],
        colorFrac=fitting["colorFrac"],
        pm_prior=fitting["pm_prior"],
        additional_error=fitting["additional_error"],
    )

    n_modests = int(config["n_modests"])
    fit_detection_arr = np.zeros((n_modests, len(cat)), dtype=bool)

    for i_round in range(n_modests):
        modest_pm_arr = new_modest_fitter(
            cat[~fit_detection_arr[i_round - 1]],
            part_fit5d,
            config,
        )

        if len(modest_pm_arr) != 0:
            outname = (
                None
                if output_prefix is None
                else f"{output_prefix}_modest{i_round + 1}"
            )
            modest_tbl = output_fits(
                modest_pm_arr, catname, f"modest{i_round + 1}", outputname=outname
            )
            print(f"Writing {len(modest_tbl)} modest movers... (round {i_round + 1})")

            modest_detections = detections_for_removal(modest_pm_arr, config)
            if len(modest_detections) != 0:
                fit_detection_arr[i_round] = fit_detection_arr[i_round - 1].copy()
                temp_detection_arr = fit_detection_arr[
                    i_round, ~fit_detection_arr[i_round - 1]
                ]
                temp_detection_arr[modest_detections] = True
                fit_detection_arr[i_round, ~fit_detection_arr[i_round - 1]] = (
                    temp_detection_arr
                )
        else:
            print(f"No modest movers found in round {i_round + 1}.")

        # free intermediates sooner
        del modest_pm_arr

    outmask_name = output_prefix if output_prefix is not None else None
    output_fits_mask(
        fit_detection_arr, n_modest=n_modests, filename=catname, outputname=outmask_name
    )

    fast_pm_arr = fast_movers(cat[~fit_detection_arr[-1]], part_fit5d, config)

    if len(fast_pm_arr) != 0:
        outname = None if output_prefix is None else f"{output_prefix}_fast"
        fast_tbl = output_fits(fast_pm_arr, catname, "fast", outputname=outname)
        print(f"Writing {len(fast_tbl)} fast movers...")

    del fast_pm_arr
    print("Done!")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect and fit proper motions from single-epoch detections using "
            "modest and fast mover algorithms."
        )
    )
    parser.add_argument("config", help="Path to YAML configuration file.")
    parser.add_argument("catalog", help="Path to input catalog FITS file.")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Optional base path/name for outputs. If omitted, filenames are "
            "derived from the input catalog name."
        ),
    )

    args = parser.parse_args()
    try:
        code = run_pm(args.config, args.catalog, args.output_prefix)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
