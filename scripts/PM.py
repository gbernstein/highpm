"""
This module provides functions for detecting and fitting proper motion from
single epoch detections from ground based surveys.
"""

import argparse
import os
import sys
import re
from functools import partial
from glob import glob
import numpy.lib.recfunctions as rfn

# Ensure project root (package parent) is importable when running this script directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import fitsio
import numpy as np
import yaml

from highpm.cat_reader import clean_cat
from highpm.fast import fast_movers
from highpm.fits_writer import output_fits
from highpm.modest import new_modest_fitter
from highpm.pmfit import fit5d
from highpm.utils import detections_for_removal

# Only the columns clean_cat + the modest/fast fitters (pmfit.fit5d/err2cov)
# actually touch. Reading just these avoids loading/copying the full ~1 GB
# detection catalog. FLAGS/IMAFLAGS_ISO are the extras clean_cat needs on top of
# the fit columns.
PM_COLUMNS = [
    "XI", "ETA", "MJD", "PAR_XI", "PAR_ETA", "EXPNUM",
    "ERRAWIN_WORLD", "BEST_RA_ERR", "BEST_DEC_ERR", "BAND",
    "COLOR", "COLOR_SOURCE",
    "SPREAD_MODEL", "SPREADERR_MODEL", "DXI_DCOLOR", "DETA_DCOLOR",
    "FLAGS", "IMAFLAGS_ISO",
]


def _read_pm_cat(path):
    # Force column order (fitsio returns file order) and pack so the detections
    # and injection arrays share an identical dtype for np.concatenate.
    arr = fitsio.read(path, columns=PM_COLUMNS, ext=1)
    return rfn.repack_fields(arr[PM_COLUMNS])


def _validate_config(config: dict):
    missing = []
    if "fitting" not in config:
        missing.append("fitting")
    else:
        if "pm_prior" not in config["fitting"]:
            config["fitting"]["pm_prior"] = None
        for k in (
            "time_sep",
            "minSeasons",
            "chisqClip",
            "reducedChisqMax",
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


def run_pm(
    config_path: str,
    catname: str,
    output_name: str | None = None,
    injection_file: str | None = None,
) -> int:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _validate_config(config)

    if not (np.isin("ra0", config) or np.isin("dec0", config)):
        from highpm.cat_reader import read_cat_header

        cat_header = read_cat_header(catname)

        config["ra0"] = cat_header["ra0"]
        config["dec0"] = cat_header["dec0"]

    print(f"Loading catalog: {catname}")
    cat = _read_pm_cat(catname)

    if injection_file is not None:
        print(f"Loading injected fake stars from: {injection_file}")
        injected_stars = _read_pm_cat(injection_file)
        cat = np.concatenate([cat, injected_stars])
        print(f"Catalog size after adding injections: {len(cat)}")

    print(f"Detections loaded: {len(cat)}")

    cleanmask = clean_cat(cat)
    cat_idx = np.arange(len(cat))[cleanmask]
    cat = cat[cleanmask]
    print(f"Detections after cleaning: {len(cat)}")

    fitting = config["fitting"]

    part_fit5d = partial(
        fit5d,
        time_sep=fitting["time_sep"],
        minSeasons=fitting["minSeasons"],
        t_season=fitting["t_season"],
        chisqClip=fitting["chisqClip"],
        reducedChisqMax=fitting["reducedChisqMax"],
        parallax_prior=fitting["parallax_prior"],
        color_prior=fitting["color_prior"],
        colorFrac=fitting["colorFrac"],
        pm_prior=fitting["pm_prior"],
        additional_error=fitting["additional_error"],
    )

    n_modests = int(config["n_modests"])
    fit_detection_arr = np.zeros((n_modests, len(cat)), dtype=bool)

    outname = None if output_name is None else output_name

    for i_round in range(n_modests):
        modest_pm_arr = new_modest_fitter(
            cat[~fit_detection_arr[i_round - 1]],
            part_fit5d,
            config,
        )

        detections_idx = cat_idx[~fit_detection_arr[i_round - 1]]

        if len(modest_pm_arr) != 0:

            modest_tbl = output_fits(
                modest_pm_arr,
                detections_idx,
                catname,
                f"modest{i_round + 1}",
                config=config,
                outputname=outname,
            )
            print(f"Writing {len(modest_tbl)} modest movers... (round {i_round + 1})")

            modest_detections = detections_for_removal(modest_pm_arr, config)
            fit_detection_arr[i_round] = fit_detection_arr[i_round - 1].copy()
            temp_detection_arr = fit_detection_arr[
                i_round, ~fit_detection_arr[i_round - 1]
            ]
            if len(modest_detections) != 0:
                temp_detection_arr[modest_detections] = True
            fit_detection_arr[i_round, ~fit_detection_arr[i_round - 1]] = (
                temp_detection_arr
            )

        else:
            print(f"No modest movers found in round {i_round + 1}.")
            fit_detection_arr[-1] = fit_detection_arr[i_round - 1]
            break

        # free intermediates sooner
        del modest_pm_arr

    print("Detections remaining for fast mover search:", np.sum(~fit_detection_arr[-1]))

    fast_pm_arr = fast_movers(cat[~fit_detection_arr[-1]], part_fit5d, config)
    detections_idx = cat_idx[~fit_detection_arr[-1]]

    # fitsio.write(
    #     "remaining_detections.fits",
    #     cat[~fit_detection_arr[-1]],
    #     overwrite=True,
    # )

    if len(fast_pm_arr) != 0:
        fast_tbl = output_fits(
            fast_pm_arr,
            detections_idx,
            catname,
            "fast",
            config=config,
            outputname=outname,
        )
        print(f"Writing {len(fast_tbl)} fast movers...")

    del fast_pm_arr
    print("Done!")
    return 0


def select_hp(paths, hp_number):
    pattern = re.compile(r"hp(\d+)\.fits$")
    for p in paths:
        m = pattern.search(p)
        if m and int(m.group(1)) == hp_number:
            return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect and fit proper motions from single-epoch detections using "
            "modest and fast mover algorithms."
        )
    )
    parser.add_argument("--config", help="Path to YAML configuration file.")
    parser.add_argument("--catalog", help="Path to input catalog FITS file.")
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Optional base path/name for outputs. If omitted, filenames are "
            "derived from the input catalog name."
        ),
    )

    parser.add_argument(
        "--detections", help="Path to directory of detections catalogs."
    )
    parser.add_argument(
        "--index", type=int, help="Index of healpixels in --healpix-npy."
    )
    parser.add_argument(
        "--healpix-npy", help="Path to file with array of heapixels to process."
    )
    parser.add_argument(
        "--injection-file",
        help="Path to file with injected fake stars for this healpixel.",
        default=None,
    )

    args = parser.parse_args()

    if args.catalog is None:
        healpixToDo = np.load(args.healpix_npy, allow_pickle=True)
        print(args.index, type(args.index))
        healpix = healpixToDo[args.index]
        detection_cats = glob(args.detections)
        catalog = select_hp(detection_cats, healpix)
        if catalog is None:
            print(
                f"No cleaned_detections_hp{int(healpix):05d}.fits among "
                f"{len(detection_cats)} files matching {args.detections!r}. Nothing to do."
            )
            return 0
        output_name = args.output_name + f"PM_hp{healpix:05d}.fits"
        injection_file = args.injection_file
    else:
        catalog = args.catalog
        injection_file = args.injection_file
        output_name = args.output_name

    # try:
    if True:
        code = run_pm(args.config, catalog, output_name, injection_file=injection_file)
    # except Exception as e:
    #     print(f"Error: {e}")
    #     return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
