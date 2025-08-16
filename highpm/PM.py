"""
This module provides functions for detecting and fitting proper motion from
single epoch detections from ground based surveys.
"""

import sys
from functools import partial
import yaml
import numpy as np
from pmfit import fit5d
from modest import new_modest_fitter
from fast import fast_movers
from utils import detections_for_removal
from cat_reader import clean_cat, read_cat_data, read_cat_header
from fits_writer import output_fits, output_fits_mask


if __name__ == "__main__":
    help = "Still need to write the help section"

    if len(sys.argv) == 3:
        if sys.argv[1] == "-h" or sys.argv[1] == "--help":
            print(help)
            sys.exit(1)
        config_path = sys.argv[1]
        catname = sys.argv[2]
    else:
        print(help)
        sys.exit(1)

    header = read_cat_header(catname)

    cat = read_cat_data(catname)

    cat = clean_cat(cat)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    fitting = config["fitting"]

    time_sep = fitting["time_sep"]
    minSeasons = fitting["minSeasons"]
    chisqClip = fitting["chisqClip"]
    parallax_prior = fitting["parallax_prior"]
    color_prior = fitting["color_prior"]
    colorFrac = fitting["colorFrac"]
    pm_prior = fitting["pm_prior"]
    additional_error = fitting["additional_error"]

    part_fit5d = partial(
        fit5d,
        time_sep=time_sep,
        minSeasons=minSeasons,
        chisqClip=chisqClip,
        parallax_prior=parallax_prior,
        color_prior=color_prior,
        colorFrac=colorFrac,
        pm_prior=pm_prior,
        additional_error=additional_error,
    )

    n_modests = config["n_modests"]

    fit_detection_arr = np.zeros((n_modests, len(cat)), dtype=bool)

    for _ in range(n_modests):
        modest_pm_arr = new_modest_fitter(
            cat[~fit_detection_arr[_ - 1]],
            part_fit5d,
            config,
        )

        if len(modest_pm_arr) != 0:
            modest_tbl = output_fits(modest_pm_arr, catname, f"modest{_ + 1}")

            print(f"Writing {len(modest_tbl)} modest movers... (round {_ + 1})")

            modest_detections = detections_for_removal(modest_pm_arr, config)
            if len(modest_detections) != 0:
                fit_detection_arr[_] = fit_detection_arr[_ - 1].copy()
                temp_detection_arr = fit_detection_arr[_, ~fit_detection_arr[_ - 1]]
                temp_detection_arr[modest_detections] = True
                fit_detection_arr[_, ~fit_detection_arr[_ - 1]] = temp_detection_arr

        else:
            print(f"No modest movers found in round {_ + 1}.")
            modest_tbl = None
            modest_detections = None

        del modest_pm_arr
        del modest_tbl
        del modest_detections

    output_fits_mask(fit_detection_arr, n_modest=n_modests, filename=catname)

    fast_pm_arr = fast_movers(cat[~fit_detection_arr[-1]], part_fit5d, config)

    if len(fast_pm_arr) != 0:
        fast_tbl = output_fits(fast_pm_arr, catname, "fast")

        print(f"Writing {len(fast_tbl)} fast movers...")

    del fast_pm_arr

    print("Done!")

    sys.exit(0)
