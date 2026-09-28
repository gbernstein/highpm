"""Catalog reading, config validation and fitter construction shared by the
main PM run (scripts/PM.py) and the fast checker (scripts/fastChecks.py), so
the two can't drift apart in what detections they use or how they fit them."""

from functools import partial

import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn

from .pmfit import fit5d

# Only the columns clean_cat + the modest/fast fitters (pmfit.fit5d/err2cov)
# actually touch. Reading just these avoids loading/copying the full ~1 GB
# detection catalog. FLAGS/IMAFLAGS_ISO/TRAP_FLAG are the extras clean_cat
# needs on top of the fit columns.
PM_COLUMNS = [
    "XI", "ETA", "MJD", "PAR_XI", "PAR_ETA", "EXPNUM",
    "BEST_RA", "BEST_DEC", "BEST_RA_ERR", "BEST_DEC_ERR", "BEST_RA_DEC_CORR", "BAND",
    "COLOR", "COLOR_SOURCE",
    "SPREAD_MODEL", "SPREADERR_MODEL", "DXI_DCOLOR", "DETA_DCOLOR",
    "FLAGS", "IMAFLAGS_ISO", "TRAP_FLAG",
]

FIT_KEYS = (
    "time_sep",
    "minSeasons",
    "t_season",
    "chisqClip",
    "reducedChisqMax",
    "parallax_prior",
    "color_prior",
    "colorFrac",
    "pm_prior",
)


def read_pm_cat(path):
    # Force column order (fitsio returns file order) and pack so the detections
    # and injection arrays share an identical dtype for np.concatenate.
    # TRAP_FLAG is absent from pre-reprocessing catalogs and from fake-star
    # injections (which can't have a real charge-trap shift), so it's read
    # only when present and defaulted to False (not flagged) otherwise.
    available = fitsio.FITS(path)[1].get_colnames()
    has_trap = "TRAP_FLAG" in available
    columns = PM_COLUMNS if has_trap else [c for c in PM_COLUMNS if c != "TRAP_FLAG"]
    arr = fitsio.read(path, columns=columns, ext=1)
    if not has_trap:
        arr = rfn.append_fields(arr, "TRAP_FLAG", np.zeros(len(arr), dtype="?"), usemask=False)
    return rfn.repack_fields(arr[PM_COLUMNS])


def validate_config(config: dict):
    missing = []
    if "fitting" not in config:
        missing.append("fitting")
    else:
        if "pm_prior" not in config["fitting"]:
            config["fitting"]["pm_prior"] = None
        for k in FIT_KEYS:
            if k not in config["fitting"]:
                missing.append(f"fitting.{k}")
    if "n_modests" not in config:
        missing.append("n_modests")
    if missing:
        raise ValueError("Missing required config keys: " + ", ".join(missing))


def build_fitter(config: dict):
    return partial(fit5d, **{k: config["fitting"][k] for k in FIT_KEYS})
