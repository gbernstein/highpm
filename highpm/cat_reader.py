import numpy as np
from astropy.table import QTable


def read_cat_header(filename):
    # Takes catalog produced by getFinalcutTile.py and returns the
    # corresponding header
    tbl = QTable.read(filename)
    header = tbl.meta
    return header


def read_cat_data(filename):
    # Takes catalog produced by getFinalcutTile.py and returns the
    # corresponding data table
    cat = QTable.read(filename)
    return cat


def clean_cat(catname):
    # Removes detections with "FLAGS"!=0 and "IMAFLAGS!=0"
    cleancat = catname[
        np.logical_and(catname["FLAGS"] < 4, catname["IMAFLAGS_ISO"] == 0)
    ]
    try:
        cleancat = cleancat[
            np.abs(cleancat["SPREAD_MODEL"]) < 3 * cleancat["SPREADERR_MODEL"]
        ]
    except KeyError:
        cleancat = catname
        print("No Spead Model cleaning...")
    return cleancat
