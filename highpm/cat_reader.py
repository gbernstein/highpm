"""
This module provides functions to read and clean catalog data from FITS files.
"""

import fitsio
import numpy as np


def read_cat_header(filename):
    """Reads the header data from a catalog file.

    Parameters
    ----------
    filename : str
        The path to the catalog file to be read.
    Returns
    -------
    header : dict
        The metadata header of the catalog file as a dictionary.
    """

    header = fitsio.read_header(filename)
    return header


def read_cat_data(filename):
    """Reads catalog data from a file using fitsio.

    Parameters
    ----------
    filename : str
        Path to the file containing the catalog data.
    Returns
    -------
    cat : np.ndarray
        The catalog data read from the specified file.
    """

    cat = fitsio.read(filename)
    return cat


def clean_cat(catname):
    """Cleans a catalog by applying quality cuts on specific columns.

    Filters the input catalog based on the following criteria:
    - 'FLAGS' column values less than 4.
    - 'IMAFLAGS_ISO' column values equal to 0.
    - Absolute value of 'SPREAD_MODEL' less than three times 'SPREADERR_MODEL',
        if these columns exist. If not, skips this cut and prints a warning.
    Parameters
    ----------
    catname : numpy.ndarray
            Input catalog containing at least the columns 'FLAGS' and
            'IMAFLAGS_ISO'. For additional filtering, should also contain
            'SPREAD_MODEL' and 'SPREADERR_MODEL'.
    Returns
    -------
    cleancat : same type as `catname`
            The filtered catalog after applying the quality cuts.
    Raises
    ------
    KeyError
            If 'SPREAD_MODEL' or 'SPREADERR_MODEL' columns are missing, the
            function skips the related cut and prints a warning instead.
    """

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
