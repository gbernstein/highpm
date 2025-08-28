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

    header = fitsio.read_header(filename, ext=1)
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

    cat = fitsio.read(filename, ext=1)
    return cat


def wavg_extended_class_y6a2(spread_model, spread_model_error):
    """DES DR2 extended classifier for stars and galaxies using WAVG quantities.

        0 : High-confidence stars
        1 : Likely stars
        2 : Likely galaxies
        3 : High-confidence galaxies
       -9 : Failures

    Parameters
    ----------
    spread_model      : SExtractor spread model value in a band
    spread_model_error: SExtractor spread model error in a band

    Returns
    -------
    extended_class    : Extended classifier output
    """
    extend_val = ((spread_model + 3.0 * spread_model_error) > 0.005) * 1
    extend_val += ((spread_model + 1.0 * spread_model_error) > 0.003) * 1
    extend_val += ((spread_model - 0.5 * spread_model_error) > 0.001) * 1

    extend_val[spread_model == -1] = -9
    extend_val[(spread_model == 0) & (spread_model_error == 0)] = -9
    extend_val[(spread_model == 1) & (spread_model_error == 1)] = -9
    return extend_val


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

    cleanmask = np.logical_and(catname["FLAGS"] < 4, catname["IMAFLAGS_ISO"] == 0)

    try:
        # cleanmask &= np.abs(catname["SPREAD_MODEL"]) < 3 * catname["SPREADERR_MODEL"]

        extval = wavg_extended_class_y6a2(
            catname["SPREAD_MODEL"], catname["SPREADERR_MODEL"]
        )
        cleanmask &= extval != 3
    except KeyError:
        print("No Spead Model cleaning...")
    return cleanmask
