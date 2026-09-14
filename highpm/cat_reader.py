"""
This module provides functions to read and clean catalog data from FITS files.
"""

import os

import fitsio
import numpy as np
from astropy.time import Time

from highpm.detection_packaging import load_exposure_radec
from highpm.friends_of_friends import query_pairs_groups
from highpm.gnomonic_converter import projectGnomonic
from highpm.pmfit import error_size
from highpm.utils import arborist


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


def detprob_logit(m, params):
    """
    logit function
    params = (m50, k, c)

    Arguments:
    - m: magnitude argument
    - params: tuple with m50, k and c
    """
    m50, k, c = params
    logit = c / (1 + np.exp(k * (m - m50)))
    return logit


def flux_to_mag(expnum, ccdnum, flux, zeropoint_cat):

    zp_keys = np.core.records.fromarrays(
        [zeropoint_cat["EXPNUM"], zeropoint_cat["CCDNUM"]],
        names=["expnum", "ccdnum"],
    )

    flux_keys = np.core.records.fromarrays(
        [expnum, ccdnum],
        names=["expnum", "ccdnum"],
    )

    sort_idx = np.argsort(zp_keys)
    indices = np.searchsorted(zp_keys[sort_idx], flux_keys)

    zp_mag = zeropoint_cat["MAG_ZERO"][indices]

    mag = -2.5 * np.log10(flux) + zp_mag

    return mag


def completeness_limit(
    expnum_arr, mag_arr, completeness_cat, completeness_threshold=0.05
):
    unique_expnum = np.unique(expnum_arr)

    cat_mask = np.isin(completeness_cat["expnum"], unique_expnum)

    completeness_subcat = completeness_cat[cat_mask]

    completeness_mask = np.ones_like(expnum_arr, dtype=bool)

    for expnum in unique_expnum:
        try:
            expnum_mask = expnum_arr == expnum
            params = (
                completeness_subcat["m50"][completeness_subcat["expnum"] == expnum][0],
                completeness_subcat["k"][completeness_subcat["expnum"] == expnum][0],
                completeness_subcat["c"][completeness_subcat["expnum"] == expnum][0],
            )
            mag = mag_arr[expnum_arr == expnum]
            prob = detprob_logit(mag, params)
            completeness_mask[expnum_mask] = prob > completeness_threshold
        except IndexError:
            print(f"Warning: expnum {expnum} not found in completeness catalog.")
            continue

    # Include all invalid magnitudes
    completeness_mask[mag_arr < 0] = True
    completeness_mask[mag_arr > 50] = True

    return completeness_mask


def clean_cat(catname, config=None, completeness_cat=None, completeness_threshold=0.05):
    """Cleans a catalog by applying quality cuts on specific columns.

    Filters the input catalog based on the following criteria:
    - 'FLAGS' column values less than 4.
    - 'IMAFLAGS_ISO' column values equal to 0.
    - Absolute value of 'SPREAD_MODEL' less than three times 'SPREADERR_MODEL',
        if these columns exist. If not, skips this cut and prints a warning.
    - 'TRAP_FLAG' column is False, if the column exists. If not, skips this
        cut and prints a warning.
    - If `config['max_exposures_per_month']` is set, caps exposures per
        colocated-pointing cluster per calendar month (see
        `thin_exposures_by_pointing`).
    Parameters
    ----------
    catname : numpy.ndarray
            Input catalog containing at least the columns 'FLAGS' and
            'IMAFLAGS_ISO'. For additional filtering, should also contain
            'SPREAD_MODEL' and 'SPREADERR_MODEL'.
    config : dict, optional
            Pipeline config. If it sets 'max_exposures_per_month', also
            requires 'ra0'/'dec0' and either 'exposures_file' or the
            $DES_EXPOSURES env var (a pixmappy exposures table).
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
        # ponytail: extval<=1 (high-confidence + likely star, -9 failures
        # still pass through same as before) instead of just !=3 -- crowded
        # Sculptor-core groups were found to be dominated by extval==2
        # ("likely galaxy") detections (e.g. one 239-detection group was only
        # 34% extval==0), inflating friends-of-friends large-group counts
        # with astrometrically noisy extended/blended sources this pipeline
        # isn't trying to measure proper motions for anyway.
        cleanmask &= extval <= 1

        # print(np.sum(cleanmask), "detections after Spread Model cleaning.")

        # mags = flux_to_mag(
        #     catname["EXPNUM"], catname["CCDNUM"], catname["FLUX_AUTO"], zp_cat
        # )

        # cleanmask &= completeness_limit(
        #     catname["EXPNUM"],
        #     mags,
        #     completeness_cat,
        #     completeness_threshold,
        # )

    except (KeyError, ValueError):
        # KeyError: astropy Table/dict-like. ValueError: plain numpy structured
        # array (what fitsio.read actually returns) raises this instead on a
        # missing field.
        print("No Spread Model cleaning...")

    try:
        cleanmask &= ~catname["TRAP_FLAG"]
    except (KeyError, ValueError):
        print("No Trap Flag cleaning...")

    max_exp_per_month = None if config is None else config.get("max_exposures_per_month")
    if max_exp_per_month is not None:
        exposures_file = config.get("exposures_file") or os.environ["DES_EXPOSURES"]
        expnum, ra, dec = load_exposure_radec(exposures_file)
        zeros = np.zeros_like(ra)
        xi, eta, *_ = projectGnomonic(ra, dec, zeros, zeros, config["ra0"], config["dec0"])
        expnum = expnum.tolist()
        pointing_xi = dict(zip(expnum, xi.tolist()))
        pointing_eta = dict(zip(expnum, eta.tolist()))
        cleanmask &= thin_exposures_by_pointing(
            catname,
            max_exp_per_month,
            pointing_xi,
            pointing_eta,
            config.get("pointing_linklength_arcmin", 5.0),
        )

    return cleanmask


def thin_exposures_by_pointing(
    cat,
    max_exposures_per_month,
    pointing_xi,
    pointing_eta,
    linklength_arcmin=5.0,
    err_band_arcsec=(0.005, 0.05),
):
    """Cap exposures per colocated-pointing cluster per calendar month,
    keeping the astrometrically best ones.

    Dithered campaigns (many exposures re-pointed by a few arcsec/arcmin
    within a few nights) add little proper-motion baseline but combinatorially
    inflate friends-of-friends group sizes. Exposures are first split by
    calendar month; within each month, exposures are clustered by pointing via
    friends-of-friends on each exposure's true pointing center
    (`pointing_xi`/`pointing_eta`, from the pixmappy DELVE exposures table,
    not the exposure's own detections) with `linklength_arcmin`. Any resulting
    month-cluster with more than `max_exposures_per_month` exposures is
    thinned down to that many, keeping the exposures with the lowest median
    error_size() among their own detections that fall within
    `err_band_arcsec` (this pipeline's typical-precision band, not
    implausibly tight or noisy outliers). Month-clusters at or under the cap
    are left untouched.

    An exposure with no detections inside err_band_arcsec can't be ranked by
    this criterion and is treated as worst (its rank key is +inf), so it is
    dropped first if its month-cluster ends up over the cap.

    Parameters
    ----------
    cat : np.ndarray
        Catalog with 'EXPNUM', 'MJD', 'BEST_RA_ERR', 'BEST_DEC_ERR',
        'BEST_RA_DEC_CORR'.
    max_exposures_per_month : int
        Maximum exposures to keep per pointing cluster per calendar month.
    pointing_xi, pointing_eta : dict[int, float]
        Exposure pointing center, in the same gnomonic (XI, ETA) frame as
        `cat`, keyed by EXPNUM. Must cover every EXPNUM in `cat`. See
        `highpm.detection_packaging.load_exposure_radec` +
        `highpm.gnomonic_converter.projectGnomonic` to build these from a
        pixmappy exposures table.
    linklength_arcmin : float, optional
        Friends-of-friends linking length, in arcmin, used to cluster
        exposure pointings. Default 5.0.
    err_band_arcsec : tuple of (float, float), optional
        (low, high) error_size band in arcsec used for the ranking median.
        Default (0.005, 0.05) = 5-50 mas.

    Returns
    -------
    keepmask : np.ndarray of bool, shape (len(cat),)
        True for every detection whose exposure survives thinning.
    """
    sigma = error_size(cat)
    expnum = cat["EXPNUM"]

    unique_expnum, first_idx = np.unique(expnum, return_index=True)
    unique_mjd = cat["MJD"][first_idx]

    xi_centroid = np.array([pointing_xi[exp] for exp in unique_expnum])
    eta_centroid = np.array([pointing_eta[exp] for exp in unique_expnum])

    dt = Time(unique_mjd, format="mjd").datetime
    year_month = [(d.year, d.month) for d in dt]

    months = {}
    for i, key in enumerate(year_month):
        months.setdefault(key, []).append(i)

    lo, hi = err_band_arcsec
    rank_key = np.full(len(unique_expnum), np.inf, dtype=np.float64)
    for i, exp in enumerate(unique_expnum):
        exp_sigma = sigma[expnum == exp]
        band_sigma = exp_sigma[(exp_sigma > lo) & (exp_sigma < hi)]
        if len(band_sigma) > 0:
            rank_key[i] = np.median(band_sigma)

    kept_expnums = []
    for month_idxs in months.values():
        month_idxs = np.asarray(month_idxs)
        pointing_tree = arborist(xi_centroid[month_idxs], eta_centroid[month_idxs])
        pointing_groups = query_pairs_groups(pointing_tree, linklength_arcmin / 60.0)
        for local_idxs in pointing_groups:
            idxs = month_idxs[local_idxs]
            if len(idxs) <= max_exposures_per_month:
                kept_expnums.append(unique_expnum[idxs])
            else:
                order = np.argsort(rank_key[idxs], kind="stable")
                keep_idxs = idxs[order[:max_exposures_per_month]]
                kept_expnums.append(unique_expnum[keep_idxs])

    kept_expnums = np.concatenate(kept_expnums)

    return np.isin(expnum, kept_expnums)


if __name__ == "__main__":
    # Two pointing clusters (far apart), 8 exposures each within one month.
    # Cap=5 per cluster-month should keep exactly 5 from each, favoring the
    # exposures with the smallest error_size.
    n_exp_per_cluster = 8
    rows = []
    pointing_xi, pointing_eta = {}, {}
    for cluster_center in [(0.0, 0.0), (10.0, 10.0)]:
        for exp in range(n_exp_per_cluster):
            expnum = cluster_center[0] * 1000 + exp  # unique per cluster
            ra_err = 0.01 + 0.001 * exp  # increasing (worst last)
            pointing_xi[expnum] = cluster_center[0]
            pointing_eta[expnum] = cluster_center[1]
            for _ in range(3):  # detections per exposure
                rows.append((expnum, 59001.0 + exp, ra_err, ra_err, 0.0))  # same month (2020-06)

    dtype = [
        ("EXPNUM", "f8"),
        ("MJD", "f8"),
        ("BEST_RA_ERR", "f8"),
        ("BEST_DEC_ERR", "f8"),
        ("BEST_RA_DEC_CORR", "f8"),
    ]
    cat = np.array(rows, dtype=dtype)

    keepmask = thin_exposures_by_pointing(
        cat, 5, pointing_xi, pointing_eta, linklength_arcmin=5.0
    )
    kept_expnums = np.unique(cat["EXPNUM"][keepmask])
    assert len(kept_expnums) == 10, kept_expnums  # 5 kept per cluster x 2 clusters
    for cluster_center in [(0.0, 0.0), (10.0, 10.0)]:
        cluster_kept = kept_expnums[
            (kept_expnums >= cluster_center[0] * 1000)
            & (kept_expnums < cluster_center[0] * 1000 + n_exp_per_cluster)
        ]
        assert sorted(cluster_kept) == [
            cluster_center[0] * 1000 + i for i in range(5)
        ], cluster_kept  # lowest-error (lowest exp index) 5 survive

    # A colocated cluster split across two calendar months should not be
    # thinned if neither month alone exceeds the cap.
    rows = []
    pointing_xi2, pointing_eta2 = {}, {}
    for exp, mjd in enumerate([59000.0, 59001.0, 59002.0, 59032.0, 59033.0]):
        pointing_xi2[exp], pointing_eta2[exp] = 0.0, 0.0
        for _ in range(3):
            rows.append((exp, mjd, 0.01, 0.01, 0.0))
    cat2 = np.array(rows, dtype=dtype)
    keepmask2 = thin_exposures_by_pointing(
        cat2, 3, pointing_xi2, pointing_eta2, linklength_arcmin=5.0
    )
    assert np.all(keepmask2)  # 3 in Jan, 2 in Feb 2020 (mjd 59032-33) -- both under cap

    print("thin_exposures_by_pointing self-check OK")
