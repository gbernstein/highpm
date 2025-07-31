"""
This module provides functions for detecting and fitting proper motion from
single epoch detections from ground based surveys.
"""

import sys
from functools import partial
from multiprocessing import Pool

import fitsio
import numpy as np
import scipy.spatial as spspace
import tqdm
import yaml
from cat_reader import clean_cat, read_cat_data, read_cat_header
from fits_writer import output_fits
from friends_of_friends import find_friend, friends_of_friends
from pmfit import err2cov, fit5d
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cluster import DBSCAN


def arborist(xi, eta):
    """Builds a KDTree from the provided xi and eta coordinate arrays.

    Parameters
    ----------
    xi : array_like
        The x-coordinates of the points to be included in the KDTree.
    eta : array_like
        The y-coordinates of the points to be included in the KDTree.

    Returns
    -------
    tree : spspace.KDTree
        A KDTree object constructed from the input coordinates, which can be
        used for efficient spatial queries.

    Notes
    -----
    The input arrays `xi` and `eta` must have the same length. The function
    stacks them as columns to form a 2D array of shape (n_points, 2).
    """
    tree = spspace.KDTree(np.array([xi, eta]).T)
    return tree


def detections_for_removal(pm_arr, config):
    """Identifies and returns detections for removal based on proper motion
    limits. This function processes an array of proper motion fit results and
    their covariances, filtering out detections whose proper motion and
    associated errors fall below specified thresholds.

    Parameters
    ----------
    pm_arr : np.ndarray
        Array of shape (N, ...) where each element contains fit results and
        covariance matrices for proper motion measurements.
    config : dict
        Configuration dictionary containing the following keys:
        - 'pm_lim': float
            The upper limit for the magnitude of proper motion (in mas/yr)
            below which detections are considered for removal.
        - 'pm_err_lim': float
            The upper limit for the error in proper motion (in mas/yr) for both
            RA and Dec components.
    Returns
    -------
    removals : np.ndarray
        Array of detections that meet the criteria for removal. If no
        detections meet the criteria, an empty array is returned.
    Notes
    -----
    - The function multiplies proper motion values and errors by 1000 to
      convert from arcsec/yr to mas/yr.
    - If no detections meet the criteria, a message is printed and an empty
      array is returned.
    """

    if "pm_lim" not in config or "pm_err_lim" not in config:
        print("No proper motion or proper motion error limit set, skipping removal.")
        return np.array([])

    p_fits = np.vstack(pm_arr[:, 0])
    cov = np.array([np.linalg.inv(i) for i in pm_arr[:, 1]])

    pmra = 1000 * p_fits[:, 2]
    pmra_err = 1000 * cov[:, 2, 2]
    pmdec = 1000 * p_fits[:, 3]
    pmdec_err = 1000 * cov[:, 3, 3]
    try:
        removals = np.concatenate(
            [
                pm_arr[i][6]
                for i in range(len(pm_arr))
                if np.logical_and(
                    abs(pmra_err[i]) < config["pm_err_lim"],
                    abs(pmdec_err[i]) < config["pm_err_lim"],
                )
                and np.hypot(pmra[i], pmdec[i]) < config["pm_lim"]
            ]
        )
    except ValueError:  # In case no removals are found
        print("No removals found.")
        removals = np.array([])

    return removals


def filter_list(L):
    """Filters a list of lists by removing any list that is a proper subset of
    another.

    Parameters
    ----------
    L : list of list
        A list containing lists of hashable elements.

    Returns
    -------
    us : list of list
        A list of lists where each list is not a proper subset of any other
        list in the input.

    Examples
    --------
    >>> filter_list([[1, 2], [1], [2], [1, 2, 3]])
    [[1, 2, 3], [1, 2]]
    """

    sets = {frozenset(e) for e in L}
    us = []
    for e in sets:
        if any(e < s for s in sets):
            continue
        else:
            us.append(list(e))
    return us


def new_posvel(pairs, x, y, t, cov_xy, config):
    """Compute average positions, proper motions, and their covariances for
     pairs of detections. Given pairs of detection indices, positions, times,
     and covariance matrices, this function calculates the average position and
     proper motion for each pair, along with the associated covariance
     estimates. It also returns the time difference for each pair and a boolean
     mask indicating valid pairs.

     Parameters
     ----------
     pairs : ndarray of shape (N, 2)
         Array of index pairs, where each row contains two indices (i, j)
         referencing detections.
     x : ndarray of shape (M,)
         Array of x positions for each detection.
     y : ndarray of shape (M,)
         Array of y positions for each detection.
     t : ndarray of shape (M,)
         Array of observation times.
     cov_xy : ndarray of shape (M, 3)
         Covariance matrix for each detection, where the first column is the
         variance in x and the second column is the variance in y.
     config : dict
         Configuration dictionary containing the following keys:
         - 'mjd_ref': float, optional
             Reference time (in MJD) for normalization. Default is 57388.0.
    Returns
     -------
     posvel : ndarray of shape (K, 4)
         Array containing average x, average y, proper motion in x, and proper
         motion in y for each valid pair.
     cov : ndarray of shape (K, 4)
         Covariance estimates for average x, average y, proper motion in x, and
         proper motion in y for each valid pair.
     dt : ndarray of shape (K,)
         Absolute time differences between the pairs.
     good_pairs : ndarray of bool, shape (N,)
         Boolean mask indicating which pairs have non-zero time difference and
         were used in the calculations.
     Notes
     -----
     Pairs with zero time difference are excluded from the output.
    """
    if "mjd_ref" not in config:
        config["mjd_ref"] = 57388.0
        print("No MJD reference set, using default 57388.0")

    i, j = pairs.T

    dt = t[j] - t[i]
    dx = x[j] - x[i]
    dy = y[j] - y[i]

    good_pairs = dt != 0
    i = i[good_pairs]
    j = j[good_pairs]
    dx = dx[good_pairs]
    dy = dy[good_pairs]
    dt = dt[good_pairs]

    vx = dx / dt
    vy = dy / dt

    cov_x = (cov_xy[i, 0] + cov_xy[j, 0]) / 4
    cov_y = (cov_xy[i, 1] + cov_xy[j, 1]) / 4
    cov_vx = (cov_xy[i, 0] + cov_xy[j, 0]) / dt**2
    cov_vy = (cov_xy[i, 1] + cov_xy[j, 1]) / dt**2

    cov = np.array([cov_x, cov_y, cov_vx, cov_vy]).T

    avg_x = (x[i] + x[j]) / 2 - vx * (
        (t[i] + t[j]) / 2.0 - config["mjd_ref"] / 365.2425
    )
    avg_y = (y[i] + y[j]) / 2 - vy * (
        (t[i] + t[j]) / 2.0 - config["mjd_ref"] / 365.2425
    )

    posvel = np.array([avg_x, avg_y, vx, vy]).T

    return posvel, cov, np.abs(dt), good_pairs


# =============================================================================
# New Slow / Modest Mover algorithm
# =============================================================================


def new_modest_mover(sample, cat, config, mode="modest"):
    """Identifies clusters of moving objects in a catalog based on positional
     and temporal data using DBSCAN clustering.

     Parameters
     ----------
     sample : list
         Indices or boolean mask selecting the subset of detections to analyze
         from the catalog.
     cat : np.ndarray
         A structured array with the fields "XI", "ETA", and "MJD" representing
         the x and y positions (in degrees) and the Modified Julian Date of each
         detection, respectively.
     config : dict
         Configuration dictionary containing the following keys:
         - 'eps': float
             The maximum distance between two samples for one to be considered
             as in the neighborhood of the other (in 4D position-velocity space).
         - 'min_pairs': int
             The minimum number of samples in a neighborhood for a point to be
             considered as a core point.
         - 'n_detections': int
             The minimum number of detections required to form a valid cluster.
         - 'mjd_ref': float, optional
             Reference Modified Julian Date for time normalization (default is
             57388.0).
         - 'min_dt': float, optional
             Minimum time difference (in years) required between detection pairs
             to be considered for clustering (default is 0.8).
     mode : str, optional
         The mode of operation, either "modest" or "fast". Determines the
         clustering algorithm and parameters used. Default is "modest".
    Returns
     -------
     obj_list : list of list of int or None
         List of clusters, where each cluster is a list of detection indices
         corresponding to a moving object. Returns None if no valid clusters are
         found.
     Notes
     -----
     - Uses DBSCAN clustering on a custom distance matrix derived from position
       and velocity estimates.
     - Requires external functions: `err2cov`, `new_posvel`, and `filter_list`.
     - Assumes the catalog fields are in degrees and converts them to
       arcseconds.
    """

    config_reqs = ["mjd_ref", "min_dt", "eps", "min_pairs", "n_detections"]

    if np.any([key not in config and key not in config[mode] for key in config_reqs]):
        raise ValueError(f"Missing required config keys: {config_reqs}")

    sample = np.array(sample)

    x = np.array(cat["XI"][sample]) * 3600.0
    y = np.array(cat["ETA"][sample]) * 3600.0
    t = np.array(cat["MJD"][sample]) / 365.2425

    cov = err2cov(cat[sample], additional_error=False)

    pairs = np.array(np.triu_indices(len(sample), k=1)).T

    posvel, cov, dt, good_pairs = new_posvel(pairs, x, y, t, cov, config)

    if np.all(dt < config[mode]["min_dt"]):
        return None

    pairs = pairs[good_pairs]
    pairs = pairs[dt >= config[mode]["min_dt"]]
    posvel = posvel[dt >= config[mode]["min_dt"]]
    cov = cov[dt >= config[mode]["min_dt"]]
    dt = dt[dt >= config[mode]["min_dt"]]

    X = posvel
    Sigma = np.sqrt(cov)

    D = np.zeros((len(dt), len(dt)))
    for k in range(len(dt)):
        xk = X[k]
        sk2 = Sigma[k] ** 2
        dk = X - xk
        var_sum = Sigma**2 + sk2
        D[k] = np.sum(dk**2 / var_sum, axis=1)

    D = np.sqrt(D)

    indices = np.column_stack((sample[pairs[:, 0]], sample[pairs[:, 1]]))

    clustering = DBSCAN(
        eps=config[mode]["eps"],
        min_samples=config[mode]["min_pairs"],
        n_jobs=1,
        metric="precomputed",
    ).fit(D)

    obj_list = []

    for cluster_id in np.unique(clustering.labels_):
        if cluster_id == -1:
            continue  # Skip noise points
        cluster_mask = clustering.labels_ == cluster_id
        cluster_indices = np.unique(indices[cluster_mask])

        if len(cluster_indices) >= config["n_detections"]:
            obj_list.append(cluster_indices.tolist())

    if len(obj_list) == 0:
        return None

    obj_list = filter_list(obj_list)
    return obj_list


def new_modest_fitter(cat, fitter, config):
    """Fits proper motion models to groups of detections in a catalog using a
     friends-of-friends algorithm and parallel processing.

     Parameters
     ----------
     cat : np.ndarray
         A structured array with the fields "XI", "ETA", and "MJD" representing
         the x and y positions (in degrees) and the Modified Julian Date of each
         detection, respectively.
     fitter : callable
         Fitting function or object to be used in the multi_fit5d step.
     config : dict
         Configuration dictionary containing the following keys:
         - 'linklength': float, optional
             Linking length (in degrees) for the friends-of-friends algorithm.
             Default is 1.0 / 3600.0.
         - 'cores': int, optional
             Number of CPU cores to use for parallel processing. Default is 1.
         - 'mjd_ref': float, optional
             Reference Modified Julian Date for time normalization (default is
             57388.0).
         - 'min_dt': float, optional
             Minimum time difference (in years) required between detection pairs
             to be considered for clustering (default is 0.8).
         - 'n_detections': int, optional
             Minimum number of detections required to form a valid group.
    Returns
     -------
     modest_pm_arr : np.ndarray
         Array of fitted proper motion parameters for the detected groups.
     Notes
     -----
     Requires the following functions to be defined elsewhere: `arborist`,
     `find_friend`, `friends_of_friends`, `multithreader`, `new_modest_mover`,
     and `multi_fit5d`. The variable `n_detections` must also be defined in the
     scope.
    """

    config_reqs = ["linklength", "cores", "mjd_ref", "min_dt", "eps", "n_detections"]
    if np.any(
        [key not in config and key not in config["modest"] for key in config_reqs]
    ):
        raise ValueError(f"Missing required config keys: {config_reqs}")

    linklength = config["modest"]["linklength"] / 3600.0

    modest_tree = arborist(cat["XI"], cat["ETA"])
    modest_friends = find_friend(modest_tree, linklength, cores=config["cores"])
    modest_groups = friends_of_friends(modest_friends)

    modest_groups = [i for i in modest_groups if len(i) >= config["n_detections"]]

    partial_new_modest_mover = partial(new_modest_mover, mode="modest")

    modest_pm_ls = multithreader(partial_new_modest_mover, modest_groups, cat, config)

    modest_pm_ls = [i for i in modest_pm_ls if i is not None]
    modest_pm_obj = []
    for i in modest_pm_ls:
        for j in i:
            modest_pm_obj += [j]

    modest_pm_obj = [i for i in modest_pm_obj if len(i) >= config["n_detections"]]

    modest_pm_arr = multi_fit5d(fitter, modest_pm_obj, cat, config)

    return modest_pm_arr


# =============================================================================
# Fast mover algorithm
# =============================================================================


def fast_movers(cat, fitter, config):
    """Identify and fit fast-moving objects from a catalog of detections. This
    function processes a catalog of astronomical detections to identify
    candidate fast-moving stars by clustering detection pairs in
    position-proper motion space and fitting their motion parameters.

    Parameters
    ----------
    cat : np.ndarray
        Catalog of detections containing at least the columns "XI", "ETA", and
        "MJD".
    fitter : callable
        Function used to fit the motion parameters of candidate objects.
    config : dict
        Configuration dictionary containing the following keys:
        - 'n_detections': int
            Minimum number of detections required to form a valid candidate
            object.
        - 'mjd_ref': float, optional
            Reference Modified Julian Date for time normalization (default is
            57388.0).
        - 'cores': int, optional
            Number of CPU cores to use for parallel processing. Default is 1.
        - 'chunksize': int, optional
            Size of chunks for parallel processing. Default is 1000.
        - 'fast': dict
            Configuration parameters for fast mover detection, including:
            - 'pairlength': float, optional
                Maximum separation (in arcseconds) for initial detection pairing.
                Default is 20.0.
            - 'linklength': float, optional
                Maximum distance in 4D position-velocity space for clustering.
                Default is 1.0.
            - 'min_pairs': int, optional
                Minimum number of detection pairs required for a candidate object.
                Default is 4.
            - 'min_sep': float, optional
                Minimum proper motion (in arcseconds/year) for candidate pair.
                Default is 1.0.
            - 'max_sep': float, optional
                Maximum proper motion (in arcseconds/year) for candidate pair.
                Default is 20.0.
            - 'eps': float, optional
                DBSCAN clustering threshold in 4D space. Default is 3.0.
                pairlength : float, optional
            - 'min_dt': float, optional
                Minimum time difference (in years) between detection pairs.
                Default is 0.8.
    Returns
    -------
    fast_pm_arr : list
        List of fitted parameter arrays for each identified fast-moving object.
    Notes
    -----
    - Requires external functions: `err2cov`, `arborist`, `new_posvel`,
      `spspace.KDTree`, `DBSCAN`, `multithreader`, `new_modest_mover`, and
      `multi_fit5d`.
    - The catalog must provide columns for positions ("XI", "ETA") and times
      ("MJD").
    - The function uses DBSCAN clustering to group detection pairs in
      position-proper motion space and fits motion models to each candidate
      object.
    """
    config_reqs = [
        "pairlength",
        "linklength",
        "cores",
        "min_pairs",
        "min_sep",
        "max_sep",
        "eps",
        "mjd_ref",
        "min_dt",
        "n_detections",
    ]
    if np.any([key not in config and key not in config["fast"] for key in config_reqs]):
        raise ValueError(f"Missing required config keys: {config_reqs}")

    x = np.array(cat["XI"]) * 3600.0
    y = np.array(cat["ETA"]) * 3600.0
    t = np.array(cat["MJD"]) / 365.2425

    cov_xy = err2cov(cat, additional_error=False)

    fast_tree = arborist(x, y)
    fast_pairs = fast_tree.query_pairs(r=config["fast"]["pairlength"])
    fast_pairs = np.array(list(fast_pairs))

    fast_posvel, cov, _, good_pairs = new_posvel(fast_pairs, x, y, t, cov_xy, config)

    fast_pairs = fast_pairs[good_pairs]

    pm_keep = np.logical_and(
        np.hypot(fast_posvel[:, 2], fast_posvel[:, 3]) > config["fast"]["min_sep"],
        np.hypot(fast_posvel[:, 2], fast_posvel[:, 3]) < config["fast"]["max_sep"],
    )
    fast_pairs = fast_pairs[pm_keep]
    cov = cov[pm_keep]
    fast_posvel = fast_posvel[pm_keep]

    fast_4dtree = spspace.KDTree(fast_posvel)

    fast_4dpairs = np.array(
        list(fast_4dtree.query_pairs(r=config["fast"]["linklength"]))
    )

    diff = fast_posvel[fast_4dpairs[:, 1]] - fast_posvel[fast_4dpairs[:, 0]]
    invcov = 1.0 / (cov[fast_4dpairs[:, 0]] + cov[fast_4dpairs[:, 1]])

    dist2 = np.sum(diff * invcov * diff, axis=1)
    dist = np.sqrt(dist2)

    mask = dist < config["fast"]["eps"]
    i_idx, j_idx = fast_4dpairs[mask].T
    D = dist[mask]

    rows = np.concatenate([i_idx, j_idx])
    cols = np.concatenate([j_idx, i_idx])
    data = np.concatenate([D, D])

    D_sparse = coo_matrix(
        (data, (rows, cols)), shape=(len(fast_posvel), len(fast_posvel))
    )

    D_sparse = csr_matrix(D_sparse)

    # D_sparse = sklearn.neighbors.sort_graph_by_row_values(
    #     D_sparse, warn_when_not_sorted=False
    # )

    clustering = DBSCAN(
        eps=config["fast"]["eps"],
        min_samples=config["fast"]["min_pairs"],
        n_jobs=config["cores"],
        metric="precomputed",
    ).fit(D_sparse)

    obj_list = []
    indices = np.arange(len(fast_posvel))

    for cluster_id in tqdm.tqdm(
        np.unique(clustering.labels_), total=len(np.unique(clustering.labels_))
    ):
        if cluster_id == -1:
            continue  # Skip noise points
        cluster_mask = clustering.labels_ == cluster_id
        cluster_indices = np.unique(indices[cluster_mask])

        if len(cluster_indices) >= config["n_detections"]:
            obj_list.append(cluster_indices.tolist())

    fast_objects = obj_list

    hipm_object_sets = [i for i in fast_objects if len(i) > config["fast"]["min_pairs"]]
    fast_obj = []
    for i in hipm_object_sets:
        temp_object = []
        for j in i:
            detection_pair = [int(fast_pairs[j, 0]), int(fast_pairs[j, 1])]
            temp_object += detection_pair
        fast_obj += [list(np.unique(temp_object))]
    fast_candidates = [i for i in fast_obj if len(i) > config["n_detections"]]

    partial_new_modest_mover = partial(new_modest_mover, mode="fast")

    fast_pm_ls = multithreader(partial_new_modest_mover, fast_candidates, cat, config)

    fast_pm_ls = [i for i in fast_pm_ls if i is not None]
    fast_pm_obj = []
    for i in fast_pm_ls:
        for j in i:
            fast_pm_obj += [j]

    fast_pm_obj = [i for i in fast_pm_obj if len(i) >= config["n_detections"]]

    fast_pm_arr = multi_fit5d(fitter, fast_pm_obj, cat, config)
    return fast_pm_arr


# =============================================================================


def fast_checker(idx, cat, fast_cat):
    temp_idx_circ = (cat["XI"] - fast_cat["xi"][idx]) ** 2 + (
        cat["ETA"] - fast_cat["eta"][idx]
    ) ** 2 < ((3 * fast_cat["pm"][idx] + 1000) / 3600000) ** 2
    temp_cat_circ = cat[temp_idx_circ]

    slope = fast_cat["pmdec"][idx] / fast_cat["pmra"][idx]
    temp_idx = (
        np.abs(
            slope * (temp_cat_circ["XI"] - fast_cat["xi"][idx])
            + fast_cat["eta"][idx]
            - temp_cat_circ["ETA"]
        )
        / np.sqrt(1 + slope**2)
        < 2 / 3600.0
    )
    temp_cat = temp_cat_circ[temp_idx]

    pm = fast_cat["pm"][idx] / 1000

    res = 30
    if pm < 0.1:
        res = 300

    w = round(2 * 3 * res * (pm)) + 2 * res
    h = round(2 * res * (pm)) + 2 * res

    ra_0 = np.mean(temp_cat["XI"])
    dec_0 = np.mean(temp_cat["ETA"])

    obj_lol = []

    return obj_lol


# =============================================================================


# =============================================================================
# Multithreading Functions
# =============================================================================


def multithreader(func, lol, cat, config, fitting=False):
    """Executes a function in parallel across multiple processes using a pool.

    Parameters
    ----------
    func : callable
        The function to apply to each element of `lol`. Must accept an element
        from `lol` as its first argument and `cat` as a keyword argument.
    lol : list
        List of elements to process in parallel.
    cat : any
        Additional argument to pass to `func` as a keyword argument.
    config : dict
        Configuration dictionary containing parameters for parallel execution,
        such as the number of cores and chunk size.
    fitting : bool, optional
        If True, the function is assumed to be a fitting function that requires
        additional parameters. If False, it is assumed to be a general function
        that only requires `cat` as a keyword argument.
    Returns
    -------
    ls_out : list
        List of results returned by applying `func` to each element in `lol`.

    Notes
    -----
    Uses `multiprocessing.Pool` for parallel execution and `tqdm` for progress
    display. The function is partially applied with the `cat` argument.
    """
    config_reqs = ["cores", "chunksize"]
    if np.any([key not in config for key in config_reqs]):
        raise ValueError(f"Missing required config keys: {config_reqs}")

    ls_out = []
    if fitting:
        partial_func = partial(func, cat=cat)
    else:
        partial_func = partial(func, cat=cat, config=config)

    with Pool(processes=config["cores"]) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(partial_func, lol, chunksize=config["chunksize"]),
            total=len(lol),
        ):
            ls_out.append(_)
            pass
    pool.close()
    pool.join()
    return ls_out


def multi_fit5d(fitter, detections_groups, cat, config):
    """Applies a 5D fitter to groups of detections using multithreading.

    Parameters
    ----------
    fitter : callable
        A function or callable object that performs fitting on a group of
        detections.
    detections_groups : iterable
        An iterable of detection groups to be processed by the fitter.
    cat : object
        Catalog or additional data required by the fitter.
    config : dict
        Configuration dictionary containing fitting parameters, including:
        - 'cores': int
            Number of CPU cores to use for multithreading.
        - 'chunksize': int
            Number of detection groups to process per thread chunk.
        - 'fitting': dict
            Dictionary containing fitting parameters:
            - 'time_sep': float
                Time separation threshold for fitting.
            - 'chisqClip': float
                Chi-squared clipping threshold for fitting.
            - 'parallax_prior': float
                Parallax prior value for fitting.
            - 'color_prior': float
                Color prior value for fitting.
            - 'colorFrac': float
                Color fraction for fitting.
            - 'pm_prior': float
                Proper motion prior value for fitting.
            - 'additional_error': bool
                whether or not to add additional error to be added to the fitting
                process.
    Returns
    -------
    np.ndarray
        An array of fit results, with failed fits (None) removed. The array has
        dtype=object.
    """
    config_reqs = [
        "cores",
        "chunksize",
        "time_sep",
        "chisqClip",
        "parallax_prior",
        "color_prior",
        "colorFrac",
        "pm_prior",
        "additional_error",
    ]
    if np.any(
        [key not in config and key not in config["fitting"] for key in config_reqs]
    ):
        raise ValueError(f"Missing required config keys: {config_reqs}")

    partial_fitter = partial(
        fitter,
        time_sep=config["fitting"]["time_sep"],
        chisqClip=config["fitting"]["chisqClip"],
        parallax_prior=config["fitting"]["parallax_prior"],
        color_prior=config["fitting"]["color_prior"],
        colorFrac=config["fitting"]["colorFrac"],
        pm_prior=config["fitting"]["pm_prior"],
        additional_error=config["fitting"]["additional_error"],
    )

    # Multithreaded application of a 5D fitter to groups of detections
    pm_list = multithreader(
        partial_fitter, detections_groups, cat, config, fitting=True
    )

    # Remove fits that return None
    clean_pm_list = [i for i in pm_list if i is not None]

    # Convert list to array and discard list
    clean_pm_arr = np.array(clean_pm_list, dtype=object)
    clean_pm_list.clear()

    return clean_pm_arr


# =============================================================================


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

    cat_copy = cat.copy()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    fitting = config["fitting"]

    time_sep = fitting["time_sep"]
    chisqClip = fitting["chisqClip"]
    parallax_prior = fitting["parallax_prior"]
    color_prior = fitting["color_prior"]
    colorFrac = fitting["colorFrac"]
    pm_prior = fitting["pm_prior"]
    additional_error = fitting["additional_error"]

    part_fit5d = partial(
        fit5d,
        time_sep=time_sep,
        chisqClip=chisqClip,
        parallax_prior=parallax_prior,
        color_prior=color_prior,
        colorFrac=colorFrac,
        pm_prior=pm_prior,
        additional_error=additional_error,
    )

    for _ in range(3):
        modest_pm_arr = new_modest_fitter(
            cat,
            part_fit5d,
            config,
        )

        if len(modest_pm_arr) != 0:
            modest_tbl = output_fits(modest_pm_arr, catname, f"modest{_ + 1}")

            print(f"Writing {len(modest_tbl)} modest movers... (round {_ + 1})")

            modest_detections = detections_for_removal(modest_pm_arr, config)
            if len(modest_detections) != 0:
                cat = np.delete(cat, modest_detections, axis=0)
        else:
            print(f"No modest movers found in round {_ + 1}.")
            modest_tbl = None
            modest_detections = None

        fitsio.write(
            f"pre_fast{_ + 1}_" + catname[:-5] + "_header.fits",
            cat,
            clobber=True,
        )

        del modest_pm_arr
        del modest_tbl
        del modest_detections

    fast_pm_arr = fast_movers(cat, part_fit5d, config)

    if len(fast_pm_arr) != 0:
        fast_tbl = output_fits(fast_pm_arr, catname, "fast")

        print(f"Writing {len(fast_tbl)} fast movers...")

    del fast_pm_arr

    print("Done!")

    sys.exit(0)
