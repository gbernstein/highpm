#!/usr/bin/env python
# Establishes proper motions for stars in the DES footprint

from __future__ import print_function

# import argparse
# import os
import sys
from functools import partial
from multiprocessing import Pool

import astropy.units as u
import numpy as np
import scipy.spatial as spspace
import tqdm
from cat_reader import clean_cat, read_cat_data, read_cat_header
from fits_writer import output_fits
from friends_of_friends import find_friend, friends_of_friends
from pmfit import err2cov, fit5d
from scipy.sparse import coo_matrix
from sklearn.cluster import DBSCAN

n_detections = 5


def arborist(xi, eta):
    """
    Creates KD-Tree for possitions in tile coordinates.

    Parameters
    ---------
    xi : astropy.table.column.Column
        RA tile coordinates in pixels.
    eta : astropy.table.column.Column
        DEC tile coordinates in pixels.

    Returns
    -------
    scipy.spatial.kdtree.KDTree
        KD-Tree for all detections in a tile.

    """

    tree = spspace.KDTree(np.array([xi, eta]).transpose())
    return tree


def detections_for_removal(pm_arr, pm_lim, pm_err_lim):
    """
    Removes detections that have been included in a fit.

    Parameters
    ----------
    pm_arr :
    pm_lim :
    pm_err_lim :

    Returns
    -------
    removals : numpy.ndarray
    """

    p_fits = np.vstack(pm_arr[:, 0])
    cov = np.array([np.linalg.inv(i) for i in pm_arr[:, 1]])

    pmra = (p_fits[:, 2] * u.arcsec / u.year).to(u.mas / u.year)
    pmra_err = (cov[:, 2, 2] * u.arcsec / u.year).to(u.mas / u.year)
    pmdec = (p_fits[:, 3] * u.arcsec / u.year).to(u.mas / u.year)
    pmdec_err = (cov[:, 3, 3] * u.arcsec / u.year).to(u.mas / u.year)
    try:
        removals = np.concatenate(
            [
                pm_arr[i][6]
                for i in range(len(pm_arr))
                if np.logical_and(
                    abs(pmra_err[i]) < pm_err_lim * u.mas / u.year,
                    abs(pmdec_err[i]) < pm_err_lim * u.mas / u.year,
                )
                and np.hypot(pmra[i], pmdec[i]) < pm_lim * u.mas / u.year
            ]
        )
    except ValueError:  # In case no removals are found
        print("No removals found.")
        removals = np.array([])

    return removals


def filter_list(L):
    sets = {frozenset(e) for e in L}
    us = []
    for e in sets:
        if any(e < s for s in sets):
            continue
        else:
            us.append(list(e))
    return us


# =============================================================================
# New Slow / Modest Mover algorithm
# =============================================================================


def new_modest_mover(sample, cat, mjd_ref=57388.0, min_dt=0.8):

    sample = np.array(sample)

    x = cat["XI"][sample] * 3600.0
    y = cat["ETA"][sample] * 3600.0
    t = cat["MJD"][sample] / 365.2425

    x -= np.mean(x)
    y -= np.mean(y)

    cov = err2cov(cat[sample], additional_error=False)
    x_err = np.sqrt(cov[:, 0])
    y_err = np.sqrt(cov[:, 1])

    t_sorted = np.argsort(t)
    x = x[t_sorted]
    y = y[t_sorted]
    sample = sample[t_sorted]
    x_err = x_err[t_sorted]
    y_err = y_err[t_sorted]
    t = t[t_sorted]

    i, j = np.triu_indices(len(sample), k=1)

    dt = t[j] - t[i]
    if np.all(dt < min_dt):
        return None

    i = i[dt >= min_dt]
    j = j[dt >= min_dt]
    dt = dt[dt >= min_dt]

    dx = x[j] - x[i]
    dy = y[j] - y[i]

    vx = dx / dt
    vy = dy / dt

    vx_err = np.hypot(x_err[i], x_err[j]) / dt
    vy_err = np.hypot(y_err[i], y_err[j]) / dt

    avg_x = (x[i] + x[j]) / 2 - vx * ((t[i] + t[j]) / 2.0 - mjd_ref / 365.2425)
    avg_y = (y[i] + y[j]) / 2 - vy * ((t[i] + t[j]) / 2.0 - mjd_ref / 365.2425)

    avg_x_err = np.hypot(x_err[i], x_err[j]) / 2
    avg_y_err = np.hypot(y_err[i], y_err[j]) / 2

    X = np.column_stack((avg_x, avg_y, vx, vy))
    Sigma = np.column_stack((avg_x_err, avg_y_err, vx_err, vy_err))

    D = np.zeros((len(dt), len(dt)))
    for k in range(len(dt)):
        xk = X[k]
        sk2 = Sigma[k] ** 2
        dk = X - xk
        var_sum = Sigma**2 + sk2
        D[k] = np.sum(dk**2 / var_sum, axis=1)

    D = np.sqrt(D)

    indices = np.column_stack((sample[i], sample[j]))

    clustering = DBSCAN(
        eps=3,
        min_samples=4,
        n_jobs=1,
        metric="precomputed",
    ).fit(D)

    obj_list = []

    for cluster_id in np.unique(clustering.labels_):
        if cluster_id == -1:
            continue  # Skip noise points
        cluster_mask = clustering.labels_ == cluster_id
        cluster_indices = np.unique(indices[cluster_mask])

        if len(cluster_indices) >= n_detections:
            obj_list.append(cluster_indices.tolist())

    if len(obj_list) == 0:
        return None

    obj_list = filter_list(obj_list)
    return obj_list


def new_modest_fitter(cat, fitter, linklength=1.0 / 3600.0, cores=1):

    modest_tree = arborist(cat["XI"], cat["ETA"])
    modest_friends = find_friend(modest_tree, linklength, cores)
    modest_groups = friends_of_friends(modest_friends)

    modest_groups = [i for i in modest_groups if len(i) >= n_detections]

    modest_pm_ls = multithreader(
        new_modest_mover,
        modest_groups,
        cat,
        cores,
        chunksize=1000,
    )

    modest_pm_ls = [i for i in modest_pm_ls if i is not None]
    modest_pm_obj = []
    for i in modest_pm_ls:
        for j in i:
            modest_pm_obj += [j]

    modest_pm_obj = [i for i in modest_pm_obj if len(i) >= n_detections]

    modest_pm_arr = multi_fit5d(fitter, modest_pm_obj, cat, cores, chunksize=1000)

    return modest_pm_arr


# =============================================================================
# Fast mover algorithm
# =============================================================================


def posvel(pair, cat):
    """
    Calculates position and velocity for (xi,eta) pairs.

    Parameters
    ----------
    pairs :
    cat :

    Returns
    -------
    pair_posvel : numpy.ndarray

    """

    mjd_ref = 57388.0
    first, second = pair
    time_sep = (cat["MJD"][second] - cat["MJD"][first]) / 365.2425
    pair_posvel = np.zeros((7))
    if time_sep != 0:
        vel_ra = (cat["XI"][second] - cat["XI"][first]) / time_sep
        vel_dec = (cat["ETA"][second] - cat["ETA"][first]) / time_sep
        avg_ra = (
            np.mean(cat["XI"][[first, second]])
            - vel_ra * (np.mean(cat["MJD"][[first, second]]) - mjd_ref) / 365.2425
        )
        avg_dec = (
            np.mean(cat["ETA"][[first, second]])
            - vel_dec * (np.mean(cat["MJD"][[first, second]]) - mjd_ref) / 365.2425
        )

        pair_posvel = np.array(
            [avg_ra, avg_dec, vel_ra, vel_dec, time_sep, first, second]
        )
    return pair_posvel


def fast_movers(
    cat,
    fitter,
    pairlength=20.0 / 3600.0,
    linklength=1.0 / 3600.0,
    cores=1,
    min_pairs=4,
    min_sep=1.0,
    max_sep=20.0,
    eps=3.0,
):
    """
    Fast movers algorithm.

    Parameters
    ----------
    cat :
    linklength :
    cores :
    min_pairs :

    Returns
    -------
    fast_obj : list of lists
    """

    xi = cat["XI"]
    eta = cat["ETA"]

    cov_xy = err2cov(cat, additional_error=False)

    fast_tree = arborist(xi, eta)
    print("fast_tree built...")
    fast_pairs = fast_tree.query_pairs(r=pairlength)
    fast_pairs = np.array(list(fast_pairs))
    print("fast_pairs generated...")
    fast_posvel = multithreader(posvel, fast_pairs, cat, cores, chunksize=10000)
    print("fast_posvel completed...")
    fast_posvel = np.vstack(fast_posvel)

    cov_vx = (cov_xy[fast_pairs[:, 0], 0] + cov_xy[fast_pairs[:, 1], 0]) / fast_posvel[
        :, 4
    ] ** 2
    cov_vy = (cov_xy[fast_pairs[:, 0], 1] + cov_xy[fast_pairs[:, 1], 1]) / fast_posvel[
        :, 4
    ] ** 2

    cov = np.zeros((len(fast_pairs), 4, 4))
    cov[:, 0, 0] = (cov_xy[fast_pairs[:, 0], 0] + cov_xy[fast_pairs[:, 1], 0]) / 4
    cov[:, 1, 1] = (cov_xy[fast_pairs[:, 0], 1] + cov_xy[fast_pairs[:, 1], 1]) / 4
    cov[:, 0, 1] = cov[:, 1, 0] = (
        cov_xy[fast_pairs[:, 0], 2] + cov_xy[fast_pairs[:, 1], 2]
    ) / 4
    cov[:, 2, 2] = cov_vx
    cov[:, 3, 3] = cov_vy

    cov = cov[fast_posvel[:, 4] != 0]
    fast_posvel = fast_posvel[fast_posvel[:, 4] != 0]

    print("fast_posvel stacked...")
    pm_keep = np.logical_and(
        np.hypot(fast_posvel[:, 2], fast_posvel[:, 3]) > min_sep / 3600,
        np.hypot(fast_posvel[:, 2], fast_posvel[:, 3]) < max_sep / 3600,
    )
    print("keepers found...")
    cov = cov[pm_keep]
    fast_posvel = fast_posvel[pm_keep]

    print("prepped for 4d tree...")
    print(len(fast_posvel))

    fast_4dtree = spspace.KDTree(fast_posvel[:, :4])
    print("4d tree planted...")

    fast_4dpairs = np.array(list(fast_4dtree.query_pairs(r=linklength)))

    diff = fast_posvel[fast_4dpairs[:, 1], :4] - fast_posvel[fast_4dpairs[:, 0], :4]
    invcov = np.linalg.inv(cov[fast_4dpairs[:, 0]] + cov[fast_4dpairs[:, 1]])

    dist2 = np.einsum("...i,...ij,...j->...", diff, invcov, diff)
    dist = np.sqrt(dist2)

    mask = dist < eps
    i_idx, j_idx = fast_4dpairs[mask].T
    D = dist[mask]

    rows = np.concatenate([i_idx, j_idx])
    cols = np.concatenate([j_idx, i_idx])
    data = np.concatenate([D, D])

    D_sparse = coo_matrix(
        (data, (rows, cols)), shape=(len(fast_posvel), len(fast_posvel))
    ).tocsr()

    # D_sparse = sklearn.neighbors.sort_graph_by_row_values(
    #     D_sparse, warn_when_not_sorted=False
    # )

    print("distance matrix ready...")
    clustering = DBSCAN(
        eps=3,
        min_samples=4,
        n_jobs=cores,
        metric="precomputed",
    ).fit(D_sparse)

    print("clustering complete...")

    obj_list = []
    indices = np.arange(len(fast_posvel))

    for cluster_id in tqdm.tqdm(
        np.unique(clustering.labels_), total=len(np.unique(clustering.labels_))
    ):
        if cluster_id == -1:
            continue  # Skip noise points
        cluster_mask = clustering.labels_ == cluster_id
        cluster_indices = np.unique(indices[cluster_mask])

        if len(cluster_indices) >= n_detections:
            obj_list.append(cluster_indices.tolist())

    print("fast objects found...")

    fast_objects = obj_list

    print(str(len(fast_objects)) + " friends of friends found...")
    hipm_object_sets = [i for i in fast_objects if len(i) > min_pairs]
    print("high pm objects cleaned...")
    fast_obj = []
    for i in hipm_object_sets:
        temp_object = []
        for j in i:
            detection_pair = [int(fast_posvel[j, -2]), int(fast_posvel[j, -1])]
            temp_object += detection_pair
        fast_obj += [list(np.unique(temp_object))]
    print("fast objects grouped...")
    fast_candidates = [i for i in fast_obj if len(i) > n_detections]
    print("fast objects culled...")

    fast_pm_ls = multithreader(
        new_modest_mover, fast_candidates, cat, cores, chunksize=1000
    )

    fast_pm_ls = [i for i in fast_pm_ls if i is not None]
    fast_pm_obj = []
    for i in fast_pm_ls:
        for j in i:
            fast_pm_obj += [j]

    fast_pm_obj = [i for i in fast_pm_obj if len(i) >= n_detections]

    print("fitting fast objects...")
    fast_pm_arr = multi_fit5d(fitter, fast_pm_obj, cat, cores, chunksize=1000)
    return fast_pm_arr


def fast_checker(idx, cat, fast_cat):
    """
    Fast mover checking algorithm.

    Parameters
    ----------
    idx :
    cat :
    fast_cat :

    Returns
    -------
    obj_lol : list of lists

    """

    temp_idx_circ = (cat["XI"] - fast_cat["xi"][idx]) ** 2 + (
        cat["ETA"] - fast_cat["eta"][idx]
    ) ** 2 < ((3 * u.yr / u.mas * fast_cat["pm"][idx] + 1000) / 3600000) ** 2
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

    pm = (fast_cat["pm"][idx] / 1000).value

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


def multithreader(func, lol, cat, cores=1, chunksize=1000):
    """
    General purpose multithreader with progress bar.

    Parameters
    ----------
    func :
    lol :
    cat :
    cores :

    Returns
    -------
    ls_out :

    """

    ls_out = []
    partial_func = partial(func, cat=cat)
    with Pool(processes=cores) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(partial_func, lol, chunksize=chunksize), total=len(lol)
        ):
            ls_out.append(_)
            pass
        # ls_out = pool.map(partial_func,lol)
    pool.close()
    pool.join()
    return ls_out


def multi_fit5d(fitter, detections_groups, cat, cores=1, chunksize=1000):
    """
    5-dimensional fitting multithreader.

    Parameters
    ----------
    fitter :
    detections_groups :
    cat :
    cores :

    Returns
    -------
    clean_pm_arr :

    """

    # Multithreaded application of a 5D fitter to groups of detections
    pm_list = multithreader(fitter, detections_groups, cat, cores, chunksize)

    # Remove fits that return None
    clean_pm_list = [i for i in pm_list if i is not None]

    # Convert list to array and discard list
    clean_pm_arr = np.array(clean_pm_list, dtype=object)
    clean_pm_list.clear()

    return clean_pm_arr


# =============================================================================


if __name__ == "__main__":
    help = "Still need to write the help section"

    my_cores = 24  # os.cpu_count()

    if len(sys.argv) == 2:
        if sys.argv[1] == "-h" or sys.argv[1] == "--help":
            print(help)
            sys.exit(1)
        catname = sys.argv[1]
    else:
        print(help)
        sys.exit(1)

    header = read_cat_header(catname)

    cat = read_cat_data(catname)

    cat = clean_cat(cat)

    cat_copy = cat.copy()

    ra0 = 15.1083
    dec0 = -33.7186

    part_fit5d_no_add_err = partial(fit5d, additional_error=False)
    part_fit5d = partial(fit5d, additional_error=False)

    for _ in range(3):
        modest_pm_arr = new_modest_fitter(cat, part_fit5d, cores=my_cores)

        if len(modest_pm_arr) != 0:
            modest_tbl = output_fits(modest_pm_arr, catname, f"modest{_+1}")

            print(f"Writing {len(modest_tbl)} modest movers... (round {_ + 1})")

            modest_detections = detections_for_removal(modest_pm_arr, 5000, 50)
            if len(modest_detections) != 0:
                cat.remove_rows(modest_detections)

        cat.write(f"pre_fast{_+1}_" + catname, format="fits", overwrite=True)

        del modest_pm_arr
        del modest_tbl
        del modest_detections

    fast_pm_arr = fast_movers(cat, part_fit5d, cores=my_cores)

    if len(fast_pm_arr) != 0:
        fast_tbl = output_fits(fast_pm_arr, catname, "fast")

        print(f"Writing {len(fast_tbl)} fast movers...")

    del fast_pm_arr

    cat_copy.write("NEW_" + catname, format="fits", overwrite=True)

    print("Done!")

    sys.exit(0)
