import numpy as np
import tqdm
from scipy import spatial as spspace
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cluster import DBSCAN

from .multithreader import multi_fit5d
from .pmfit import err2cov
from .utils import arborist, filter_list, new_posvel

# =============================================================================
# Fast mover algorithm
# =============================================================================


def cleanOverlapping(partition, fast_candidates, config):
    partition_candidates = [
        fast_candidates[partition[i]] for i in range(len(partition))
    ]
    for i in range(len(partition_candidates)):
        for j in range(i, len(partition_candidates)):
            min_len_id = np.argmin(
                [len(partition_candidates[i]), len(partition_candidates[j])]
            )
            min_len = len(partition_candidates[min_len_id])
            overlap = len(partition_candidates[i] & partition_candidates[j]) / min_len
            if overlap > config["min_overlap"]:
                partition_candidates[[i, j][min_len_id]] = (
                    partition_candidates[i] | partition_candidates[j]
                )

    return partition_candidates


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

    x = np.array(cat["XI"], dtype=np.float64) * 3600.0
    y = np.array(cat["ETA"], dtype=np.float64) * 3600.0
    t = np.array(cat["MJD"], dtype=np.float64) / 365.2425

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

    # Vectorized grouping of indices by cluster label (skip noise label -1)
    obj_list = []
    labels = clustering.labels_
    if labels.size:
        valid_mask = labels != -1
        if np.any(valid_mask):
            valid_indices = np.nonzero(valid_mask)[0]
            valid_labels = labels[valid_mask]

            # Sort by label so equal labels are contiguous, then split once
            order = np.argsort(valid_labels, kind="mergesort")
            sorted_indices = valid_indices[order]
            sorted_labels = valid_labels[order]

            # Find boundaries between different labels
            split_points = np.flatnonzero(np.diff(sorted_labels)) + 1
            groups = np.split(sorted_indices, split_points)

            # Filter groups by required minimum detections
            for g in groups:
                if g.size >= config["n_detections"]:
                    obj_list.append(g.tolist())

    fast_objects = obj_list

    hipm_object_sets = [i for i in fast_objects if len(i) > config["fast"]["min_pairs"]]
    fast_obj = []
    for i in hipm_object_sets:
        temp_object = []
        for j in i:
            detection_pair = [int(fast_pairs[j, 0]), int(fast_pairs[j, 1])]
            temp_object += detection_pair
        fast_obj += [list(np.unique(temp_object))]
    fast_candidates = [set(i) for i in fast_obj if len(i) > config["n_detections"]]

    n_fast_candidates = len(fast_candidates)
    print(n_fast_candidates, "fast candidates found.")

    fast_candidate_ra = []
    fast_candidate_dec = []
    for candidate in fast_candidates:
        candidate = list(candidate)
        candidate_xi = np.median(cat["XI"][candidate])
        candidate_eta = np.median(cat["ETA"][candidate])
        fast_candidate_ra.append(candidate_xi)
        fast_candidate_dec.append(candidate_eta)

    fast_candidate_pos = np.array([fast_candidate_ra, fast_candidate_dec]).T
    fast_candidate_tree = spspace.KDTree(
        fast_candidate_pos, balanced_tree=True, compact_nodes=True
    )
    flat_candidate_tree = fast_candidate_tree.indices

    chunksize = n_fast_candidates // config["cores"]
    remainder = n_fast_candidates % config["cores"]

    sizes = np.full(config["cores"], chunksize)
    sizes[:remainder] += 1

    boundaries = np.cumsum(sizes)[:-1]
    partitions = np.split(flat_candidate_tree, boundaries)

    fast_candidates = [
        cleanOverlapping(part, fast_candidates, config) for part in partitions
    ]
    fast_candidates = [item for sublist in fast_candidates for item in sublist]

    fast_pm_obj = filter_list(fast_candidates)

    # partial_new_modest_mover = partial(new_modest_mover, mode="fast")

    # fast_pm_ls = multithreader(partial_new_modest_mover, fast_candidates, cat, config)

    # fast_pm_ls = [i for i in fast_pm_ls if i is not None]
    # fast_pm_obj = []
    # for i in fast_pm_ls:
    #     for j in i:
    #         fast_pm_obj += [j]

    # fast_pm_obj = [i for i in fast_pm_obj if len(i) >= config["n_detections"]]

    fast_pm_arr = multi_fit5d(fitter, fast_pm_obj, cat, config)
    return fast_pm_arr
