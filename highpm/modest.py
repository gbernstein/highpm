from functools import partial

import numpy as np
from sklearn.cluster import DBSCAN

from .friends_of_friends import find_friend, friends_of_friends
from .multithreader import multi_fit5d, multithreader
from .pmfit import err2cov
from .utils import arborist, filter_list, new_posvel

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

    cov = err2cov(cat[sample])

    pairs = np.array(np.triu_indices(len(sample), k=1)).T

    posvel, cov, dt, good_pairs = new_posvel(pairs, x, y, t, cov, config)

    if np.all(dt < config[mode]["min_dt"]):
        return None

    pairs = pairs[good_pairs]
    keep = dt >= config[mode]["min_dt"]
    pairs = pairs[keep]
    posvel = posvel[keep]
    cov = cov[keep]
    dt = dt[keep]

    # ponytail: a handful of detections carry NaN/inf position or error
    # columns (bad upstream fits); drop any pair touched by one here so the
    # precomputed distance matrix below never feeds DBSCAN a NaN.
    finite = np.all(np.isfinite(posvel), axis=1) & np.all(np.isfinite(cov), axis=1)
    pairs = pairs[finite]
    posvel = posvel[finite]
    cov = cov[finite]
    dt = dt[finite]

    if len(dt) == 0:
        return None

    X = posvel
    Sigma = np.sqrt(cov)

    D = np.zeros((len(dt), len(dt)))
    for k in range(len(dt)):
        xk = X[k]
        sk2 = Sigma[k] ** 2
        dk = X - xk
        # ponytail: floor guards the true 0/0 case (dk==0 and var_sum==0, i.e.
        # k compared with itself and both have exactly zero reported error).
        var_sum = np.maximum(Sigma**2 + sk2, 1e-12)
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
        cluster_indices, counts = np.unique(indices[cluster_mask], return_counts=True)

        most_common = cluster_indices[np.argmax(counts)]
        mask = np.isin(indices[cluster_mask], most_common).any(axis=1)

        if np.mean(mask) > config[mode]["shared_pairs"]:
            continue  # Skip clusters with too many shared pairs

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

    print(f"Initial modest mover groups found: {len(modest_groups)}")

    modest_groups = [i for i in modest_groups if len(i) >= config["n_detections"]]

    # new_modest_mover's DBSCAN distance matrix is O(group_size^4) (it's built
    # over all pairs-of-pairs within the group). With `cores` workers running
    # concurrently, several large groups landing in flight at once can spike
    # aggregate memory well past what any single group needs (observed: 32
    # workers x ~1.5GB for 166-detection groups -> OOM). Run large groups in
    # their own pass with far fewer concurrent workers to bound that spike.
    large_threshold = config["modest"].get("large_group_size", 100)
    cores_divisor = config["modest"].get("large_group_cores_divisor", 4)

    small_groups = [g for g in modest_groups if len(g) < large_threshold]
    large_groups = [g for g in modest_groups if len(g) >= large_threshold]

    partial_new_modest_mover = partial(new_modest_mover, mode="modest")

    modest_pm_ls = multithreader(partial_new_modest_mover, small_groups, cat, config)

    if large_groups:
        large_cores = max(1, config["cores"] // cores_divisor)
        print(
            f"{len(large_groups)} large modest groups (>= {large_threshold} "
            f"detections); running with {large_cores} cores."
        )
        large_config = dict(config, cores=large_cores)
        modest_pm_ls += multithreader(
            partial_new_modest_mover, large_groups, cat, large_config
        )

    modest_pm_ls = [i for i in modest_pm_ls if i is not None]
    modest_pm_obj = []
    for i in modest_pm_ls:
        for j in i:
            modest_pm_obj += [j]

    modest_pm_obj = [i for i in modest_pm_obj if len(i) >= config["n_detections"]]

    modest_pm_arr = multi_fit5d(fitter, modest_pm_obj, cat, config)

    return modest_pm_arr
