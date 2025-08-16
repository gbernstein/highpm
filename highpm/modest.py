from functools import partial
import numpy as np
from friends_of_friends import find_friend, friends_of_friends
from pmfit import err2cov
from sklearn.cluster import DBSCAN
from utils import arborist, new_posvel, filter_list
from multithreader import multithreader, multi_fit5d


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
