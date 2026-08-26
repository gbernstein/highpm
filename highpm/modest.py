from functools import partial

import numpy as np
import scipy.spatial as spspace
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cluster import DBSCAN

from .friends_of_friends import query_pairs_groups
from .multithreader import multi_fit5d, multithreader
from .pmfit import err2cov
from .utils import arborist, chunked_pairs, filter_list, new_posvel

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
     - Groups with `len(sample) >= config[mode].get("large_group_size", 100)`
       use a sparse, chunked distance matrix instead of the dense one, to
       bound memory on large groups.
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

    indices = np.column_stack((sample[pairs[:, 0]], sample[pairs[:, 1]]))

    large_threshold = config[mode].get("large_group_size", 100)
    if len(sample) < large_threshold:
        X = posvel
        Sigma = np.sqrt(cov)

        D = np.zeros((len(dt), len(dt)))
        for k in range(len(dt)):
            xk = X[k]
            sk2 = Sigma[k] ** 2
            dk = X - xk
            # ponytail: floor guards the true 0/0 case (dk==0 and var_sum==0,
            # i.e. k compared with itself and both have exactly zero reported
            # error).
            var_sum = np.maximum(Sigma**2 + sk2, 1e-12)
            D[k] = np.sum(dk**2 / var_sum, axis=1)

        D = np.sqrt(D)
    else:
        # ponytail: dense D above is O(n_pairs^2) = O(len(sample)^4) -- for
        # ~240-detection groups (Sculptor-core healpixels) that's a
        # multi-GB-per-group matrix, and with several groups in flight across
        # worker processes it OOMs the cluster. Mirror fast.py's
        # whitened-KDTree + chunked-candidate-search + exact-recompute +
        # sparse-matrix pattern instead: DBSCAN with a sparse precomputed
        # matrix treats any pair absent from it as farther than eps, which is
        # exactly what the exact-recompute-then-eps-cut below guarantees.
        X = posvel
        sigma0 = np.sqrt(np.maximum(np.median(cov, axis=0), 1e-12))
        tree = spspace.KDTree(X / sigma0)
        eps_pad = config[mode].get("eps_pad", 3.0)
        whitened_radius = config[mode]["eps"] * eps_pad
        # ponytail: a large group's whole pair-point set (~16k-29k) is smaller
        # than fast.py's pair_batch_size default (20_000, sized for a
        # multi-million-point global field) -- so chunked_pairs never actually
        # split a dense group into multiple batches, and each of lines
        # 145/148/149 below allocated one candidate-sized temp array per
        # group (measured: single dense groups spiking several GB). A batch
        # size scaled to this group's own density, not the global field's,
        # keeps each batch's temp arrays bounded regardless of group density.
        batch_size = config[mode].get("large_group_batch_size", 2_000)
        workers = config.get("cores", 1)

        i_kept, j_kept, d_kept = [], [], []
        for i_batch, j_batch in chunked_pairs(tree, whitened_radius, batch_size, workers=workers):
            diff = X[j_batch] - X[i_batch]
            # ponytail: floor guards 0*inf=nan when two pairs share identical
            # posvel and both report exactly zero variance.
            invvar = 1.0 / np.maximum(cov[i_batch] + cov[j_batch], 1e-12)
            d = np.sqrt(np.sum(diff * invvar * diff, axis=1))
            mask = d < config[mode]["eps"]
            if np.any(mask):
                i_kept.append(i_batch[mask])
                j_kept.append(j_batch[mask])
                d_kept.append(d[mask])

        if not i_kept:
            return None

        i_idx = np.concatenate(i_kept)
        j_idx = np.concatenate(j_kept)
        d_all = np.concatenate(d_kept)

        rows = np.concatenate([i_idx, j_idx])
        cols = np.concatenate([j_idx, i_idx])
        data = np.concatenate([d_all, d_all])

        D = csr_matrix(coo_matrix((data, (rows, cols)), shape=(len(dt), len(dt))))

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
     `query_pairs_groups`, `multithreader`, `new_modest_mover`,
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
    modest_groups = query_pairs_groups(modest_tree, linklength)

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
