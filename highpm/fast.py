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


def _chunked_pairs(tree, r, batch_size, workers=1):
    """Stream index pairs (i < j) within radius r of a KDTree's own points, a
    batch of query points at a time.

    tree.query_pairs(r) materializes every pair in the dense field at once
    (tens of GB before any downstream filtering) -- this yields the same
    pairs in bounded-size chunks so a caller can filter/discard each batch
    before moving on.
    """
    n = tree.n
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        neighbor_lists = tree.query_ball_point(tree.data[start:stop], r=r, workers=workers)
        counts = np.fromiter((len(nb) for nb in neighbor_lists), dtype=np.int64, count=stop - start)
        if counts.sum() == 0:
            continue
        i_idx = np.repeat(np.arange(start, stop), counts)
        j_idx = np.concatenate(neighbor_lists)
        mask = j_idx > i_idx
        if np.any(mask):
            yield i_idx[mask], j_idx[mask]


def cleanOverlapping(partition, fast_candidates, config):
    partition_candidates = [
        fast_candidates[partition[i]] for i in range(len(partition))
    ]
    for i in range(len(partition_candidates)):
        for j in range(i, len(partition_candidates)):
            # np.argmin on a 2-element list, ~10M times, was pure overhead.
            min_len_id = 0 if len(partition_candidates[i]) <= len(partition_candidates[j]) else 1
            min_len = len(partition_candidates[[i, j][min_len_id]])
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
            - 'eps_pad': float, optional
                Safety margin, in multiples of eps, for the whitened 4D
                linking search radius (covers pairs whose combined variance
                exceeds the pixel's typical value). Default is 3.0.
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

    cov_xy = err2cov(cat)

    fast_tree = arborist(x, y)

    # ponytail: process query points in batches so we never hold every raw
    # pair from the dense field in memory -- only pairs that survive the
    # min_sep/max_sep proper-motion cut (a small fraction) are accumulated.
    batch_size = config["fast"].get("pair_batch_size", 20_000)
    workers = config.get("cores", 1)

    pairs_kept, posvel_kept, cov_kept = [], [], []
    for i_idx, j_idx in _chunked_pairs(
        fast_tree, config["fast"]["pairlength"], batch_size, workers=workers
    ):
        pairs = np.stack([i_idx, j_idx], axis=1)
        posvel, cov_batch, dt, good_pairs = new_posvel(pairs, x, y, t, cov_xy, config)
        pairs = pairs[good_pairs]

        # A short dt gives a huge cov_vx ~ err^2/dt^2 (new_posvel), which makes
        # the pair's Mahalanobis distance to *everything* forgiving in stage 2
        # -- min_dt was declared as required config but never actually enforced,
        # letting short-baseline pairs act as universal connectors.
        keep = (dt >= config["fast"]["min_dt"]) & np.logical_and(
            np.hypot(posvel[:, 2], posvel[:, 3]) > config["fast"]["min_sep"],
            np.hypot(posvel[:, 2], posvel[:, 3]) < config["fast"]["max_sep"],
        )
        if np.any(keep):
            pairs_kept.append(pairs[keep])
            posvel_kept.append(posvel[keep])
            cov_kept.append(cov_batch[keep])

    if not pairs_kept:
        # ponytail: no pairs (empty/sparse catalog) -> no fast movers, nothing to fit
        return []

    fast_pairs = np.concatenate(pairs_kept)
    fast_posvel = np.concatenate(posvel_kept)
    cov = np.concatenate(cov_kept)
    print(f"fast_movers: {len(fast_pairs)} candidate pairs after pm cut", flush=True)

    # Whiten posvel by each dimension's typical (median) combined variance so
    # the KDTree radius search is itself a real Mahalanobis-scale cut instead
    # of an arbitrary fixed arcsec box. A flat radius has to stay loose
    # enough for the worst-measured pairs, which floods dense fields with
    # spatially-close-but-unrelated stars long before the exact eps cut below
    # gets a chance to reject them. eps_pad covers pairs whose true combined
    # variance exceeds this pixel's typical value; the exact recompute+mask
    # below is unchanged, so this only changes how many candidates the tree
    # search has to hand it, not which pairs ultimately pass.
    sigma0 = np.sqrt(np.maximum(np.median(cov, axis=0), 1e-12))
    fast_4dtree = spspace.KDTree(fast_posvel / sigma0)
    whitened_radius = config["fast"]["eps"] * config["fast"].get("eps_pad", 3.0)

    # Same rationale as the 2D pairing above: query_pairs() over the full 4D
    # posvel space can still be huge before the eps cut, so stream it in
    # batches and keep only pairs that pass the (much tighter) eps threshold.
    i_kept, j_kept, D_kept = [], [], []
    for i_batch, j_batch in _chunked_pairs(
        fast_4dtree, whitened_radius, batch_size, workers=workers
    ):
        diff = fast_posvel[j_batch] - fast_posvel[i_batch]
        invcov = 1.0 / (cov[i_batch] + cov[j_batch])
        dist = np.sqrt(np.sum(diff * invcov * diff, axis=1))

        mask = dist < config["fast"]["eps"]
        if np.any(mask):
            i_kept.append(i_batch[mask])
            j_kept.append(j_batch[mask])
            D_kept.append(dist[mask])

    if not i_kept:
        return []

    i_idx = np.concatenate(i_kept)
    j_idx = np.concatenate(j_kept)
    D = np.concatenate(D_kept)
    print(f"fast_movers: {len(i_idx)} candidate pairs after eps cut", flush=True)

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


if __name__ == "__main__":
    # _chunked_pairs must return the same pairs as query_pairs, in any order,
    # regardless of batch size.
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, size=(500, 2))
    tree = arborist(pts[:, 0], pts[:, 1])
    r = 8.0

    expected = {tuple(sorted(p)) for p in tree.query_pairs(r)}
    for batch_size in (1, 7, 500, 10_000):
        got = set()
        for i_idx, j_idx in _chunked_pairs(tree, r, batch_size):
            got.update(zip(i_idx.tolist(), j_idx.tolist()))
        assert got == expected, (batch_size, len(got), len(expected))
    print("_chunked_pairs self-check OK")

    # Whitened-radius search (fast_movers' 4D linking step) must not drop any
    # pair a brute-force exact Mahalanobis scan would keep, as long as no
    # pair's combined variance exceeds eps_pad**2 times the pixel's median --
    # the guarantee eps_pad is meant to provide.
    n_pts, eps, eps_pad = 300, 2.0, 3.0
    posvel = rng.uniform(-50, 50, size=(n_pts, 4))
    med_var = 0.05
    cov = rng.uniform(0.5 * med_var, 2.0 * med_var, size=(n_pts, 4))  # max/median ratio < eps_pad**2

    i_all, j_all = np.triu_indices(n_pts, k=1)
    diff = posvel[j_all] - posvel[i_all]
    invcov = 1.0 / (cov[i_all] + cov[j_all])
    true_dist = np.sqrt(np.sum(diff * invcov * diff, axis=1))
    true_matches = {(i, j) for i, j, d in zip(i_all, j_all, true_dist) if d < eps}

    sigma0 = np.sqrt(np.maximum(np.median(cov, axis=0), 1e-12))
    whitened_tree = spspace.KDTree(posvel / sigma0)
    found_matches = set()
    for i_batch, j_batch in _chunked_pairs(whitened_tree, eps * eps_pad, 37):
        diff = posvel[j_batch] - posvel[i_batch]
        invcov = 1.0 / (cov[i_batch] + cov[j_batch])
        dist = np.sqrt(np.sum(diff * invcov * diff, axis=1))
        mask = dist < eps
        found_matches.update(zip(i_batch[mask].tolist(), j_batch[mask].tolist()))

    assert found_matches == true_matches, (
        len(found_matches), len(true_matches), found_matches ^ true_matches
    )
    print("whitened-radius self-check OK")
