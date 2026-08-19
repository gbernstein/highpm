import numpy as np
import scipy.spatial as spspace


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


def season_labels(mjd, gap_days):
    """Greedy season grouping: a new season starts whenever a point falls
    more than gap_days past its season's first point."""
    order = np.argsort(mjd)
    sm = mjd[order]
    labels_sorted = np.empty(len(sm), dtype=np.int64)
    start, sid, n = 0, 0, len(sm)
    while start < n:
        end = np.searchsorted(sm, sm[start] + gap_days, side="right")
        labels_sorted[start:end] = sid
        sid += 1
        start = end
    labels = np.empty(len(sm), dtype=np.int64)
    labels[order] = labels_sorted
    return labels


def season_subsample(mjd, gap_days, frac, rng=None):
    """Randomly keep `frac` of detections within each season (grouped by
    gap_days), to bound per-season pair combinatorics in dense/crowded
    fast-mover candidate pools. Returns sorted indices into mjd."""
    if frac >= 1.0:
        return np.arange(len(mjd))
    rng = rng if rng is not None else np.random.default_rng()
    labels = season_labels(mjd, gap_days)
    keep = []
    for sid in np.unique(labels):
        idx = np.flatnonzero(labels == sid)
        n_keep = max(1, int(round(len(idx) * frac)))
        keep.append(idx if n_keep >= len(idx) else rng.choice(idx, n_keep, replace=False))
    return np.sort(np.concatenate(keep))


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
    cov = np.linalg.inv(np.stack(pm_arr[:, 1]))

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

    # Keep only maximal sets (drop any that is a proper subset of another).
    # Process largest-first and test each candidate only against already-kept
    # sets via an inverted index on its rarest element -- the old
    # `any(e < s for s in sets)` was O(n^2) over all pairs.
    sets = sorted({frozenset(e) for e in L}, key=len, reverse=True)
    kept = []
    postings = {}  # element -> indices of kept sets containing it
    us = []
    for e in sets:
        if e:
            rarest = min(e, key=lambda x: len(postings.get(x, ())))
            sub = any(e < kept[k] for k in postings.get(rarest, ()))
        else:
            sub = bool(kept)  # empty set is a proper subset of any non-empty set
        if sub:
            continue
        k = len(kept)
        kept.append(e)
        for x in e:
            postings.setdefault(x, []).append(k)
        us.append(list(e))
    return us


def _filter_list_ref(L):
    sets = {frozenset(e) for e in L}
    return [list(e) for e in sets if not any(e < s for s in sets)]


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


if __name__ == "__main__":
    # filter_list must match the brute-force reference (order-independent).
    import random

    def _canon(ll):
        return sorted(tuple(sorted(s)) for s in ll)

    ex = [[1, 2], [1], [2], [1, 2, 3]]
    assert _canon(filter_list(ex)) == _canon(_filter_list_ref(ex)) == _canon([[1, 2, 3]])
    for _ in range(500):
        L = [
            [random.randint(0, 8) for _ in range(random.randint(0, 5))]
            for _ in range(random.randint(0, 12))
        ]
        assert _canon(filter_list(L)) == _canon(_filter_list_ref(L)), L

    # season_labels: points within gap_days of a season's start share a
    # label; a lone point past the gap starts a new one.
    mjd = np.array([0, 1, 2, 100, 101, 200])
    labels = season_labels(mjd, gap_days=5)
    assert np.array_equal(labels, [0, 0, 0, 1, 1, 2]), labels

    # season_subsample: frac=1 is a no-op; frac<1 keeps ~frac per season,
    # never drops a singleton season, and returns real subset indices.
    assert np.array_equal(season_subsample(mjd, 5, 1.0), np.arange(len(mjd)))
    rng = np.random.default_rng(0)
    kept = season_subsample(mjd, 5, 0.5, rng=rng)
    kept_labels = season_labels(mjd, 5)[kept]
    assert len(kept) < len(mjd) and len(np.unique(kept)) == len(kept)
    for sid, n in zip(*np.unique(season_labels(mjd, 5), return_counts=True)):
        assert np.sum(kept_labels == sid) == max(1, round(n * 0.5))
    print("season_labels/season_subsample self-check OK")
    print("filter_list self-check OK")
