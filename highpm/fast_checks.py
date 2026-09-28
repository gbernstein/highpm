import numpy as np
import tqdm

from functools import partial

from highpm.modest import new_modest_mover
from highpm.multithreader import multi_fit5d, multithreader
from highpm.pmfit import count_seasons_gap
from highpm.utils import arborist
from highpm.friends_of_friends import find_friend, friends_of_friends


def forester(cat):
    # One KD-tree of detections per epoch, so a projected position is only ever
    # matched against detections observed at that same MJD.
    mjd = np.unique(cat["MJD"])
    orig_idx = np.arange(len(cat), dtype=int)
    trees = {}
    for m in mjd:
        mask = cat["MJD"] == m
        trees[m] = {
            "tree": arborist(3600.0 * cat["XI"][mask], 3600.0 * cat["ETA"][mask]),
            "idx": orig_idx[mask],
            # Parallax factor is set by the epoch's date and field, so it's
            # effectively constant across an exposure's detections.
            "par": (np.mean(cat["PAR_XI"][mask]), np.mean(cat["PAR_ETA"][mask])),
        }
    return trees


def detection_search(trees, xi, eta, pmra, pmdec, parallax, mjd, config):
    # Project the star to each epoch (proper motion plus parallax, the same
    # model fit5d solves for) and collect ALL detections observed at that same
    # epoch within the search radius. Restricting to the projection's own MJD
    # avoids spurious cross-epoch matches for high-PM stars whose track sweeps
    # across other stars' detections. This does E tree queries per star
    # instead of the old E**2.
    # xi/eta are in deg, pmra/pmdec in mas/yr, parallax in arcsec; the search
    # works in arcsec.
    r = config["fastcheck"]["search_radius"]
    dt = (mjd - config["mjd_ref"]) / 365.2425
    par = np.array([trees[m]["par"] for m in mjd])
    temp_xi = 3600.0 * xi + pmra / 1000.0 * dt + parallax * par[:, 0]
    temp_eta = 3600.0 * eta + pmdec / 1000.0 * dt + parallax * par[:, 1]

    matched = []
    for m, px, py in zip(mjd, temp_xi, temp_eta):
        idx = trees[m]["tree"].query_ball_point([px, py], r)
        if idx:
            matched.append(trees[m]["idx"][idx])
    if not matched:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(matched))


def build_slow_tree(slow_cat, config):
    """KD-tree of slow movers, built once per run (not per star)."""
    if slow_cat is None:
        return None
    slow_sel = slow_cat[slow_cat["pm"] < config["fastcheck"]["slow_pm_threshold"]]
    if len(slow_sel) == 0:
        return None
    return arborist(3600.0 * slow_sel["xi"], 3600.0 * slow_sel["eta"])


def check_slow_detections(cat, slow_tree, fast_check_pm_detections, config):
    """Drops detections that sit within the search radius of a slow mover."""
    dets = np.asarray(fast_check_pm_detections)
    if slow_tree is None or len(dets) == 0:
        return dets
    pts = np.column_stack([3600.0 * cat["XI"][dets], 3600.0 * cat["ETA"][dets]])
    neighbors = slow_tree.query_ball_point(pts, config["fastcheck"]["search_radius"])
    keep = np.array([len(nb) == 0 for nb in neighbors], dtype=bool)
    return dets[keep]


def check_static_detections(cat, fast_check_pm_detections, config):

    close_friends = find_friend(
        arborist(
            3600.0 * cat["XI"][fast_check_pm_detections],
            3600.0 * cat["ETA"][fast_check_pm_detections],
        ),
        config["static_check"]["linklength"],
        cores=None,  # per-star groups are tiny; threading them just adds overhead
    )
    close_groups = friends_of_friends(close_friends)

    mjd = cat["MJD"][fast_check_pm_detections] / 365.2524

    static_detections_mask = np.ones(len(fast_check_pm_detections), dtype=bool)

    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Circle

    # c = [plt.Circle(
    #     (
    #         3600 * cat["XI"][fast_check_pm_detections][i],
    #         3600 * cat["ETA"][fast_check_pm_detections][i],
    #     ),
    #     config["static_check"]["linklength"]/2.,
    #     color="red",
    #     fill=False,
    # ) for i in range(len(fast_check_pm_detections))]

    # fig,ax = plt.subplots()
    # for i in range(len(fast_check_pm_detections)):
    #     ax.add_patch(c[i])
    # plt.scatter(
    #     3600 * cat["XI"][fast_check_pm_detections],
    #     3600 * cat["ETA"][fast_check_pm_detections],
    #     s=5,
    #     c=(cat["MJD"][fast_check_pm_detections] - config["mjd_ref"]) / 365.2524,
    #     cmap="viridis",
    # )
    # plt.axis("equal")
    # plt.colorbar(label="MJD (years since reference)")
    # plt.xlabel("XI (arcsec)")
    # plt.ylabel("ETA (arcsec)")
    # plt.show()

    if (
        np.sum(
            np.array([len(group) for group in close_groups])
            > config["static_check"]["sub_n"]
        )
        == 1
    ):
        return np.array(fast_check_pm_detections)

    for group in close_groups:
        group_mjd = np.unique(mjd[group])
        # print(np.max(group_mjd) - np.min(group_mjd))
        # print(len(group), len(group_mjd))
        if len(group_mjd) <= config["static_check"]["sub_n"]:
            continue
        if (np.max(group_mjd) - np.min(group_mjd)) > config["static_check"]["time_sep"]:
            # print(
            #     f"Found a group of {len(group)} close detections with time separation"
            #     + f" {np.max(group_mjd) - np.min(group_mjd):.2f} years. Marking as static."
            # )
            static_detections_mask[group] = False

    # print(fast_check_pm_detections.shape, static_detections_mask.shape)

    return np.array(fast_check_pm_detections)[static_detections_mask]


def stationary_counterpart_filter(fit_rows, cat, raw_cat, cat_idx, config):
    """Flags fits that have a stationary counterpart in the uncleaned catalog.

    A fast mover has left its old position after `min_dt` years, so unfit
    detections sitting at a fit detection's position at least that far away in
    time mean the fit detections are more likely a faint slow source. Uses the
    raw catalog because cleaning (flags, extended-source, thinning) removes
    exactly the detections that would reveal the counterpart. Returns a keep
    mask over fit_rows."""
    fc = config["fastcheck"]
    radius = fc["stationary_radius"]
    min_dt = fc["stationary_min_dt"] * 365.2425
    min_hits = fc["stationary_min_hits"]

    tree = arborist(3600.0 * raw_cat["XI"], 3600.0 * raw_cat["ETA"])
    keep = np.ones(len(fit_rows), dtype=bool)
    for k, row in enumerate(fit_rows):
        members = np.asarray(row[6])
        raw_members = set(cat_idx[members].tolist())
        pts = np.column_stack(
            (3600.0 * cat["XI"][members], 3600.0 * cat["ETA"][members])
        )
        expnums = set()
        for pt, t, near in zip(
            pts, cat["MJD"][members], tree.query_ball_point(pts, radius)
        ):
            near = np.array([i for i in near if i not in raw_members], dtype=int)
            if len(near) == 0:
                continue
            far = np.abs(raw_cat["MJD"][near] - t) >= min_dt
            expnums.update(raw_cat["EXPNUM"][near[far]].tolist())
        keep[k] = len(expnums) < min_hits
    return keep


def fast_checker(cat, fast_cat, slow_cat, fitter, config, raw_cat=None, cat_idx=None):
    """Search the (cleaned) catalog along each fast mover's predicted track
    and refit the matched detections with the same fitter PM.py uses.
    raw_cat/cat_idx (uncleaned catalog and the cleaned->raw row map) enable the
    stationary-counterpart guard when fastcheck.stationary_min_hits is set."""

    mjd = np.unique(cat["MJD"])

    print("Building per-epoch trees for detections...")

    trees = forester(cat)
    slow_tree = build_slow_tree(slow_cat, config)
    fast_check_pm_lol = []
    print("Searching for detections in fast catalog...")

    for star in tqdm.tqdm(fast_cat, total=len(fast_cat), desc="Fast check progress"):
        fast_check_pm_detections = detection_search(
            trees,
            star["xi"],
            star["eta"],
            star["pmra"],
            star["pmdec"],
            star["parallax"],
            mjd,
            config,
        )

        fast_check_pm_detections = check_static_detections(
            cat, fast_check_pm_detections, config
        )

        fast_check_pm_detections = check_slow_detections(
            cat, slow_tree, fast_check_pm_detections, config
        )

        fast_check_pm_lol.append(fast_check_pm_detections)

    fast_check_pm_lol = [
        i for i in fast_check_pm_lol if len(i) >= config["n_detections"]
    ]
    print(
        f"Found {len(fast_check_pm_lol)} fast-check candidates with at least "
        f"{config['n_detections']} detections"
    )

    if config["fastcheck"].get("cluster", False):
        # Decontaminate: the search radius pulls in unrelated detections along
        # with the star's own, so keep only the mutually PM-consistent
        # subsets (same 4D DBSCAN linking as the modest/fast movers) and fit
        # those. A group may split into several objects.
        clustered = multithreader(
            partial(new_modest_mover, mode="fastcheck"), fast_check_pm_lol, cat, config
        )
        fast_check_pm_lol = [
            c
            for groups in clustered
            if groups is not None
            for c in groups
            if len(c) >= config["n_detections"]
        ]
        print(f"Clustering kept {len(fast_check_pm_lol)} candidate groups")

    # Otherwise fit5d's chi-square clipping handles outliers among the matched
    # detections, so they go to the fitter directly.
    fast_check_pm_arr = multi_fit5d(fitter, fast_check_pm_lol, cat, config)

    # Optional: require this many gap-separated seasons among the fit's
    # detections. fit5d's minSeasons uses count_seasons, whose anchored window
    # can split one long run into two "seasons".
    min_gap_seasons = config["fastcheck"].get("min_gap_seasons")
    if min_gap_seasons and len(fast_check_pm_arr):
        gap_days = config["fitting"]["t_season"] * 365.25
        keep = np.array(
            [
                count_seasons_gap(cat["MJD"][row[6]], gap_days) >= min_gap_seasons
                for row in fast_check_pm_arr
            ]
        )
        print(
            f"Gap-based season cut (>= {min_gap_seasons}) removed "
            f"{int((~keep).sum())} of {len(keep)} fits"
        )
        fast_check_pm_arr = fast_check_pm_arr[keep]

    if (
        config["fastcheck"].get("stationary_min_hits")
        and raw_cat is not None
        and len(fast_check_pm_arr)
    ):
        keep = stationary_counterpart_filter(
            fast_check_pm_arr, cat, raw_cat, cat_idx, config
        )
        print(
            f"Stationary-counterpart cut removed {int((~keep).sum())} of "
            f"{len(keep)} fits"
        )
        fast_check_pm_arr = fast_check_pm_arr[keep]
    return fast_check_pm_arr
