from functools import partial

import numpy as np
from astropy.coordinates import EarthLocation, get_body, solar_system_ephemeris
from astropy.time import Time

from highpm.modest import new_modest_mover
from highpm.multithreader import multi_fit5d, multithreader
from highpm.utils import arborist
from highpm.friends_of_friends import find_friend, friends_of_friends


def forester(cat):

    mjd = np.unique(cat["MJD"])

    trees = {}

    orig_idx = np.arange(len(cat), dtype=int)

    for temp_mjd in mjd:
        temp_cat = cat[cat["MJD"] == temp_mjd]

        temp_idx = orig_idx[cat["MJD"] == temp_mjd]

        x = np.array(3600.0 * temp_cat["XI"])
        y = np.array(3600.0 * temp_cat["ETA"])

        trees[temp_mjd] = {"tree": arborist(x, y), "idx": temp_idx}

    return trees


def F_ra(ra_star, R_earth, ra_sun, dec_ecliptic):
    F_ra = R_earth * np.sin(ra_sun) * np.cos(ra_star) * np.cos(
        dec_ecliptic
    ) + R_earth * np.sin(ra_star) * np.cos(ra_sun)
    return F_ra


def F_dec(ra_star, R_earth, ra_sun, dec_ecliptic, dec_star):
    F_dec = R_earth * (
        (
            np.sin(dec_ecliptic) * np.cos(dec_star)
            - np.cos(dec_ecliptic) * np.sin(ra_star) * np.sin(dec_star)
        )
        * np.sin(ra_sun)
        - np.cos(ra_star) * np.sin(dec_star) * np.cos(ra_sun)
    )
    return F_dec


def temp_pos(mjd, xi, eta, ra, dec, pmra, pmdec, parallax, sol, config):
    # f_ra = F_ra(ra, sol.distance, np.deg2rad(sol.ra), np.deg2rad(sol.dec)).value
    # f_dec = F_dec(ra, sol.distance, np.deg2rad(sol.ra), np.deg2rad(sol.dec), dec).value

    temp_xi = (
        3600.0 * xi
        + pmra / 1000.0 * (mjd - config["mjd_ref"]) / 365.2524
        # + 3600.0 * f_ra * parallax
    )
    temp_eta = (
        3600.0 * eta
        + pmdec / 1000.0 * (mjd - config["mjd_ref"]) / 365.2524
        # + 3600.0 * f_dec * parallax
    )

    return temp_xi, temp_eta


def detection_search(trees, xi, eta, ra, dec, pmra, pmdec, parallax, mjd, sol, config):

    indicies = []

    all_ra, all_dec = [], []

    for temp_mjd, temp_sol in zip(mjd, sol):

        temp_ra, temp_dec = temp_pos(
            temp_mjd,
            xi,
            eta,
            ra,
            dec,
            pmra,
            pmdec,
            parallax,
            temp_sol,
            config,
        )
        all_ra.append(temp_ra)
        all_dec.append(temp_dec)

    all_idx = []
    for ra, dec in zip(all_ra, all_dec):
        temp_idx = []
        for temp_mjd in mjd:

            mjd_idx = trees[temp_mjd]["tree"].query_ball_point(
                np.array([ra, dec]),
                config["fastcheck"]["search_radius"],
            )

            temp_idx.extend(trees[temp_mjd]["idx"][mjd_idx])

        all_idx.extend(temp_idx)
    all_idx = np.array(all_idx, dtype=int)

    if len(all_idx) > 0:
        indicies.extend(all_idx)

    # print(len(np.unique(indicies)), "detections found for this star.")

    return np.unique(indicies)


def slow_mover_search(slow_tree, xi, eta, config):

    idx = slow_tree.query_ball_point(
        np.array([xi, eta]),
        # 10,
        config["fastcheck"]["search_radius"],
    )

    return idx


def check_slow_detections(cat, slow_cat, fast_check_pm_detections, config):
    """Returns indicies of fast_check_pm_detections that are associated with slow movers."""
    slow_cat = slow_cat[slow_cat["pm"] < config["fastcheck"]["slow_pm_threshold"]]

    slow_tree = arborist(3600.0 * slow_cat["xi"], 3600.0 * slow_cat["eta"])

    fast_detections_mask = np.ones(len(fast_check_pm_detections), dtype=bool)

    for i, detection in enumerate(fast_check_pm_detections):
        xi = 3600.0 * cat[detection]["XI"]
        eta = 3600.0 * cat[detection]["ETA"]

        slow_mover = slow_mover_search(
            slow_tree,
            xi,
            eta,
            config,
        )

        if len(slow_mover) > 0:
            fast_detections_mask[i] = False

    # print(
    #     f"Filtered {np.sum(~fast_detections_mask)} detections associated with slow movers."
    # )

    return np.array(fast_check_pm_detections)[fast_detections_mask]


def check_static_detections(cat, fast_check_pm_detections, config):

    close_friends = find_friend(
        arborist(
            3600.0 * cat["XI"][fast_check_pm_detections],
            3600.0 * cat["ETA"][fast_check_pm_detections],
        ),
        config["static_check"]["linklength"],
        cores=config["cores"],
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


def fast_checker(cat, fast_cat, slow_cat, fitter, config):

    mjd = np.unique(cat["MJD"])

    loc = EarthLocation.of_site("Cerro Tololo Interamerican Observatory")

    with solar_system_ephemeris.set("builtin"):
        sol = get_body("sun", Time(mjd, format="mjd"), loc)

    print("Building trees for detections...")

    trees = forester(cat)
    fast_check_pm_lol = []
    print("Searching for detections in fast catalog...")

    import tqdm

    for star in tqdm.tqdm(fast_cat,total=len(fast_cat), desc="Fast check progress"):
        fast_check_pm_detections = detection_search(
            trees,
            star["xi"],
            star["eta"],
            star["ra"],
            star["dec"],
            star["pmra"],
            star["pmdec"],
            star["parallax"],
            mjd,
            sol,
            config,
        )

        # print(
        #     len(fast_check_pm_detections),
        #     "detections found for this star before checks.",
        # )

        fast_check_pm_detections = check_static_detections(
            cat, fast_check_pm_detections, config
        )

        # print(
        #     len(fast_check_pm_detections),
        #     "detections found for this star after static check.",
        # )

        fast_check_pm_detections = check_slow_detections(
            cat, slow_cat, fast_check_pm_detections, config
        )

        # print(
        #     len(fast_check_pm_detections),
        #     "detections found for this star after checks.",
        # )

        # print(60 * "-")

        fast_check_pm_lol.append(fast_check_pm_detections)

    fast_check_pm_lol = [
        i for i in fast_check_pm_lol if len(i) >= config["n_detections"]
    ]

    partial_new_modest_mover = partial(new_modest_mover, mode="fastcheck")

    fast_check_pm_ls = multithreader(
        partial_new_modest_mover, fast_check_pm_lol, cat, config
    )
    fast_check_pm_ls = [i for i in fast_check_pm_ls if i is not None]

    print(f"Found {len(fast_check_pm_ls)} fast-check movers")

    fast_check_pm_obj = []
    for i in fast_check_pm_ls:
        for j in i:
            fast_check_pm_obj += [j]
    fast_check_pm_obj = [
        i for i in fast_check_pm_obj if len(i) >= config["n_detections"]
    ]

    print(
        f"Found {len(fast_check_pm_obj)} fast-check movers with at least {config['n_detections']} detections"
    )

    fast_check_pm_arr = multi_fit5d(fitter, fast_check_pm_lol, cat, config)

    return fast_check_pm_arr
