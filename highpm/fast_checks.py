from functools import partial

import numpy as np
from astropy.coordinates import EarthLocation, get_body, solar_system_ephemeris
from astropy.time import Time

from highpm.modest import new_modest_mover
from highpm.multithreader import multi_fit5d, multithreader
from highpm.utils import arborist


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

        temp_idx = trees[temp_mjd]["tree"].query_ball_point(
            np.array([temp_ra, temp_dec]),
            config["fastcheck"]["search_radius"],
        )

        idx = trees[temp_mjd]["idx"][temp_idx]

        indicies.extend(idx)

    return indicies


def fast_checker(cat, fast_cat, fitter, config):

    mjd = np.unique(cat["MJD"])

    loc = EarthLocation.of_site("Cerro Tololo Interamerican Observatory")

    with solar_system_ephemeris.set("builtin"):
        sol = get_body("sun", Time(mjd, format="mjd"), loc)

    print("Building trees for detections...")

    trees = forester(cat)
    fast_check_pm_lol = []
    print("Searching for detections in fast catalog...")
    for star in fast_cat:
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

        fast_check_pm_lol.append(fast_check_pm_detections)

    partial_new_modest_mover = partial(new_modest_mover, mode="fast")

    fast_check_pm_ls = multithreader(
        partial_new_modest_mover, fast_check_pm_lol, cat, config
    )
    fast_check_pm_ls = [i for i in fast_check_pm_ls if i is not None]
    fast_check_pm_obj = []
    for i in fast_check_pm_ls:
        for j in i:
            fast_check_pm_obj += [j]
    fast_check_pm_obj = [
        i for i in fast_check_pm_obj if len(i) >= config["n_detections"]
    ]

    fast_check_pm_arr = multi_fit5d(fitter, fast_check_pm_lol, cat, config)

    return fast_check_pm_arr
