"""
Generate fake detections from a fake star catalog.
Includes functions to generate fake detections with applied proper motions
over multiple epochs and use cleanup functions from fake_cleanup.py.
"""

import os
import sys
import numpy as np
import healpy as hp
import fitsio
from matplotlib.path import Path

# Ensure project root (package parent) is importable when running this script directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from astropy.coordinates import solar_system_ephemeris, EarthLocation, get_body
from astropy.time import Time

from highpm.gnomonic_converter import gnomonic_plate2sky, projectGnomonic
from highpm.detection_packaging import get_healpix_center

rng = np.random.default_rng(42)


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


def generate_errors(mags: np.ndarray) -> np.ndarray:

    band_params = np.array(
        [
            [21.66, 2.51e-6],  # g
            [20.77, 1.38e-6],  # r
            [19.58, 7.86e-7],  # i
            [19.29, 8.64e-7],  # z
        ]
    )

    errors = (
        band_params[:, 1]
        * 10 ** (0.2 * (mags - band_params[:, 0]))
        * np.sqrt(1 + 10 ** (0.4 * (mags - band_params[:, 0])))
    )

    return errors


def generate_noise(
    band_idx: np.ndarray, errors: np.ndarray, turb_error: float
) -> np.ndarray:
    """
    Generate unique noise for each detection based on band and error.
    """
    noise = (
        rng.multivariate_normal(
            mean=np.zeros(2),
            cov=np.eye(2),
            size=len(errors[:, band_idx].ravel()),
        ).reshape(*errors[:, band_idx].shape, 2)
        * errors[:, band_idx, np.newaxis]
    )

    turb_nosie = (
        rng.multivariate_normal(mean=np.zeros(2), cov=np.eye(2))
        * np.ones_like(errors[:, band_idx, np.newaxis])
        * turb_error / 3600.0
    )

    return noise + turb_nosie


def detprob_logit(m, params):
    """
    logit function
    params = (m50, k, c)

    Arguments:
    - m: magnitude argument
    - params: tuple with m50, k and c
    """
    m50, k, c = params
    logit = c / (1 + np.exp(k * (m - m50)))
    return logit


def detection_completeness(mags, expnum, band):

    band_idx = np.array([{"g": 0, "r": 1, "i": 2, "z": 3}[b] for b in band])

    band_mag = mags[:, band_idx]

    print("Shape band_mag:", band_mag.shape)
    print("Shape expnum:", expnum.shape)

    full_completeness_cat = fitsio.read(
        "/home/vwetzell/gitrepos/highpm/data/y6a1c.exposures.positions.fits",
        ext=1,
        columns=["expnum", "m50", "k", "c"],
    )

    completeness_cat = full_completeness_cat[
        np.isin(full_completeness_cat["expnum"], np.unique(expnum))
    ]

    det_probs = np.zeros((len(band_mag), len(expnum)), dtype=float)
    for i, exp in enumerate(expnum):
        try:
            params = completeness_cat[completeness_cat["expnum"] == exp][
                ["m50", "k", "c"]
            ][0]

            det_probs[:, i] = detprob_logit(band_mag[:, i], params)
        except IndexError:
            print(f"Warning: No completeness data for expnum {exp}")

    detection_mask = rng.uniform(0, 1, det_probs.shape) < det_probs

    return detection_mask


def fov_mask(ra, dec, expnum):

    full_corners_cat = fitsio.read(
        "/home/vwetzell/gitrepos/highpm/data/y6a1.ccdcorners.fits.gz",
        ext=1,
        columns=["expnum", "ccdnum", "ra", "dec"],
    )

    corners_cat = full_corners_cat[
        np.isin(full_corners_cat["expnum"], np.unique(expnum))
    ]

    fov_masks = np.zeros_like(ra, dtype=bool)
    for i, exp in enumerate(expnum):

        exp_mask = expnum == exp
        corners_exp = corners_cat[corners_cat["expnum"] == exp]

        for ccd in corners_exp["ccdnum"]:
            ccd_mask = corners_exp["ccdnum"] == ccd

            # import matplotlib.pyplot as plt

            # plt.figure()
            # plt.scatter(ra[exp_mask], dec[exp_mask], s=1, alpha=0.5)
            # plt.plot(
            #     corners_exp["ra"][ccd_mask][0][:4],
            #     corners_exp["dec"][ccd_mask][0][:4],
            #     "r-",
            # )
            # plt.show()

            vertices = np.array(
                [
                    corners_exp["ra"][ccd_mask][0][:4],
                    corners_exp["dec"][ccd_mask][0][:4],
                ]
            ).T
            path = Path(vertices)
            points = np.array([ra[:, i], dec[:, i]]).T
            inside = path.contains_points(points)
            # print(np.sum(inside), "points inside exp", exp, "ccd", ccd)

            fov_masks[:, i] = fov_masks[:, i] | inside

    return fov_masks


def generate_fake_detections(
    fake_star_catalog: str,
    real_detection_catalog: str,
    healpix: int,
    nside: int,
    output_file: str,
    reference_epoch: float = 57388.0,
):
    """
    Generate fake detections for a given star catalog over multiple epochs.
    """

    real_detections = fitsio.read(
        real_detection_catalog,
        ext=1,
        columns=["EXPNUM", "MJD", "BAND"],
    )

    real_header = fitsio.read_header(
        real_detection_catalog,
        ext=1,
    )

    unique_observations = np.unique(real_detections, axis=0)

    loc = EarthLocation.of_site("Cerro Tololo Interamerican Observatory")

    with solar_system_ephemeris.set("builtin"):
        sol = get_body("sun", Time(unique_observations["MJD"], format="mjd"), loc)

    t = (unique_observations["MJD"] - reference_epoch) / 365.2425

    params = np.array(
        (
            np.repeat(real_header["RA0"], len(unique_observations)),
            sol.distance,
            sol.ra * np.pi / 180,
            sol.dec * np.pi / 180,
            np.repeat(real_header["DEC0"], len(unique_observations)),
        )
    )

    f_ra = np.zeros((len(unique_observations),))
    f_dec = np.zeros((len(unique_observations),))
    for i in range(len(unique_observations)):
        f_ra[i] = F_ra(params[0, i], params[1, i], params[2, i], params[3, i])
        f_dec[i] = F_dec(
            params[0, i], params[1, i], params[2, i], params[3, i], params[4, i]
        )

    fake_stars = fitsio.read(fake_star_catalog, ext=1)

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(8, 8))
    # plt.axis("equal")
    # plt.scatter(fake_stars["ra"], fake_stars["dec"], s=1, alpha=0.5)
    # plt.xlabel("RA (deg)")
    # plt.ylabel("Dec (deg)")
    # plt.show()

    # plt.figure()
    # plt.hist(
    #     np.hypot(fake_stars["pm_xi"], fake_stars["pm_eta"]),
    #     bins=100,
    #     range=(0, 20),
    #     histtype="step",
    # )
    # plt.xlabel("Proper Motion (arcsec/yr)")
    # plt.show()

    # Dimesions: (n_stars, n_epochs)
    xi_detections = (
        fake_stars["xi"][:, np.newaxis]
        + np.outer(fake_stars["pm_xi"], t) / 3600
        + fake_stars["parallax"][:, np.newaxis] * f_ra[np.newaxis, :] / 3600
    )
    eta_detections = (
        fake_stars["eta"][:, np.newaxis]
        + np.outer(fake_stars["pm_eta"], t) / 3600
        + fake_stars["parallax"][:, np.newaxis] * f_dec[np.newaxis, :] / 3600
    )
    print("Shape xi_detections:", xi_detections.shape)
    print("Shape eta_detections:", eta_detections.shape)

    errors = generate_errors(
        np.array(
            [
                fake_stars["g_mag"],
                fake_stars["r_mag"],
                fake_stars["i_mag"],
                fake_stars["z_mag"],
            ]
        ).T
    )
    print("Shape errors:", errors.shape)

    band_idx = np.array(
        [{"g": 0, "r": 1, "i": 2, "z": 3}[b] for b in unique_observations["BAND"]]
    )

    noise = generate_noise(band_idx, errors, turb_error=0.005)

    print("Shape noise:", noise.shape)

    xi_noisy = xi_detections + noise[:, :, 0]
    eta_noisy = eta_detections + noise[:, :, 1]

    ra_detections, dec_detections, *_ = gnomonic_plate2sky(
        xi_noisy,
        eta_noisy,
        real_header["RA0"],
        real_header["DEC0"],
    )

    print("Shape ra_detections:", ra_detections.shape)
    print("Shape dec_detections:", dec_detections.shape)

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(8, 8))
    # plt.axis("equal")
    # plt.scatter(ra_detections, dec_detections, s=1, alpha=0.5)
    # plt.xlabel("RA (deg)")
    # plt.ylabel("Dec (deg)")
    # plt.title("Fake Detections Scatter Plot")
    # plt.grid(True)
    # plt.show()

    completeness_mask = detection_completeness(
        mags=np.array(
            [
                fake_stars["g_mag"],
                fake_stars["r_mag"],
                fake_stars["i_mag"],
                fake_stars["z_mag"],
            ]
        ).T,
        expnum=unique_observations["EXPNUM"],
        band=unique_observations["BAND"],
    )

    fov_masks = fov_mask(
        ra=ra_detections,
        dec=dec_detections,
        expnum=unique_observations["EXPNUM"],
    )

    mask = completeness_mask & fov_masks

    print("Total fake detections:", len(ra_detections))
    print("Detections after completeness:", np.sum(completeness_mask))
    print("Detections inside FOV:", np.sum(fov_masks))
    print("Final detections:", np.sum(mask))

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(8, 8))
    # plt.axis("equal")
    # plt.scatter(ra_detections[mask], dec_detections[mask], s=5, alpha=1)
    # plt.xlabel("RA (deg)")
    # plt.ylabel("Dec (deg)")
    # plt.title("Final Fake Detections Scatter Plot")
    # plt.grid(True)
    # plt.show()

    dtype = np.dtype(
        [
            ("OBJECT_NUMBER", ">i4"),
            ("CCDNUM", ">i4"),
            ("NEW_RA", ">f8"),
            ("NEW_DEC", ">f8"),
            ("NEW_RA_ERR", ">f8"),
            ("NEW_DEC_ERR", ">f8"),
            ("HAS_UNIQUE_COLOR", "i1"),
            ("BAND", "<U1"),
            ("FLAGS", ">i2"),
            ("FLUX_AUTO", ">f4"),
            ("FLUXERR_AUTO", ">f4"),
            ("SPREAD_MODEL", ">f4"),
            ("SPREADERR_MODEL", ">f4"),
            ("IMAFLAGS_ISO", ">i2"),
            ("ERRAWIN_WORLD", ">f4"),
            ("XWIN_IMAGE", ">f4"),
            ("YWIN_IMAGE", ">f4"),
            ("MAG_AUTO_G", ">f4"),
            ("MAG_AUTO_R", ">f4"),
            ("MAG_AUTO_I", ">f4"),
            ("MAG_AUTO_Z", ">f4"),
            ("MJD", ">f8"),
            ("PAR_XI", ">f8"),
            ("PAR_ETA", ">f8"),
            ("NEW_XWIN_IMAGE", ">f8"),
            ("NEW_YWIN_IMAGE", ">f8"),
            ("BEST_RA", ">f8"),
            ("BEST_DEC", ">f8"),
            ("DRA_DCOLOR", ">f8"),
            ("DDEC_DCOLOR", ">f8"),
            ("EXPNUM", ">i8"),
            ("ID", ">i8"),
            ("XI", ">f8"),
            ("ETA", ">f8"),
            ("DXI_DCOLOR", ">f8"),
            ("DETA_DCOLOR", ">f8"),
        ]
    )

    fake_detections = np.zeros(np.sum(mask), dtype=dtype)

    fake_detections["OBJECT_NUMBER"] = np.arange(np.sum(mask))
    fake_detections["NEW_RA"] = ra_detections[mask]
    fake_detections["NEW_DEC"] = dec_detections[mask]
    fake_detections["NEW_RA_ERR"] = np.sqrt(2) * 0.005
    fake_detections["NEW_DEC_ERR"] = np.sqrt(2) * 0.005
    fake_detections["HAS_UNIQUE_COLOR"] = 1
    fake_detections["BAND"] = np.repeat(
        unique_observations["BAND"][np.newaxis, :], len(fake_stars), axis=0
    )[mask]
    fake_detections["FLAGS"] = 0
    fake_detections["FLUX_AUTO"] = 1e5
    fake_detections["FLUXERR_AUTO"] = 100.0
    fake_detections["SPREAD_MODEL"] = 0.0
    fake_detections["SPREADERR_MODEL"] = 0.001
    fake_detections["IMAFLAGS_ISO"] = 0
    fake_detections["ERRAWIN_WORLD"] = errors[:, band_idx][mask]
    fake_detections["XWIN_IMAGE"] = 0.0
    fake_detections["YWIN_IMAGE"] = 0.0
    fake_detections["MAG_AUTO_G"] = np.repeat(
        fake_stars["g_mag"][:, np.newaxis], len(unique_observations), axis=1
    )[mask]
    fake_detections["MAG_AUTO_R"] = np.repeat(
        fake_stars["r_mag"][:, np.newaxis], len(unique_observations), axis=1
    )[mask]
    fake_detections["MAG_AUTO_I"] = np.repeat(
        fake_stars["i_mag"][:, np.newaxis], len(unique_observations), axis=1
    )[mask]
    fake_detections["MAG_AUTO_Z"] = np.repeat(
        fake_stars["z_mag"][:, np.newaxis], len(unique_observations), axis=1
    )[mask]
    fake_detections["MJD"] = np.repeat(
        unique_observations["MJD"][np.newaxis, :], len(fake_stars), axis=0
    )[mask]
    fake_detections["PAR_XI"] = np.repeat(f_ra[np.newaxis, :], len(fake_stars), axis=0)[
        mask
    ]
    fake_detections["PAR_ETA"] = np.repeat(
        f_dec[np.newaxis, :], len(fake_stars), axis=0
    )[mask]
    fake_detections["NEW_XWIN_IMAGE"] = 0.0
    fake_detections["NEW_YWIN_IMAGE"] = 0.0
    fake_detections["BEST_RA"] = ra_detections[mask]
    fake_detections["BEST_DEC"] = dec_detections[mask]
    fake_detections["DRA_DCOLOR"] = 0.0
    fake_detections["DDEC_DCOLOR"] = 0.0
    fake_detections["EXPNUM"] = np.repeat(
        unique_observations["EXPNUM"][np.newaxis, :], len(fake_stars), axis=0
    )[mask]
    fake_detections["ID"] = fake_detections["OBJECT_NUMBER"]
    fake_detections["XI"] = xi_noisy[mask]
    fake_detections["ETA"] = eta_noisy[mask]
    fake_detections["DXI_DCOLOR"] = 0.0
    fake_detections["DETA_DCOLOR"] = 0.0

    header_dict = {
        "RA0": real_header["RA0"],
        "DEC0": real_header["DEC0"],
        "HEALPIX": healpix,
        "NSIDE": nside,
    }

    fitsio.write(output_file, fake_detections, header=header_dict, clobber=True)


if __name__ == "__main__":

    healpixel = sys.argv[1]
    nside = int(sys.argv[2])
    fake_star_catalog = f"fake_stars_hp{healpixel}_nside{nside}.fits"
    real_detection_catalog = sys.argv[3]
    output_file = f"fake_detections_hp{healpixel}_nside{nside}.fits"

    generate_fake_detections(
        fake_star_catalog=fake_star_catalog,
        real_detection_catalog=real_detection_catalog,
        healpix=int(healpixel),
        nside=nside,
        output_file=output_file,
    )
