"""
Generate a fake proper motion catalog for a given healpixel.
Includes functions to create synthetic stars with known positions, proper motions,
and magnitudes.
"""

from math import exp
import os
import sys
import numpy as np
import healpy as hp
import fitsio
from requests import get

# Ensure project root (package parent) is importable when running this script directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from highpm.gnomonic_converter import gnomonic_plate2sky
from highpm.detection_packaging import get_healpix_center


rng = np.random.default_rng(42)


def generate_fake_mags(g_mag: np.ndarray, gi_color: np.ndarray):
    """
    Generate fake magnitudes in multiple bands based on g magnitude and g-i color.
    Stellar locus approximations from Douglas Lee Tucker
    """

    i_mag = g_mag - gi_color

    ri_color = np.where(
        gi_color < 1.81, (gi_color - 0.127) / 3.458, (gi_color - 1.196) / 1.226
    )

    r_mag = i_mag + ri_color

    rz_color = np.where(
        gi_color < 1.88, (gi_color - 0.182) / 2.125, (gi_color - 1.182) / 0.863
    )

    z_mag = r_mag - rz_color

    return r_mag, i_mag, z_mag


def generate_fake_star_catalog(
    density: float, healpix: int, nside: int, output_file: str
):
    """
    Generate a fake star catalog for a given healpixel.
    Parameters
    ----------
    density : float
        Star density in stars per square degree.
    healpix : int
        Healpix index for which to generate the star catalog.
    nside : int
        Healpix nside parameter.
    output_file : str
        Path to the output FITS file where the catalog will be saved.

    """

    # Generate star positions in gnomonic projection
    max_radius = hp.max_pixrad(nside,degrees=True)
    n_stars = int(density * np.pi * max_radius**2)

    print(max_radius, n_stars)

    r = max_radius * np.sqrt(rng.uniform(0, 1, n_stars))
    angle = rng.uniform(0, 2 * np.pi, n_stars)

    xi = r * np.cos(angle)
    eta = r * np.sin(angle)

    # Convert gnomonic coordinates to sky coordinates
    ra_center, dec_center = get_healpix_center(healpix, nside)

    print(ra_center, dec_center)

    ra, dec = gnomonic_plate2sky(xi, eta, ra_center, dec_center)

    # Generate proper motions
    uniform_pm = rng.uniform(0, 20, int(n_stars * 0.2))
    exp_pm = rng.exponential(1.0 , len(ra) - int(n_stars * 0.2))
    exp_pm = np.clip(exp_pm, 0, 20)

    pm = np.concatenate([uniform_pm, exp_pm])
    pm_angle = rng.uniform(0, 2 * np.pi, n_stars)

    pm_xi = pm * np.cos(pm_angle)
    pm_eta = pm * np.sin(pm_angle)

    # Generate paralaxes

    parallax = rng.exponential(1.0 / 10.0, n_stars)

    # Generate magnitudes and colors
    g_mag = rng.uniform(20, 27, n_stars)
    gi_color = rng.normal(-1, 1.8, n_stars)

    r_mag, i_mag, z_mag = generate_fake_mags(g_mag, gi_color)

    # Create structured array for the catalog
    dtype = np.dtype(
        [
            ("ra", "f8"),
            ("dec", "f8"),
            ("xi", "f4"),
            ("eta", "f4"),
            ("pm_xi", "f4"),
            ("pm_eta", "f4"),
            ("parallax", "f4"),
            ("g_mag", "f4"),
            ("r_mag", "f4"),
            ("i_mag", "f4"),
            ("z_mag", "f4"),
        ]
    )

    catalog = np.zeros(n_stars, dtype=dtype)
    catalog["ra"] = ra
    catalog["dec"] = dec
    catalog["xi"] = xi
    catalog["eta"] = eta
    catalog["pm_xi"] = pm_xi
    catalog["pm_eta"] = pm_eta
    catalog["parallax"] = parallax
    catalog["g_mag"] = g_mag
    catalog["r_mag"] = r_mag
    catalog["i_mag"] = i_mag
    catalog["z_mag"] = z_mag

    # Only keep stars within the healpixel

    phi = np.radians(ra)
    theta = np.radians(90.0 - dec)
    ipix_stars = hp.ang2pix(nside, theta, phi)
    mask = ipix_stars == healpix

    catalog = catalog[mask]

    # Save catalog to FITS file
    fitsio.write(output_file, catalog, clobber=True)

    print(f"Fake star catalog with {np.sum(mask)} stars written to {output_file}")


if __name__ == "__main__":

    healpixel = sys.argv[1]
    nside = int(sys.argv[2])
    density = sys.argv[3] if len(sys.argv) > 3 else 3600.0
    fake_star_catalog = f"fake_stars_hp{healpixel}_nside{nside}.fits"

    generate_fake_star_catalog(
        density=float(density),
        healpix=int(healpixel),
        nside=nside,
        output_file=fake_star_catalog,
    )
