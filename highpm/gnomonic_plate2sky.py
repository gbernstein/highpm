"""
Convert gnomonic projection plate coordinates to sky coordinates (RA, Dec).
"""

import numpy as np


def gnomonic_plate2sky(xi, eta, ra0, dec0):
    """Convert gnomonic projection plate coordinates to sky coordinates
    (RA, Dec).

    Parameters
    ----------
    xi : float or array-like
        Gnomonic projection x-coordinate(s) in degrees.
    eta : float or array-like
        Gnomonic projection y-coordinate(s) in degrees.
    ra0 : float
        Right ascension of the projection center in degrees.
    dec0 : float
        Declination of the projection center in degrees.
    Returns
    -------
    ra : float or ndarray
        Right ascension(s) corresponding to the input plate coordinates, in
        degrees.
    dec : float or ndarray
        Declination(s) corresponding to the input plate coordinates, in
        degrees.
    Notes
    -----
    This function assumes small-angle approximations are not used and works for
    general positions on the sky. The input coordinates can be scalars or
    arrays.
    """

    xi_rad = xi * np.pi / 180.0
    eta_rad = eta * np.pi / 180.0

    ra0_rad = ra0 * np.pi / 180.0
    dec0_rad = dec0 * np.pi / 180.0

    rho = np.hypot(xi_rad, eta_rad)
    c = np.arctan(rho)

    ra_rad = ra0_rad + np.arctan(
        xi_rad
        * np.sin(c)
        / (rho * np.cos(dec0_rad) * np.cos(c) - eta_rad * np.sin(dec0_rad) * np.sin(c))
    )

    dec_rad = np.arcsin(
        np.cos(c) * np.sin(dec0_rad) + eta_rad * np.sin(c) * np.cos(dec0_rad) / rho
    )

    ra = ra_rad * 180.0 / np.pi
    dec = dec_rad * 180.0 / np.pi

    return ra, dec
