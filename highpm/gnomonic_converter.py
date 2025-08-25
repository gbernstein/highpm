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


def projectGnomonic(ra, dec, delta_ra, delta_dec, ra0, dec0):
    deg2rad = np.pi / 180.0

    # Convert reference point and coordinates to radians
    ra0_rad = ra0 * deg2rad
    dec0_rad = dec0 * deg2rad
    ra_rad = np.asarray(ra) * deg2rad
    dec_rad = np.asarray(dec) * deg2rad

    # Convert direction vectors to radians
    delta_ra_rad = np.asarray(delta_ra) * deg2rad
    delta_dec_rad = np.asarray(delta_dec) * deg2rad

    # Angular differences
    dra = ra_rad - ra0_rad

    # Trigonometric terms
    sin_dra = np.sin(dra)
    cos_dra = np.cos(dra)
    sin_d0 = np.sin(dec0_rad)
    cos_d0 = np.cos(dec0_rad)
    sin_d = np.sin(dec_rad)
    cos_d = np.cos(dec_rad)

    # Denominator D
    D = sin_d0 * sin_d + cos_d0 * cos_d * cos_dra
    valid = D > 0

    # Initialize outputs with NaNs
    x = np.full_like(ra, np.nan)
    y = np.full_like(ra, np.nan)
    dx_proj = np.full_like(ra, np.nan)
    dy_proj = np.full_like(ra, np.nan)

    if not np.any(valid):
        return x, y, dx_proj, dy_proj

    # Compute x, y for valid points
    num_x = cos_d[valid] * sin_dra[valid]
    num_y = cos_d0 * sin_d[valid] - sin_d0 * cos_d[valid] * cos_dra[valid]
    D_valid = D[valid]
    x_valid = num_x / D_valid
    y_valid = num_y / D_valid

    # Partial derivatives (Jacobian components)
    term1 = sin_d0 * cos_d[valid] - cos_d0 * sin_d[valid] * cos_dra[valid]
    term2 = cos_d0 * cos_d[valid] + sin_d0 * sin_d[valid] * cos_dra[valid]

    dx_dra = (
        cos_d[valid]
        * (cos_dra[valid] * D_valid + cos_d0 * cos_d[valid] * sin_dra[valid] ** 2)
    ) / (D_valid**2)
    dx_ddec = (-sin_d[valid] * sin_dra[valid] * D_valid - num_x * term1) / (D_valid**2)

    dy_dra = (
        sin_d0 * cos_d[valid] * sin_dra[valid] * D_valid
        + num_y * cos_d0 * cos_d[valid] * sin_dra[valid]
    ) / (D_valid**2)
    dy_ddec = (term2 * D_valid - num_y * term1) / (D_valid**2)

    # Project direction vectors using Jacobian
    delta_ra_rad_valid = delta_ra_rad[valid]
    delta_dec_rad_valid = delta_dec_rad[valid]

    dx_valid = dx_dra * delta_ra_rad_valid + dx_ddec * delta_dec_rad_valid
    dy_valid = dy_dra * delta_ra_rad_valid + dy_ddec * delta_dec_rad_valid

    # Assign results for valid points
    x[valid] = x_valid
    y[valid] = y_valid
    dx_proj[valid] = dx_valid
    dy_proj[valid] = dy_valid

    return np.degrees(x), np.degrees(y), np.degrees(dx_proj), np.degrees(dy_proj)
