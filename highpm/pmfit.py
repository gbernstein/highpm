"""Fit full five-parameter PM/parallax model to observations."""

## ??? Need to gaurd against bad fits,  perhaps a v weak prior on PM

from typing import Optional, Tuple

import numpy as np
import scipy.stats as sps


def singleFit(
    xy_in: np.ndarray,
    cov_xy: np.ndarray,
    t: np.ndarray,
    par_xy: np.ndarray,
    parallax_prior: Optional[float] = None,
    solve_color: bool = False,
    dxy_dcolor: Optional[np.ndarray] = None,
    color_prior: Optional[float] = None,
    pm_prior: Optional[float] = None,
) -> Tuple[
    np.ndarray, Optional[float], Optional[float], np.ndarray, np.ndarray, np.ndarray
]:
    """Fits astrometric parameters to input positions with optional priors and color
    terms.

    Parameters
    ----------
    xy_in : np.ndarray
        Array of observed positions with shape (N, 2).
    cov_xy : np.ndarray
        Covariance matrices for each position, shape (N, 3) where columns are
        [var_x, var_y, cov_xy].
    t : np.ndarray
        Array of time values for each observation, shape (N,).
    par_xy : np.ndarray
        Parallax factors for each observation, shape (N, 2).
    parallax_prior : Optional[float], optional
        Standard deviation of Gaussian prior on parallax. If None, no prior is
        applied.
    solve_color : bool, optional
        If True, fit an additional color term using dxy_dcolor.
    dxy_dcolor : Optional[np.ndarray], optional
        Derivative of position with respect to color, shape (N, 2). Required if
        solve_color is True.
    color_prior : Optional[float], optional
        Standard deviation of Gaussian prior on color term. Used only if
        solve_color is True.
    pm_prior : Optional[float], optional
        Standard deviation of Gaussian prior on proper motion.
    Returns
    -------
    p : np.ndarray
        Best-fit parameters. If `solve_color` is True, returns first 5 parameters.
    color : Optional[float]
        Best-fit color term if `solve_color` is True, else None.
    color_err : Optional[float]
        Uncertainty on color term if `solve_color` is True, else None.
    fit : np.ndarray
        Fitted positions, shape (N, 2).
    chisq : np.ndarray
        Chi-squared value per observation, shape (N,).
    alpha : np.ndarray
        Covariance or normal matrix for the fit parameters. If `solve_color` is
        True, returns the 5x5 submatrix for astrometric parameters.
    Raises
    ------
    ValueError
        If `solve_color` is True and `dxy_dcolor` is not provided.
    Notes
    -----
    The function centers the input positions for numerical stability and restores
    the mean after fitting. Priors are implemented as additional diagonal terms in
    the normal matrix.
    """

    # Remove means from xy for numerical stability
    xyMean = np.mean(xy_in, axis=0)
    xy = xy_in - xyMean
    # Build matrices
    npts = xy.shape[0]
    one = np.ones(npts, dtype=float)
    zero = np.zeros(npts, dtype=float)

    if solve_color:
        if dxy_dcolor is None:
            raise ValueError("dxy_dcolor must be provided when solve_color is True")

        A = np.array(
            [
                [one, zero, t, zero, par_xy[:, 0], dxy_dcolor[:, 0]],
                [zero, one, zero, t, par_xy[:, 1], dxy_dcolor[:, 1]],
            ]
        )
    else:
        # Build 2 x N x 5 matrix of coefficients
        A = np.array(
            [[one, zero, t, zero, par_xy[:, 0]], [zero, one, zero, t, par_xy[:, 1]]]
        )

    A = np.swapaxes(A, 1, 2)

    # xy is the 2 x N "x" vector
    # Build 2 x 2 x N vector of inverse covariances:
    det = cov_xy[:, 0] * cov_xy[:, 1] - cov_xy[:, 2] * cov_xy[:, 2]
    invC = (
        np.array([[cov_xy[:, 1], -cov_xy[:, 2]], [-cov_xy[:, 2], cov_xy[:, 0]]]) / det
    )

    # Now the solution - contract over first two dimensions of A
    alpha = np.einsum("ikm,ijk,jkn", A, invC, A)
    beta = np.einsum("ikm,ijk,kj", A, invC, xy)

    if pm_prior is not None:
        # Add prior on PM
        alpha[2, 2] += pm_prior**-2
        alpha[3, 3] += pm_prior**-2

    # Add parallax prior
    if parallax_prior is not None:
        alpha[4, 4] += parallax_prior**-2
    if solve_color and color_prior is not None:
        alpha[5, 5] += color_prior**-2
    # Solve for 5 parameters
    p = np.linalg.solve(alpha, beta)

    # Calculate fit and chisq per point
    fit = np.dot(A, p).T
    resid = xy - fit
    chisq = np.einsum("ki,ijk,kj->k", resid, invC, resid)

    # Put mean back into position
    p[:2] += xyMean

    if solve_color:
        color = p[5]

        cov = np.linalg.inv(alpha)
        color_err = np.sqrt(cov[5, 5])

        return p[:5], color, color_err, fit, chisq, alpha[:5, :5]

    return p, None, None, fit, chisq, alpha


def err2cov(temp_cat, additional_error=True):
    """Convert error ellipse parameters to covariance matrix components.

    Parameters
    ----------
    temp_cat : array-like or pandas.DataFrame
        Input catalog containing error ellipse parameters. Must contain the
        columns 'ERRAWIN_WORLD', 'NEW_RA_ERR', and 'NEW_DEC_ERR'.
    additional_error : bool, optional
        If True (default), adds an additional error of 0.1 arcsec in quadrature
        to the error ellipse axes.
    Returns
    -------
    cov_xy : numpy.ndarray
        Array of shape (N, 3), where N is the number of entries in `temp_cat`.
        Each row contains the covariance matrix components (cov_xx, cov_yy,
        cov_xy) for the corresponding entry.
    Notes
    -----
    The function assumes that the position angle is zero for all entries.
    The covariance matrix is computed as the sum of the original and
    turbulent error contributions.
    """
    
    degree = 3600.0  # in arcsec

    a = np.array(temp_cat["ERRAWIN_WORLD"]) * degree
    b = np.array(temp_cat["ERRAWIN_WORLD"]) * degree

    if additional_error:
        a = np.hypot(a, 0.1)
        b = np.hypot(b, 0.1)

    turb_a = np.array(temp_cat["NEW_RA_ERR"])
    turb_b = np.array(temp_cat["NEW_DEC_ERR"])

    pa = np.zeros(len(temp_cat), dtype=float)

    # Convert to cov
    ee = a * a - b * b
    cov_xy = (
        np.array(
            [
                a * a + b * b + ee * np.cos(pa),
                a * a + b * b - ee * np.cos(pa),
                ee * np.sin(pa),
            ]
        ).T
        / 2.0
    )

    turb_cov_xy = (
        np.array(
            [
                turb_a * turb_a + turb_b * turb_b + ee * np.cos(pa),
                turb_a * turb_a + turb_b * turb_b - ee * np.cos(pa),
                ee * np.sin(pa),
            ]
        ).T
        / 2.0
    )

    cov_xy += turb_cov_xy

    return cov_xy


def fit5d(
    indices,
    cat,
    time_sep=1.8,
    chisqClip=11.0,
    parallax_prior=1e-5,
    color_prior=5.0,
    mjd_ref=57388.0,
    minPts=5,
    colorFrac=0.9,
    pm_prior=None,
    additional_error=True,
):
    """Fits a 5-parameter astrometric model to a set of catalog entries, with iterative
    outlier rejection and optional color term solving.

    Parameters
    ----------
    indices : array_like
        Indices of the catalog entries to use for fitting.
    cat : astropy.table.Table or similar
        Catalog containing the data to be fit. Must support row removal and column
        access by name.
    time_sep : float, optional
        Minimum time span (in years) required for a valid fit. Default is 1.8.
    chisqClip : float, optional
        Chi-squared threshold for outlier rejection. Default is 11.0.
    parallax_prior : float, optional
        Prior on parallax parameter. Default is 1e-5.
    color_prior : float, optional
        Prior on color term when solving for color. Default is 5.0.
    mjd_ref : float, optional
        Reference Modified Julian Date for time normalization. Default is 57388.0.
    minPts : int, optional
        Minimum number of points required to attempt a fit. Default is 5.
    colorFrac : float, optional
        Fractional threshold for color mode selection. Default is 0.9.
    pm_prior : float or None, optional
        Prior on proper motion parameter. Default is None (no prior).
    additional_error : bool, optional
        Whether to include additional error in covariance calculation. Default is
        True.
    Returns
    -------
    tuple or None
        If the fit is successful, returns a tuple containing:
            p : ndarray
                Best-fit parameters.
            alpha : ndarray
                Covariance matrix or fit uncertainties.
            chisqTotal : float
                Total chi-squared of the fit.
            t : ndarray
                Time values used in the fit.
            dof : int
                Degrees of freedom of the fit.
            nClip : int
                Number of points clipped during fitting.
            indices : ndarray
                Indices of the points used in the final fit.
            clips : ndarray
                Indices of the points that were clipped.
            g_mag, r_mag, i_mag, z_mag : float
                Mode magnitudes in each band.
            color : float
                Fitted or computed color (g - i).
            color_err : float
                Uncertainty in color.
            band_n : int
                Number of points in each band (g, r, i, z).
            band_spread : float
                Median SPREAD_MODEL in each band.
            band_spread_err : float
                Median SPREADERR_MODEL in each band.
        Returns None if the fit is unsuccessful or does not meet criteria.
    Raises
    ------
    RuntimeWarning
        If a singular matrix is encountered during fitting.
    Notes
    -----
    This function performs iterative outlier rejection based on chi-squared and
    duplicate exposure numbers, and optionally solves for a color term if the
    photometric data are insufficient.
    """

    bands = ["g", "r", "i", "z"]

    degree = 3600.0  # in arcsec
    day = 1.0 / 365.2425  # in years

    indices = np.array(indices, dtype=int)

    temp_cat = cat[indices]

    if len(temp_cat) < minPts:
        # Not enough points to fit
        return None

    # Extract data from catalog
    xy = np.array([temp_cat["XI"], temp_cat["ETA"]]).T * degree
    t = (np.array(temp_cat["MJD"]) - mjd_ref) * day
    par_xy = np.array([temp_cat["PAR_XI"], temp_cat["PAR_ETA"]]).T
    expnum = temp_cat["EXPNUM"]

    # Put covariance into matrix form
    cov_xy = err2cov(temp_cat, additional_error=additional_error)

    nClip = 0
    clips = []

    unique_expnum = len(np.unique(expnum)) == len(expnum)

    # Begin fit/clip loop
    while xy.shape[0] >= minPts and len(np.unique([round(i) for i in t])) >= time_sep:
        p, *_, chisq, alpha = singleFit(
            xy,
            cov_xy,
            t,
            par_xy,
            parallax_prior=parallax_prior,
            pm_prior=pm_prior,
        )

        if not unique_expnum:
            unique, counts = np.unique(expnum, return_counts=True)
            if np.any(counts > 1):
                not_unique = unique[counts > 1][0]
                inu = np.argwhere(expnum == not_unique)
                iClip = inu[np.argmax(chisq[inu])][0]
                clips += [indices[iClip]]
                temp_cat.remove_row(iClip)
                indices = np.delete(indices, iClip)

                t = np.delete(t, iClip)
                xy = np.delete(xy, iClip, axis=0)
                cov_xy = np.delete(cov_xy, iClip, axis=0)
                par_xy = np.delete(par_xy, iClip, axis=0)
                expnum = np.delete(expnum, iClip)
                nClip = nClip + 1
                continue

        # See if anything is clipped
        if xy.shape[0] > minPts and np.max(chisq) > chisqClip:
            iClip = np.argmax(chisq)
            clips += [indices[iClip]]
            temp_cat.remove_row(iClip)
            indices = np.delete(indices, iClip)

            t = np.delete(t, iClip)
            xy = np.delete(xy, iClip, axis=0)
            cov_xy = np.delete(cov_xy, iClip, axis=0)
            par_xy = np.delete(par_xy, iClip, axis=0)
            nClip = nClip + 1
        else:
            # Fit is finished
            # Use true covariance for final fit

            true_cov_xy = err2cov(temp_cat, additional_error=False)

            chisqTotal = np.sum(chisq)
            dof = 2 * xy.shape[0] - 5
            if chisqTotal / dof < chisqClip and (max(t) - min(t)) > time_sep:
                g_mag, mag_mode_n = sps.mode(temp_cat["MAG_AUTO_G"])
                r_mag = sps.mode(temp_cat["MAG_AUTO_R"])[0]
                i_mag = sps.mode(temp_cat["MAG_AUTO_I"])[0]
                z_mag = sps.mode(temp_cat["MAG_AUTO_Z"])[0]

                color = g_mag - i_mag
                color_err = 0.0

                if (
                    (mag_mode_n <= colorFrac * len(temp_cat))
                    or (g_mag == -99.0)
                    or (i_mag == -99.0)
                ):
                    dxy_dcolor = (
                        np.array([temp_cat["DXI_DCOLOR"], temp_cat["DETA_DCOLOR"]]).T
                        * degree
                    )
                    try:
                        p, color, color_err, _, chisq, alpha = singleFit(
                            xy,
                            true_cov_xy,
                            t,
                            par_xy,
                            parallax_prior=parallax_prior,
                            solve_color=True,
                            dxy_dcolor=dxy_dcolor,
                            color_prior=color_prior,
                            pm_prior=pm_prior,
                        )

                    except np.linalg.LinAlgError:
                        RuntimeWarning(
                            "Singular matrix encountered in fit5d, returning None."
                        )

                        return None

                else:
                    p, *_, chisq, alpha = singleFit(
                        xy, true_cov_xy, t, par_xy, parallax_prior=parallax_prior
                    )

                band_n = {}
                band_spread = {}
                band_spread_err = {}
                for band in bands:
                    band_n[band] = len(temp_cat[temp_cat["BAND"] == band])
                    if band_n[band] > 0:
                        band_spread[band] = np.median(
                            temp_cat["SPREAD_MODEL"][temp_cat["BAND"] == band]
                        )
                        band_spread_err[band] = np.median(
                            temp_cat["SPREADERR_MODEL"][temp_cat["BAND"] == band]
                        )
                    else:
                        band_spread[band] = 0.0
                        band_spread_err[band] = 0.0

                return (
                    p,
                    alpha,
                    chisqTotal,
                    t,
                    dof,
                    nClip,
                    np.array(indices, dtype=np.int32),
                    np.array(clips, dtype=np.int32),
                    g_mag,
                    r_mag,
                    i_mag,
                    z_mag,
                    color,
                    color_err,
                    band_n["g"],
                    band_n["r"],
                    band_n["i"],
                    band_n["z"],
                    band_spread["g"],
                    band_spread["r"],
                    band_spread["i"],
                    band_spread["z"],
                    band_spread_err["g"],
                    band_spread_err["r"],
                    band_spread_err["i"],
                    band_spread_err["z"],
                )
            else:
                return None

    # Get here if fit did not start with enough points
    return None
