"""Fit full five-parameter PM/parallax model to detections."""

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
    """Fits astrometric parameters to input positions with optional priors and
    color terms.

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
        Best-fit parameters. If `solve_color` is True, returns first 5
        parameters.
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
    The function centers the input positions for numerical stability and
    restores the mean after fitting. Priors are implemented as additional
    diagonal terms in the normal matrix.
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
            [
                [one, zero, t, zero, par_xy[:, 0]],
                [zero, one, zero, t, par_xy[:, 1]],
            ]
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


def err2cov(temp_cat):
    """Read the per-detection fit covariance.

    Parameters
    ----------
    temp_cat : np.ndarray
        Input catalog. Must contain 'BEST_RA_ERR', 'BEST_DEC_ERR', and
        'BEST_RA_DEC_CORR'.
    Returns
    -------
    cov_xy : numpy.ndarray
        Array of shape (N, 3): (cov_xx, cov_yy, cov_xy) in the fit's (xi,
        eta) frame, for each entry of `temp_cat`.
    Notes
    -----
    BEST_RA_ERR/BEST_DEC_ERR/BEST_RA_DEC_CORR already carry the full combined
    covariance -- the ERRWIN ellipse plus the GPR/turbulence term -- rotated
    into the fit's (xi, eta) frame at packing time (see
    highpm.detection_packaging.rotate_covariances_to_healpix_frame). The
    off-diagonal is stored as a correlation coefficient (dimensionless, in
    [-1, 1]) rather than a raw covariance, since it reads less ambiguously
    next to the two per-axis errors -- so it's converted back to a covariance
    here.
    """
    ra_err = np.array(temp_cat["BEST_RA_ERR"], dtype=np.float64)
    dec_err = np.array(temp_cat["BEST_DEC_ERR"], dtype=np.float64)
    corr = np.array(temp_cat["BEST_RA_DEC_CORR"], dtype=np.float64)
    return np.array(
        [ra_err**2, dec_err**2, corr * ra_err * dec_err], dtype=np.float64
    ).T


def count_seasons(mjd, dt):
    mjd_sorted = np.sort(mjd)

    seasons = 1
    limit = mjd_sorted[0] + dt
    for mjd in mjd_sorted[1:]:
        if mjd > limit:
            seasons += 1
            limit = mjd + dt
    return seasons


def fit5d(
    indices,
    cat,
    time_sep=1.8,
    chisqClip=11.0,
    reducedChisqMax=3.0,
    parallax_prior=1e-5,
    color_prior=5.0,
    mjd_ref=57388.0,
    minPts=5,
    minSeasons=3,
    t_season=0.25,
    colorFrac=0.9,
    pm_prior=None,
):
    """Fits a 5-parameter astrometric model to a set of catalog entries, with
    iterative outlier rejection and optional color term solving.

    Parameters
    ----------
    indices : array_like
        Indices of the catalog entries to use for fitting.
    cat : np.ndarray
        Catalog containing the data to be fit. Must support row removal and
        column access by name.
    time_sep : float, optional
        Minimum time span (in years) required for a valid fit. Default is 1.8.
    chisqClip : float, optional
        Per-point chi-squared (2 dof) threshold for iterative outlier
        rejection. Default is 11.0.
    reducedChisqMax : float, optional
        Maximum accepted reduced chi-squared (chisqTotal / dof) for the
        overall fit, checked after outlier clipping stops. Default is 3.0.
    parallax_prior : float, optional
        Prior on parallax parameter. Default is 1e-5.
    color_prior : float, optional
        Prior on color term when solving for color. Default is 5.0.
    mjd_ref : float, optional
        Reference Modified Julian Date for time normalization.
        Default is 57388.0.
    minPts : int, optional
        Minimum number of points required to attempt a fit. Default is 5.
    minSeasons : int, optional
        Minimum number of distinct seasons with detections required for a valid
        fit. Default is 3.
    colorFrac : float, optional
        Fractional threshold for color mode selection. Default is 0.9.
    pm_prior : float or None, optional
        Prior on proper motion parameter. Default is None (no prior).
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
            color : float
                Fitted or listed g-i color.
            color_err : float
                Uncertainty in color.
            band_n : int
                Number of points in each band (g, r, i, z).
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

    # n_seasons = np.sum(
    #     np.histogram(temp_cat["MJD"] * day, range=(155, 170), bins=30)[0] > 0
    # )

    n_seasons = count_seasons(temp_cat["MJD"], t_season * 365.25)

    if (len(temp_cat) < minPts) or (n_seasons < minSeasons):
        # Not enough points to fit
        return None

    # Extract data from catalog
    xy = np.array([temp_cat["XI"], temp_cat["ETA"]]).T * degree
    t = (np.array(temp_cat["MJD"]) - mjd_ref) * day
    par_xy = np.array([temp_cat["PAR_XI"], temp_cat["PAR_ETA"]]).T
    expnum = temp_cat["EXPNUM"]

    # Put covariance into matrix form
    cov_xy = err2cov(temp_cat)

    nClip = 0
    clips = []

    unique_expnum = len(np.unique(expnum)) == len(expnum)

    # Begin fit/clip loop
    while (
        xy.shape[0] >= minPts
        and len(np.unique([round(i) for i in t])) >= time_sep
        and n_seasons >= minSeasons
    ):
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
                temp_cat = np.delete(temp_cat, iClip, axis=0)
                indices = np.delete(indices, iClip)

                t = np.delete(t, iClip)
                xy = np.delete(xy, iClip, axis=0)
                cov_xy = np.delete(cov_xy, iClip, axis=0)
                par_xy = np.delete(par_xy, iClip, axis=0)
                expnum = np.delete(expnum, iClip)
                nClip = nClip + 1
                continue

        # See if anything is clipped
        if (
            xy.shape[0] > minPts
            and np.max(chisq) > chisqClip
            and n_seasons >= minSeasons
        ):
            iClip = np.argmax(chisq)
            clips += [indices[iClip]]
            temp_cat = np.delete(temp_cat, iClip, axis=0)
            indices = np.delete(indices, iClip)

            t = np.delete(t, iClip)
            xy = np.delete(xy, iClip, axis=0)
            cov_xy = np.delete(cov_xy, iClip, axis=0)
            par_xy = np.delete(par_xy, iClip, axis=0)
            nClip = nClip + 1

            # n_seasons = np.sum(
            #     np.histogram(temp_cat["MJD"] * day, range=(155, 170), bins=15)[0] > 0
            # )

            n_seasons = count_seasons(temp_cat["MJD"], t_season * 365.25)

        else:
            # Fit is finished
            chisqTotal = np.sum(chisq)
            dof = 2 * xy.shape[0] - 5
            if chisqTotal / dof < reducedChisqMax and (max(t) - min(t)) > time_sep:
                # Each detection already carries the color (and which system it
                # came from, or -1 if none was known/GPR assumed the default)
                # used in its own GPR fit, so use that directly instead of
                # re-deriving a color from mode coadd magnitudes. Fit for a
                # color term unless (a) at least colorFrac of the detections
                # have a known color, AND (b) among those, colorFrac agree on
                # the same color value -- i.e. both checks use the same
                # threshold as the old mag-mode check did.
                frac_known_color = np.mean(temp_cat["COLOR_SOURCE"] != -1)
                color_err = 0.0
                use_fixed_color = frac_known_color >= colorFrac
                if use_fixed_color:
                    color, color_mode_n = sps.mode(temp_cat["COLOR"])
                    use_fixed_color = color_mode_n >= colorFrac * len(temp_cat)

                if not use_fixed_color:
                    dxy_dcolor = (
                        np.array([temp_cat["DXI_DCOLOR"], temp_cat["DETA_DCOLOR"]]).T
                        * degree
                    )
                    try:
                        p, color, color_err, _, chisq, alpha = singleFit(
                            xy,
                            cov_xy,
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
                        xy, cov_xy, t, par_xy, parallax_prior=parallax_prior
                    )

                band_n = {}
                for band in bands:
                    band_n[band] = len(temp_cat[temp_cat["BAND"] == band])

                return (
                    p,
                    alpha,
                    chisqTotal,
                    t,
                    dof,
                    nClip,
                    np.array(indices, dtype=np.int32),
                    np.array(clips, dtype=np.int32),
                    color,
                    color_err,
                    band_n["g"],
                    band_n["r"],
                    band_n["i"],
                    band_n["z"],
                )
            else:
                return None

    # Get here if fit did not start with enough points
    return None


def _demo_err2cov():
    """Self-check: err2cov reads back BEST_RA_ERR/BEST_DEC_ERR/BEST_RA_DEC_CORR
    and converts the correlation coefficient to a covariance, (var_xx, var_yy,
    corr * err_ra * err_dec). The actual ellipse-rotation and
    covariance-summing math lives in
    highpm.detection_packaging.rotate_covariances_to_healpix_frame now."""
    temp_cat = {
        "BEST_RA_ERR": np.array([2.0]),
        "BEST_DEC_ERR": np.array([1.0]),
        "BEST_RA_DEC_CORR": np.array([0.15]),
    }
    cov_xy = err2cov(temp_cat)
    assert np.isclose(cov_xy[0, 0], 4.0)
    assert np.isclose(cov_xy[0, 1], 1.0)
    assert np.isclose(cov_xy[0, 2], 0.3)  # 0.15 * 2.0 * 1.0
    print("err2cov self-check passed")


if __name__ == "__main__":
    _demo_err2cov()
