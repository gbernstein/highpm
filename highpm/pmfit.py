"""
Fit full five-parameter PM/parallax model to observations.
"""

## ??? Need to gaurd against bad fits,  perhaps a v weak prior on PM

import numpy as np
import scipy.stats as sps


def singleFit(
    xy_in,
    cov_xy,
    t,
    par_xy,
    parallax_prior=None,
    solve_color=False,
    dxy_dcolor=None,
    color_prior=None,
    pm_prior=None,
):
    """Fit 5-parameter model to stellar observations, where:
    `xy_in` is Nx2 array of observations of the star, in arcsec
    `cov` is Nx3 array giving (sig^2_x, sig^2_y, cov_xy) for each
    `t`  is time of observation of each (in yrs)
    `par_xy` are -1*(projected earth position components), in AU
    `parallax_prior` is prior sigma for parallax, in arcsec/yr
    Returns:
    `soln`  the 5-param covariance (x0, y0, vx, vy, parallax)
    `cov`   covariance matrix of this
    `chisq` vector of chisq of each observation."""

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
        nparams = 6
    else:
        # Build 2 x N x 5 matrix of coefficients
        A = np.array(
            [[one, zero, t, zero, par_xy[:, 0]], [zero, one, zero, t, par_xy[:, 1]]]
        )
        nparams = 5

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

    return p, fit, chisq, alpha


def err2cov(temp_cat, additional_error=True):
    degree = 3600.0  # in arcsec

    a = np.array(temp_cat["ERRAWIN_WORLD"]) * degree
    b = np.array(temp_cat["ERRAWIN_WORLD"]) * degree

    if additional_error:
        a = np.hypot(a, 0.1)
        b = np.hypot(b, 0.1)

    turb_a = np.array(temp_cat["NEW_RA_ERR"])
    turb_b = np.array(temp_cat["NEW_DEC_ERR"])

    # turb_ee = turb_aa - turb_bb

    # np.seterr(divide="ignore", invalid="ignore")

    # turb_pa = 0.5 * np.arctan(2 * np.divide(turb_ab, turb_ee))
    # turb_pa[np.isnan(turb_pa)] = 0
    # turb_sig_aa = 0.5 * (turb_aa + turb_bb - np.hypot(turb_ee, turb_ab))
    # turb_sig_bb = 0.5 * (turb_aa + turb_bb + np.hypot(turb_ee, turb_ab))

    pa = np.zeros(len(temp_cat), dtype=float)
    turb_pa = np.zeros(len(temp_cat), dtype=float)

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

    turb_ee = turb_a * turb_a - turb_b * turb_b
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
    # parallax_prior=0.15,
    parallax_prior=1e-5,
    color_prior=5.0,
    mjd_ref=57388.0,
    minPts=5,
    colorFrac=0.9,
    pm_prior=None,
    additional_error=True,
):
    """Execute 5d fit, with outlier rejection, on entries
    in the catalog at the rows specified by `indices`.
    Input catalog units are degrees but all parallax
    results come in arcsec / yrs.
    The catalog needs to have columns
    `XI, ETA, ERRAWIN_WORLD, ERRBWIN_WORLD, ERRTHETAWIN_j2000,`
    `MJD`, `PAR_XI`, `PAR_ETA`
    `chisqClip`:  parameter gives clipping threshold
        for a 2d measurement (default value is p~0.004).
    `parallax_prior`: sigma for prior on parallax
    `mjd_ref` is epoch date for positions, defaults to 2016.0
    `err_floor` is sigma of additional (circular) error to add ###
    Returns:
    `p`: best-fit 5d parameters
    `alpha`: inverse-covariance matrix for fit
    `chisq`: total for fit
    `dof`: total for fit
    `nClip`: number of points clipped
    Returns `None` if there are insufficient data for a fit."""

    bands = ["g", "r", "i", "z", "Y"]

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
        p, fit, chisq, alpha = singleFit(
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
                Y_mag = sps.mode(temp_cat["MAG_AUTO_Y"])[0]

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
                        p, color, color_err, fit, chisq, alpha = singleFit(
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

                    # fitColor = True

                else:
                    p, fit, chisq, alpha = singleFit(
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
                    Y_mag,
                    color,
                    color_err,
                    band_n["g"],
                    band_n["r"],
                    band_n["i"],
                    band_n["z"],
                    band_n["Y"],
                    band_spread["g"],
                    band_spread["r"],
                    band_spread["i"],
                    band_spread["z"],
                    band_spread["Y"],
                    band_spread_err["g"],
                    band_spread_err["r"],
                    band_spread_err["i"],
                    band_spread_err["z"],
                    band_spread_err["Y"],
                )
            else:
                return None

    # Get here if fit did not start with enough points
    return None
