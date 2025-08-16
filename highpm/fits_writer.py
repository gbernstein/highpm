"""
This module provides functionality to write proper motion fit results and
detection information to FITS files. It includes the `output_fits` function,
which processes the fit results and writes them to structured FITS tables.
It is specifically designed for the 'fit5d' type of results, which includes
proper motion and parallax data, along with associated metadata for each
source.
"""

import fitsio
import numpy as np

from .gnomonic_plate2sky import gnomonic_plate2sky


def output_fits(pm_arr, filename, mtype, fittype="fit5d", outputname=None):
    """Write proper motion fit results and detection information to FITS files.

    Parameters
    ----------
    pm_arr : np.ndarray
        Array containing proper motion fit results and associated metadata for
        each source. The expected shape and content depend on the `fittype`.
    filename : str
        Path to the input file, used as a base for output filenames if
        `outputname` is not provided.
    mtype : str
        String identifier for the mover type (e.g., "star", "galaxy").
    fittype : str, optional
        Type of fit performed. Currently, only "fit5d" is supported.
        Default is "fit5d".
    outputname : str or None, optional
        Base name for output files. If None, output filenames are constructed
        from `filename` and `mtype`. Default is None.
    Returns
    -------
    tbl : np.ndarray
        Structured array containing the main fit results for each source.
    Raises
    ------
    RuntimeError
        If an unsupported `fittype` is provided.
    Notes
    -----
    This function writes two FITS files:
      - One containing the main mover fit results.
      - One containing detection-level information for each source.
    The function is currently tailored for "fit5d" results and uses a
    hardcoded gnomonic projection center (for Sculptor).
    """

    if fittype != "fit5d":
        tbl = None
        raise RuntimeError("This fitter is not yet supported")

    if fittype == "fit5d":
        column_dtypes = [
            ("idx", "i4"),
            ("mtype", "S10"),
            ("xi", "f8"),
            ("eta", "f8"),
            ("ra", "f8"),
            ("dec", "f8"),
            ("pm", "f8"),
            ("pmra", "f8"),
            ("pmdec", "f8"),
            ("parallax", "f8"),
            ("chisqTotal", "f8"),
            ("dof", "i4"),
            ("nClip", "i4"),
            ("c_xx", "f8"),
            ("c_yy", "f8"),
            ("c_vxvx", "f8"),
            ("c_vyvy", "f8"),
            ("c_pipi", "f8"),
            ("c_xy", "f8"),
            ("c_xvx", "f8"),
            ("c_xvy", "f8"),
            ("c_xpi", "f8"),
            ("c_yvx", "f8"),
            ("c_yvy", "f8"),
            ("c_ypi", "f8"),
            ("c_vxvy", "f8"),
            ("c_vxpi", "f8"),
            ("c_vypi", "f8"),
            ("g_mag", "f8"),
            ("r_mag", "f8"),
            ("i_mag", "f8"),
            ("z_mag", "f8"),
            ("g_n", "i4"),
            ("r_n", "i4"),
            ("i_n", "i4"),
            ("z_n", "i4"),
            ("color", "f8"),
            ("color_err", "f8"),
            ("g_spread", "f8"),
            ("r_spread", "f8"),
            ("i_spread", "f8"),
            ("z_spread", "f8"),
            ("g_spread_err", "f8"),
            ("r_spread_err", "f8"),
            ("i_spread_err", "f8"),
            ("z_spread_err", "f8"),
        ]

        idx = range(len(pm_arr))
        ls_mtype = [mtype] * len(pm_arr)

        p_fits = np.vstack(pm_arr[:, 0])
        cov = np.linalg.inv(np.stack(pm_arr[:, 1]))

        xi = np.array(p_fits[:, 0]) / 3600.0
        eta = np.array(p_fits[:, 1]) / 3600.0

        # !!!!!!!!!!!!!!!! TEMPORARY FOR SCULPTOR ONLY !!!!!!!!!!!!!!!!!!!!!!!!
        # This is the center of the gnomonic projection for Sculptor
        ra0 = 15.1083
        dec0 = -33.7186
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        ra, dec = gnomonic_plate2sky(xi, eta, ra0, dec0)
        pmra = 1000 * p_fits[:, 2]
        pmdec = 1000 * p_fits[:, 3]
        pm = np.hypot(pmra, pmdec)
        parallax = p_fits[:, 4]

        c_xx = cov[:, 0, 0]
        c_yy = cov[:, 1, 1]
        c_vxvx = cov[:, 2, 2]
        c_vyvy = cov[:, 3, 3]
        c_pipi = cov[:, 4, 4]

        c_xy = cov[:, 0, 1]
        c_xvx = cov[:, 0, 2]
        c_xvy = cov[:, 0, 3]
        c_xpi = cov[:, 0, 4]

        c_yvx = cov[:, 1, 2]
        c_yvy = cov[:, 1, 3]
        c_ypi = cov[:, 1, 4]

        c_vxvy = cov[:, 2, 3]
        c_vxpi = cov[:, 2, 4]

        c_vypi = cov[:, 3, 4]

        chisqTotal = np.array(pm_arr[:, 2], dtype=np.float64)
        dof = np.array(pm_arr[:, 4], dtype=np.float64)
        nClip = np.array(pm_arr[:, 5], dtype=np.float64)
        members = pm_arr[:, 6]
        clipped = pm_arr[:, 7]

        g_mag = np.array(pm_arr[:, 8], dtype=np.float64)
        r_mag = np.array(pm_arr[:, 9], dtype=np.float64)
        i_mag = np.array(pm_arr[:, 10], dtype=np.float64)
        z_mag = np.array(pm_arr[:, 11], dtype=np.float64)

        color = np.array(pm_arr[:, 12], dtype=np.float64)
        color_err = np.array(pm_arr[:, 13], dtype=np.float64)

        g_n = np.array(pm_arr[:, 14], dtype=np.float64)
        r_n = np.array(pm_arr[:, 15], dtype=np.float64)
        i_n = np.array(pm_arr[:, 16], dtype=np.float64)
        z_n = np.array(pm_arr[:, 17], dtype=np.float64)

        g_spread = np.array(pm_arr[:, 18], dtype=np.float64)
        r_spread = np.array(pm_arr[:, 19], dtype=np.float64)
        i_spread = np.array(pm_arr[:, 20], dtype=np.float64)
        z_spread = np.array(pm_arr[:, 21], dtype=np.float64)

        g_spread_err = np.array(pm_arr[:, 22], dtype=np.float64)
        r_spread_err = np.array(pm_arr[:, 23], dtype=np.float64)
        i_spread_err = np.array(pm_arr[:, 24], dtype=np.float64)
        z_spread_err = np.array(pm_arr[:, 25], dtype=np.float64)

        data = [
            idx,
            ls_mtype,
            xi,
            eta,
            ra,
            dec,
            pm,
            pmra,
            pmdec,
            parallax,
            chisqTotal,
            dof,
            nClip,
            c_xx,
            c_yy,
            c_vxvx,
            c_vyvy,
            c_pipi,
            c_xy,
            c_xvx,
            c_xvy,
            c_xpi,
            c_yvx,
            c_yvy,
            c_ypi,
            c_vxvy,
            c_vxpi,
            c_vypi,
            g_mag,
            r_mag,
            i_mag,
            z_mag,
            g_n,
            r_n,
            i_n,
            z_n,
            color,
            color_err,
            g_spread,
            r_spread,
            i_spread,
            z_spread,
            g_spread_err,
            r_spread_err,
            i_spread_err,
            z_spread_err,
        ]

        tbl = np.zeros(len(idx), dtype=column_dtypes)

        for i, col in enumerate(column_dtypes):
            tbl[col[0]] = data[i]

        detection_column_dtypes = [
            ("idx", "i4"),
            ("detections", "i4"),
            ("clipped", "?"),
        ]

        detection_tbl_list = []

        for i in idx:
            clipbool = len(members[i]) * [False] + len(clipped[i]) * [True]
            detections = np.hstack((members[i], clipped[i]))

            temp_data = np.array(
                list(zip([i] * len(detections), detections, clipbool)),
                dtype=detection_column_dtypes,
            )
            detection_tbl_list.append(temp_data)

        detection_tbl = np.concatenate(detection_tbl_list, axis=0)

        if outputname is None:
            fitsio.write(
                filename[:-5] + "_" + mtype + "_movers.fits",
                tbl,
                clobber=True,
            )
            fitsio.write(
                filename[:-5] + "_" + mtype + "_detections.fits",
                detection_tbl,
                clobber=True,
            )

        else:
            fitsio.write(outputname + ".fits", tbl, clobber=True)
            fitsio.write(outputname + "_detections.fits", detection_tbl, clobber=True)

    else:
        tbl = None
        raise RuntimeError("Unsupported fittype: " + fittype)

    return tbl


def output_fits_mask(fit_detection_arr, n_modest, filename, outputname=None):
    column_dtypes = [(f"modest_{i}", "?") for i in range(n_modest)]

    fit_detection_recarr = np.rec.fromarrays(fit_detection_arr, dtype=column_dtypes)

    if outputname is None:
        fitsio.write(
            filename[:-5] + "_fit_detection_mask.fits",
            fit_detection_recarr,
            clobber=True,
        )
    else:
        fitsio.write(
            outputname + "_fit_detection_mask.fits", fit_detection_recarr, clobber=True
        )
    return
