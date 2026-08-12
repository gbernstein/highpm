import os

import fitsio
import healpy as hp
import numpy as np
import numpy.lib.recfunctions as rfn

from highpm.gnomonic_converter import gnomonicJacobian


def concatenate_detections(detection_files):
    all_detections = []
    for file in detection_files:
        data = fitsio.read(file, ext=1)
        header = fitsio.read_header(file, ext=1)
        expnum = int(os.path.basename(file).split(".")[0][-8:])
        # GPR_RA0/GPR_DEC0: the GPR fit's own tangent point (from position_correction's
        # gprHeader), needed downstream to rotate BEST_RA_ERR/BEST_DEC_ERR/BEST_RA_DEC_COV
        # out of the GPR's local xi/eta frame and into the healpixel's.
        data = rfn.append_fields(
            base=data,
            names=["EXPNUM", "GPR_RA0", "GPR_DEC0"],
            data=[
                np.full(len(data), expnum, dtype="i8"),
                np.full(len(data), header["RA0"], dtype="f8"),
                np.full(len(data), header["DEC0"], dtype="f8"),
            ],
            dtypes=[np.dtype("i8"), np.dtype("f8"), np.dtype("f8")],
            usemask=False,
            asrecarray=True,
        )
        all_detections.append(data)

    detections = np.concatenate(all_detections)

    return detections


def rotate_covariances_to_healpix_frame(detections, ra0, dec0):
    """Combine the ERRWIN ellipse and GPR covariances into a single total
    covariance, rotated into the healpixel's own (xi, eta) frame -- the frame
    the fit actually runs in (see scripts/detectionPacking.py's XI/ETA,
    computed about the same ra0/dec0 via the same gnomonic projection).
    Overwrites BEST_RA_ERR/BEST_DEC_ERR/BEST_RA_DEC_COV in place with the
    combined result; downstream code (pmfit.err2cov) just reads them back.

    - ERRAWIN_WORLD/ERRBWIN_WORLD/ERRTHETAWIN_J2000 describe the SExtractor
      ellipse in the local East/North frame at each detection's own sky
      position. Rotated into (xi, eta) via J_healpix (East = cos(dec)*dRA,
      North = dDec).
    - BEST_RA_ERR/BEST_DEC_ERR/BEST_RA_DEC_COV describe the GPR covariance in
      the GPR fit's own (xi, eta) frame, tangent about (GPR_RA0, GPR_DEC0)
      (see highpm.position_correction.loadGPR). Rotated via
      M = J_healpix @ inv(J_gpr), the Jacobian mapping GPR-frame (xi, eta)
      into healpixel-frame (xi, eta).

    Both rotations use the exact gnomonic Jacobian (gnomonicJacobian), not a
    small-angle approximation, since detections can be several degrees from
    the healpixel center.
    """
    ra = detections["BEST_RA"]
    dec = detections["BEST_DEC"]

    dxi_dra, dxi_ddec, deta_dra, deta_ddec = gnomonicJacobian(ra, dec, ra0, dec0)

    # ERRWIN ellipse: local East/North -> healpixel (xi, eta).
    degree = 3600.0  # in arcsec
    a = detections["ERRAWIN_WORLD"] * degree
    b = detections["ERRBWIN_WORLD"] * degree
    pa = np.radians(detections["ERRTHETAWIN_J2000"])
    ee = a * a - b * b
    exx = (a * a + b * b + ee * np.cos(pa)) / 2.0
    eyy = (a * a + b * b - ee * np.cos(pa)) / 2.0
    exy = ee * np.sin(pa) / 2.0

    cosdec = np.cos(np.radians(dec))
    j00, j01 = dxi_dra / cosdec, dxi_ddec
    j10, j11 = deta_dra / cosdec, deta_ddec

    err_xx = j00 * j00 * exx + 2 * j00 * j01 * exy + j01 * j01 * eyy
    err_yy = j10 * j10 * exx + 2 * j10 * j11 * exy + j11 * j11 * eyy
    err_xy = j00 * j10 * exx + (j00 * j11 + j01 * j10) * exy + j01 * j11 * eyy

    # GPR covariance: (xi, eta) tangent at (GPR_RA0, GPR_DEC0) -> healpixel (xi, eta).
    gxi_dra, gxi_ddec, geta_dra, geta_ddec = gnomonicJacobian(
        ra, dec, detections["GPR_RA0"], detections["GPR_DEC0"]
    )
    det_gpr = gxi_dra * geta_ddec - gxi_ddec * geta_dra
    inv00, inv01 = geta_ddec / det_gpr, -gxi_ddec / det_gpr
    inv10, inv11 = -geta_dra / det_gpr, gxi_dra / det_gpr

    m00 = dxi_dra * inv00 + dxi_ddec * inv10
    m01 = dxi_dra * inv01 + dxi_ddec * inv11
    m10 = deta_dra * inv00 + deta_ddec * inv10
    m11 = deta_dra * inv01 + deta_ddec * inv11

    cxx = detections["BEST_RA_ERR"] ** 2
    cyy = detections["BEST_DEC_ERR"] ** 2
    cxy = detections["BEST_RA_DEC_COV"]

    gpr_xx = m00 * m00 * cxx + 2 * m00 * m01 * cxy + m01 * m01 * cyy
    gpr_yy = m10 * m10 * cxx + 2 * m10 * m11 * cxy + m11 * m11 * cyy
    gpr_xy = m00 * m10 * cxx + (m00 * m11 + m01 * m10) * cxy + m01 * m11 * cyy

    detections["BEST_RA_ERR"] = np.sqrt(err_xx + gpr_xx)
    detections["BEST_DEC_ERR"] = np.sqrt(err_yy + gpr_yy)
    detections["BEST_RA_DEC_COV"] = err_xy + gpr_xy

    return rfn.drop_fields(
        detections,
        ["GPR_RA0", "GPR_DEC0", "ERRAWIN_WORLD", "ERRBWIN_WORLD", "ERRTHETAWIN_J2000"],
    )


def clean_err_detections(detections):

    good_detections = (detections["BEST_RA_ERR"] > 0.0) & (
        detections["BEST_DEC_ERR"] > 0.0
    )

    return detections[good_detections]


def clean_snr_detections(detections, snr_threshold=5.0):

    good_detections = (
        detections["FLUX_PSF"] / detections["FLUXERR_PSF"]
    ) >= snr_threshold

    return detections[good_detections]


def clean_healpix_detections(detections, ipix, nside=32, subside=16):
    """
    This function removes detections that are more than 'overlap' degrees away
    from the edge of the healpixel.
    """
    fine = nside * subside
    # Children of ipix are contiguous in NESTED ordering: parent*ratio^2 + offset.
    ratio2 = subside * subside
    parent_nest = int(hp.ring2nest(nside, ipix))
    ipix_sub = hp.nest2ring(fine, parent_nest * ratio2 + np.arange(ratio2, dtype=np.int64))

    neigh = hp.get_all_neighbours(fine, ipix_sub)  # (8, len(ipix_sub)), -1 padded
    good_subpix = np.unique(np.concatenate([ipix_sub, neigh[neigh >= 0].ravel()]))

    detections_ipix = hp.ang2pix(
        fine,
        np.radians(90.0 - detections["BEST_DEC"]),
        np.radians(detections["BEST_RA"]),
    )

    return detections[np.isin(detections_ipix, good_subpix)]


def get_healpix_center(ipix, nside=32):
    theta, phi = hp.pix2ang(nside, ipix)
    ra_center = np.degrees(phi)
    dec_center = 90.0 - np.degrees(theta)
    return ra_center, dec_center


def get_exposures_near_healpix(ipix, ra, dec, expnums, nside=32):

    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)

    pixel_id = hp.ang2pix(nside, theta, phi)

    neighbors = hp.get_all_neighbours(nside, ipix)
    valid_neighbors = neighbors[neighbors >= 0]

    valid_pixels = np.append(valid_neighbors, ipix)

    mask = np.isin(pixel_id, valid_pixels)

    return np.unique(expnums[mask])


def _demo_rotate_covariances_to_healpix_frame():
    """Self-check for rotate_covariances_to_healpix_frame: identity when the
    GPR tangent point matches the healpixel center, a nontrivial change under
    a 3-degree offset (cross-checked below), and correct summing-in of the
    ERRWIN ellipse at the tangent point (where its East/North frame coincides
    with (xi, eta), so it should add straight onto the diagonal)."""
    dtype = [
        ("BEST_RA", "f8"), ("BEST_DEC", "f8"),
        ("BEST_RA_ERR", "f8"), ("BEST_DEC_ERR", "f8"), ("BEST_RA_DEC_COV", "f8"),
        ("GPR_RA0", "f8"), ("GPR_DEC0", "f8"),
        ("ERRAWIN_WORLD", "f8"), ("ERRBWIN_WORLD", "f8"), ("ERRTHETAWIN_J2000", "f8"),
    ]
    det = np.zeros(1, dtype=dtype)
    det["BEST_RA"], det["BEST_DEC"] = 10.0, -30.0
    det["BEST_RA_ERR"], det["BEST_DEC_ERR"], det["BEST_RA_DEC_COV"] = 0.2, 0.1, 0.03
    det["GPR_RA0"], det["GPR_DEC0"] = 10.0, -30.0

    same_center = rotate_covariances_to_healpix_frame(det.copy(), 10.0, -30.0)
    assert np.isclose(same_center["BEST_RA_ERR"][0], 0.2)
    assert np.isclose(same_center["BEST_DEC_ERR"][0], 0.1)
    assert np.isclose(same_center["BEST_RA_DEC_COV"][0], 0.03)

    det_with_ellipse = det.copy()
    det_with_ellipse["ERRAWIN_WORLD"] = 0.3 / 3600.0
    det_with_ellipse["ERRBWIN_WORLD"] = 0.15 / 3600.0
    with_ellipse = rotate_covariances_to_healpix_frame(det_with_ellipse, 10.0, -30.0)
    assert np.isclose(with_ellipse["BEST_RA_ERR"][0] ** 2, 0.2**2 + 0.3**2)
    assert np.isclose(with_ellipse["BEST_DEC_ERR"][0] ** 2, 0.1**2 + 0.15**2)
    assert np.isclose(with_ellipse["BEST_RA_DEC_COV"][0], 0.03)

    ra0_hp, dec0_hp = 13.0, -30.0
    offset = rotate_covariances_to_healpix_frame(det.copy(), ra0_hp, dec0_hp)
    assert not np.isclose(offset["BEST_RA_ERR"][0], 0.2)

    # Cross-check against the same M = J_healpix @ inv(J_gpr) congruence
    # transform computed independently here, rather than by re-deriving it
    # from rotate_covariances_to_healpix_frame's own internals.
    ra, dec = det["BEST_RA"], det["BEST_DEC"]
    j_hp = np.array(gnomonicJacobian(ra, dec, ra0_hp, dec0_hp)).reshape(2, 2)
    j_gpr = np.array(
        gnomonicJacobian(ra, dec, det["GPR_RA0"], det["GPR_DEC0"])
    ).reshape(2, 2)
    m = j_hp @ np.linalg.inv(j_gpr)
    cov_before = np.array([[0.2**2, 0.03], [0.03, 0.1**2]])
    cov_after = m @ cov_before @ m.T
    assert np.isclose(offset["BEST_RA_ERR"][0] ** 2, cov_after[0, 0])
    assert np.isclose(offset["BEST_DEC_ERR"][0] ** 2, cov_after[1, 1])
    assert np.isclose(offset["BEST_RA_DEC_COV"][0], cov_after[0, 1])
    print("rotate_covariances_to_healpix_frame self-check passed")


if __name__ == "__main__":
    _demo_rotate_covariances_to_healpix_frame()
