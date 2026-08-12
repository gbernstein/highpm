import os

import fitsio
import healpy as hp
import numpy as np
import numpy.lib.recfunctions as rfn


def concatenate_detections(detection_files):
    all_detections = []
    for file in detection_files:
        data = fitsio.read(file, ext=1)
        expnum = int(os.path.basename(file).split(".")[0][-8:])
        data = rfn.append_fields(
            base=data,
            names="EXPNUM",
            data=np.full(len(data), expnum, dtype="i8"),
            dtypes=[np.dtype("i8")],
            usemask=False,
            asrecarray=True,
        )
        all_detections.append(data)

    detections = np.concatenate(all_detections)

    return detections


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
