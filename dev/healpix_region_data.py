"""Shared loader: pull PM-mover and coadd catalogs for the healpixels covering
a disc of given center + radius, and match movers/Gaia to the coadd.

Used by healpix_completeness_plot.py and healpix_precision_plot.py so both
scripts pull the same region the same way.
"""

import os
import re
from glob import glob

import fitsio
import healpy as hp
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky

MOVER_EXTS = ("modest1_movers", "modest2_movers", "modest3_movers", "fast_movers")


def disc_pixels(nside, ra0, dec0, radius_deg):
    """Healpixel indices (nested) within radius_deg of (ra0, dec0), both in degrees."""
    vec = hp.ang2vec(ra0, dec0, lonlat=True)
    pix = hp.query_disc(nside, vec, np.radians(radius_deg), inclusive=True)
    return set(int(p) for p in pix)


def _hp_index(path):
    m = re.search(r"(\d+)(?=\.fits$)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _files_in_pixels(directory, pixels, pattern="*.fits"):
    for f in sorted(glob(os.path.join(directory, pattern))):
        idx = _hp_index(f)
        if idx in pixels:
            yield f, idx


def load_movers(pmcatalog_dir, nside, pixels, pattern="*.fits", exts=MOVER_EXTS):
    found = list(_files_in_pixels(pmcatalog_dir, pixels, pattern))
    print(f"[load_movers] {len(pixels)} pixels requested, {len(found)} files found in {pmcatalog_dir} "
          f"(pattern={pattern!r})")
    if len(found) < len(pixels):
        print(f"[load_movers] WARNING: {len(pixels) - len(found)} requested pixels have no file "
              f"(wrong --nside, or the region isn't fully covered by this catalog)")

    parts = []
    for f, hp_idx in found:
        exts_data = []
        for ext in exts:
            try:
                exts_data.append(fitsio.read(f, ext=ext))
            except OSError:
                pass
        if not exts_data:
            continue
        movers = np.concatenate(exts_data)
        # Keep only movers that truly belong to this pixel (some fall in a
        # neighboring file near the pixel edge).
        in_pixel = hp.ang2pix(nside, movers["ra"], movers["dec"], lonlat=True) == hp_idx
        parts.append(movers[in_pixel])
    if not parts:
        raise RuntimeError(f"No mover files found for this region in {pmcatalog_dir}")
    raw_count = sum(len(p) for p in parts)
    movers = np.concatenate(parts)
    print(f"[load_movers] {raw_count} movers kept after in-pixel filter")
    if raw_count and raw_count < 10:
        print("[load_movers] WARNING: almost nothing survived the in-pixel filter -- "
              "check --nside matches how these files were pixelized")

    coords = SkyCoord(ra=movers["ra"] * u.degree, dec=movers["dec"] * u.degree)
    idx, d2d, _ = match_coordinates_sky(coords, coords, nthneighbor=2)
    close = d2d < 1 * u.arcsec
    keep = ~close | (np.arange(len(coords)) < idx)
    movers = movers[keep]
    print(f"[load_movers] {len(movers)} movers after dedup")
    return movers


def load_coadd(coadd_dir, pixels, pattern="*.fits"):
    found = list(_files_in_pixels(coadd_dir, pixels, pattern))
    print(f"[load_coadd] {len(pixels)} pixels requested, {len(found)} files found in {coadd_dir} "
          f"(pattern={pattern!r})")
    if len(found) < len(pixels):
        print(f"[load_coadd] WARNING: {len(pixels) - len(found)} requested pixels have no file "
              f"(wrong --nside, or the region isn't fully covered by this catalog)")
    parts = [fitsio.read(f) for f, _ in found]
    if not parts:
        raise RuntimeError(f"No coadd files found for this region in {coadd_dir}")
    coadd = np.concatenate(parts)
    print(f"[load_coadd] {len(coadd)} coadd sources loaded")
    return coadd


def match_to_coadd(cat, coadd, match_arcsec=0.1):
    cat_coords = SkyCoord(ra=cat["ra"] * u.degree, dec=cat["dec"] * u.degree)
    coadd_coords = SkyCoord(ra=coadd["ALPHAWIN_J2000"] * u.degree, dec=coadd["DELTAWIN_J2000"] * u.degree)
    idx, d2d, _ = cat_coords.match_to_catalog_sky(coadd_coords)
    close = d2d.arcsecond < match_arcsec
    matched = rfn.merge_arrays([cat[close], coadd[idx[close]]], flatten=True, usemask=False)
    print(f"[match_to_coadd] {len(cat)} input, {len(matched)} matched within {match_arcsec} arcsec")
    return matched


def _self_test():
    nside, ra, dec = 32, 53.9254167, -54.0491667
    pixels = disc_pixels(nside, ra, dec, 1.5)
    center = hp.ang2pix(nside, ra, dec, lonlat=True)
    assert center in pixels
    assert _hp_index("PM_hp09157.fits") == 9157
    assert _hp_index("dr3_gold_09157.fits") == 9157
    print(f"self-test ok: {len(pixels)} pixels for a 1.5 deg disc at nside {nside}")


if __name__ == "__main__":
    _self_test()
