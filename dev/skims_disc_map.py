"""Sky map of skim-visit coverage (/data8/shared/decampm/[GRIZ]) with each visit
drawn as a 1 deg radius circle: number of visits overlapping each nside-512
pixel. See skims_focalplane_maps.py for the DECam focal-plane version.

Writes dev/skims_disc_skymap.png and dev/skims_disc_counts_nside512.npy.
"""

import glob
import os
import re
import sys

import healpy as hp
import numpy as np
from astropy.table import Table

DEV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEV)
from exposuresFromHealpixList import _radec_from_table  # noqa: E402
from skyproj_map import plot_skymap  # noqa: E402

EXPOSURES = "/home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5"
SKIMS = "/data8/shared/decampm/"
NSIDE, RADIUS = 512, np.radians(1.0)


def main():
    skims = {int(re.search(r"D(\d+)_", os.path.basename(f)).group(1))
             for b in "GRIZ" for f in glob.glob(f"{SKIMS}{b}/D*_{b.lower()}_cat.fits")}
    en, ra, dec = _radec_from_table(Table.read(EXPOSURES, path="__astropy_table__"))
    k = np.isin(en, list(skims))
    counts = np.zeros(hp.nside2npix(NSIDE), dtype=np.int64)
    for v in hp.ang2vec(ra[k], dec[k], lonlat=True):
        counts[hp.query_disc(NSIDE, v, RADIUS, inclusive=True)] += 1
    np.save(os.path.join(DEV, "skims_disc_counts_nside512.npy"), counts)
    pix = np.flatnonzero(counts)
    print(k.sum(), "visits;", pix.size, "pixels covered; max", counts.max(), "median", np.median(counts[pix]))
    plot_skymap(pixels=pix, values=counts[pix], nside=NSIDE, norm="log", vmin=1, vmax=counts.max(), xsize=4000,
                dpi=600, label="visits overlapping pixel (1° radius)", out=os.path.join(DEV, "skims_disc_skymap.png"))


if __name__ == "__main__":
    main()
