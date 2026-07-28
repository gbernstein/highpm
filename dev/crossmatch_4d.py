"""Cross-match two FITS tables on 4 values (e.g. ra, dec, pmra, pmdec).

Fill in the placeholders below, then run: python crossmatch_4d.py
"""
import sys

import fitsio
import numpy as np
from scipy.spatial import KDTree

sys.path.append("/home/vwetzell/gitrepos/highpm")
from highpm.gnomonic_converter import projectGnomonic

TABLE1_FILE = "/data8/shared/decampm/R/D00485012_r_cat.fits"
TABLE2_FILE = "/data8/shared/decampm/prev_des/r/gpr_0485012_r.fits"

TABLE1_COL1 = "ALPHAWIN_J2000"
TABLE1_COL2 = "DELTAWIN_J2000"
TABLE1_COL3 = "XWIN_IMAGE"
TABLE1_COL4 = "YWIN_IMAGE"

# A column spec is either "COLNAME" or ("COLNAME", index) for an (N, 2)-shaped column.
TABLE2_COL1 = ("new_rd", 0)
TABLE2_COL2 = ("new_rd", 1)
TABLE2_COL3 = ("xieta", 0)
TABLE2_COL4 = ("xieta", 1)

MATCH_TOLERANCE = 1.0  # placeholder, arcsec (all 4 values are put on this scale)
COL34_SCALE = 0.264  # table1 col3/col4 (pixels) -> arcsec


def get_col(table, spec):
    return table[spec] if isinstance(spec, str) else table[spec[0]][:, spec[1]]


def crossmatch_4d(pts1, pts2, tolerance):
    """Nearest neighbor in pts2 for each row of pts1; returns (matched_mask, idx_in_pts2, dist)."""
    tree = KDTree(pts2)
    dist, idx = tree.query(pts1)
    return dist < tolerance, idx, dist


if __name__ == "__main__":
    if "--test" in sys.argv:
        pts1 = np.array([[0, 0, 0, 0], [10, 10, 10, 10]])
        pts2 = np.array([[0.01, 0, 0, 0], [5, 5, 5, 5]])
        matched, idx, _ = crossmatch_4d(pts1, pts2, tolerance=1.0)
        assert matched.tolist() == [True, False]
        assert idx[0] == 0
        print("self-test ok")
        sys.exit()

    t1 = fitsio.read(TABLE1_FILE)
    t2 = fitsio.read(TABLE2_FILE)

    ra1, dec1 = t1[TABLE1_COL1], t1[TABLE1_COL2]
    ra2, dec2 = get_col(t2, TABLE2_COL1), get_col(t2, TABLE2_COL2)
    ra0, dec0 = np.mean(ra1), np.mean(dec1)

    xi1, eta1, *_ = projectGnomonic(ra1, dec1, np.zeros_like(ra1), np.zeros_like(ra1), ra0, dec0)
    xi2, eta2, *_ = projectGnomonic(ra2, dec2, np.zeros_like(ra2), np.zeros_like(ra2), ra0, dec0)

    pts1 = np.vstack([xi1 * 3600, eta1 * 3600, t1[TABLE1_COL3] * COL34_SCALE, t1[TABLE1_COL4] * COL34_SCALE]).T
    pts2 = np.vstack([xi2 * 3600, eta2 * 3600, get_col(t2, TABLE2_COL3), get_col(t2, TABLE2_COL4)]).T

    matched, idx, dist = crossmatch_4d(pts1, pts2, MATCH_TOLERANCE)

    print(f"Table1 rows: {len(pts1)}")
    print(f"Table2 rows: {len(pts2)}")
    print(f"Nearest-neighbor 4D distance (arcsec): min={dist.min():.3g} median={np.median(dist):.3g}")
    print(f"Matched (< {MATCH_TOLERANCE} arcsec): {matched.sum()}")

    if matched.sum() == 0:
        sys.exit()

    matched_t1 = t1[matched]
    matched_t2 = t2[idx[matched]]
    matched_pts1 = pts1[matched]
    matched_pts2 = pts2[idx[matched]]

    rms_12 = np.sqrt(np.mean(np.sum((matched_pts1[:, :2] - matched_pts2[:, :2]) ** 2, axis=1)))
    rms_34 = np.sqrt(np.mean(np.sum((matched_pts1[:, 2:] - matched_pts2[:, 2:]) ** 2, axis=1)))

    print(f"RMS separation (col1, col2): {rms_12}")
    print(f"RMS separation (col3, col4): {rms_34}")
