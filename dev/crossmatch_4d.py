"""Cross-match two FITS tables on 4 values (e.g. ra, dec, pmra, pmdec).

Fill in the placeholders below, then run: python crossmatch_4d.py
"""
import sys

import fitsio
import numpy as np
from scipy.spatial import KDTree

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

MATCH_TOLERANCE = 1.0  # placeholder, same units/scale as the 4 values


def get_col(table, spec):
    return table[spec] if isinstance(spec, str) else table[spec[0]][:, spec[1]]


def crossmatch_4d(pts1, pts2, tolerance):
    """Nearest neighbor in pts2 for each row of pts1; returns (matched_mask, idx_in_pts2)."""
    tree = KDTree(pts2)
    dist, idx = tree.query(pts1)
    return dist < tolerance, idx


if __name__ == "__main__":
    if "--test" in sys.argv:
        pts1 = np.array([[0, 0, 0, 0], [10, 10, 10, 10]])
        pts2 = np.array([[0.01, 0, 0, 0], [5, 5, 5, 5]])
        matched, idx = crossmatch_4d(pts1, pts2, tolerance=1.0)
        assert matched.tolist() == [True, False]
        assert idx[0] == 0
        print("self-test ok")
        sys.exit()

    t1 = fitsio.read(TABLE1_FILE)
    t2 = fitsio.read(TABLE2_FILE)

    pts1 = np.vstack([t1[TABLE1_COL1], t1[TABLE1_COL2], t1[TABLE1_COL3], t1[TABLE1_COL4]]).T
    pts2 = np.vstack(
        [get_col(t2, TABLE2_COL1), get_col(t2, TABLE2_COL2), get_col(t2, TABLE2_COL3), get_col(t2, TABLE2_COL4)]
    ).T

    matched, idx = crossmatch_4d(pts1, pts2, MATCH_TOLERANCE)

    matched_t1 = t1[matched]
    matched_t2 = t2[idx[matched]]
    matched_pts1 = pts1[matched]
    matched_pts2 = pts2[idx[matched]]

    rms_12 = np.sqrt(np.mean(np.sum((matched_pts1[:, :2] - matched_pts2[:, :2]) ** 2, axis=1)))
    rms_34 = np.sqrt(np.mean(np.sum((matched_pts1[:, 2:] - matched_pts2[:, 2:]) ** 2, axis=1)))

    print(f"Table1 rows: {len(pts1)}")
    print(f"Table2 rows: {len(pts2)}")
    print(f"Matched: {matched.sum()}")
    print(f"RMS separation (col1, col2): {rms_12}")
    print(f"RMS separation (col3, col4): {rms_34}")
